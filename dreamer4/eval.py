#!/usr/bin/env python3
"""
eval.py -- Standalone dynamics evaluation for Dreamer4.

Computes dynamics-focused metrics (latent-space + pixel-space) and generates
report-ready figures.

Usage:
    python eval.py \
        --dynamics_ckpt ./logs/dynamics_ckpts/step_0095000.pt \
        --tokenizer_ckpt ./logs/tokenizer_ckpts/latest.pt \
        --output_dir ./eval_output
"""

import os
import sys
import json
import math
import glob as glob_mod
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from task_set import TASK_SET
from wm_dataset import WMDataset, collate_batch
from model import (
    Encoder, Decoder, Dynamics,
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
)
from train_dynamics import (
    load_frozen_tokenizer_from_pt_ckpt,
    make_tau_schedule,
    sample_autoregressive_packed_sequence,
    decode_packed_to_frames,
    dynamics_pretrain_loss,
    seed_everything,
)

import lpips

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# SSIM (pure torch)
# ============================================================
def _gaussian_window(window_size: int, sigma: float, channels: int,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return w2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor,
                 window_size: int = 11) -> torch.Tensor:
    """SSIM between two (B,C,H,W) tensors in [0,1]. Returns per-image scalar."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = img1.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=(1, 2, 3))  # (B,)


# ============================================================
# Model loading
# ============================================================
def load_dynamics_from_ckpt(
    ckpt_path: str,
    tok_args: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dynamics, Dict[str, Any], int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dyn_args = ckpt["args"]
    step = int(ckpt.get("step", 0))

    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    packing_factor = int(dyn_args["packing_factor"])
    n_spatial = n_latents // packing_factor
    d_spatial = d_bottleneck * packing_factor

    dyn = Dynamics(
        d_model=int(dyn_args["d_model_dyn"]),
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=int(dyn_args["n_register"]),
        n_agent=int(dyn_args["n_agent"]),
        n_heads=int(dyn_args["n_heads"]),
        depth=int(dyn_args["dyn_depth"]),
        k_max=int(dyn_args["k_max"]),
        dropout=0.0,
        mlp_ratio=float(dyn_args["mlp_ratio"]),
        time_every=int(dyn_args["time_every"]),
        space_mode=str(dyn_args["space_mode"]),
    ).to(device)

    dyn.load_state_dict(ckpt["dynamics"], strict=True)
    dyn.eval()
    for p in dyn.parameters():
        p.requires_grad_(False)

    return dyn, dyn_args, step


# ============================================================
# Batch evaluation
# ============================================================
@torch.no_grad()
def evaluate_batch(
    encoder: Encoder,
    decoder: Decoder,
    dyn: Dynamics,
    frames: torch.Tensor,            # (B,T,3,H,W) float [0,1]
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    *,
    H: int, W: int, C: int, patch: int,
    packing_factor: int, k_max: int,
    ctx_length: int, horizon: int,
    sched: Dict[str, Any],
    lpips_fn,
) -> Dict[str, Any]:
    B, T = frames.shape[:2]
    T_eval = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, T_eval - 1)
    horizon = min(horizon, T_eval - ctx_length)

    frames_eval = frames[:, :T_eval]
    actions_eval = None if actions is None else actions[:, :T_eval]
    act_mask_eval = None if act_mask is None else (
        act_mask[:, :T_eval] if act_mask.dim() == 3 else act_mask
    )

    # Encode ground truth
    patches = temporal_patchify(frames_eval, patch)
    z_btLd, _ = encoder(patches)
    n_spatial = z_btLd.shape[2] // packing_factor
    z_gt_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)

    # Autoregressive rollout
    z_pred_packed = sample_autoregressive_packed_sequence(
        dyn, z_gt_packed=z_gt_packed,
        ctx_length=ctx_length, horizon=horizon,
        k_max=k_max, sched=sched,
        actions=actions_eval, act_mask=act_mask_eval,
    )

    # Decode predictions
    pred_frames = decode_packed_to_frames(
        decoder, z_packed=z_pred_packed,
        H=H, W=W, C=C, patch=patch, packing_factor=packing_factor,
    )

    # Horizon slices
    gt_h = frames_eval[:, ctx_length:ctx_length + horizon]
    pred_h = pred_frames[:, ctx_length:ctx_length + horizon]
    z_gt_h = z_gt_packed[:, ctx_length:ctx_length + horizon]
    z_pred_h = z_pred_packed[:, ctx_length:ctx_length + horizon]

    # Repeat-last-frame baseline
    floor_h = frames_eval[:, ctx_length - 1:ctx_length].expand(-1, horizon, -1, -1, -1)

    # --- Latent-space metrics ---
    # Per-timestep latent MSE: (Hz,)
    latent_mse_per_t = (z_pred_h.float() - z_gt_h.float()).pow(2).mean(dim=(0, 2, 3))
    # Per-timestep cosine similarity: (Hz,)
    z_pred_flat = z_pred_h.float().reshape(B, horizon, -1)
    z_gt_flat = z_gt_h.float().reshape(B, horizon, -1)
    cos_sim_per_t = F.cosine_similarity(z_pred_flat, z_gt_flat, dim=-1).mean(dim=0)
    # Per-spatial-token MSE: (n_spatial,)
    per_token_mse = (z_pred_h.float() - z_gt_h.float()).pow(2).mean(dim=(0, 1, 3))

    # --- Pixel-space metrics ---
    pixel_mse_per_t = (pred_h.float() - gt_h.float()).pow(2).mean(dim=(0, 2, 3, 4))
    psnr_per_t = 10.0 * torch.log10(1.0 / pixel_mse_per_t.clamp_min(1e-12))

    floor_mse_per_t = (floor_h.float() - gt_h.float()).pow(2).mean(dim=(0, 2, 3, 4))
    floor_psnr_per_t = 10.0 * torch.log10(1.0 / floor_mse_per_t.clamp_min(1e-12))

    # SSIM per timestep
    ssim_per_t = []
    floor_ssim_per_t = []
    for t in range(horizon):
        ssim_per_t.append(compute_ssim(pred_h[:, t], gt_h[:, t]).mean().item())
        floor_ssim_per_t.append(compute_ssim(floor_h[:, t], gt_h[:, t]).mean().item())

    # LPIPS per timestep
    lpips_per_t = []
    floor_lpips_per_t = []
    for t in range(horizon):
        lp = lpips_fn(pred_h[:, t] * 2 - 1, gt_h[:, t] * 2 - 1)
        lpips_per_t.append(lp.mean().item())
        lp_f = lpips_fn(floor_h[:, t] * 2 - 1, gt_h[:, t] * 2 - 1)
        floor_lpips_per_t.append(lp_f.mean().item())

    return {
        "latent_mse_per_t": latent_mse_per_t.cpu().tolist(),
        "cos_sim_per_t": cos_sim_per_t.cpu().tolist(),
        "per_token_mse": per_token_mse.cpu().tolist(),
        "pixel_mse_per_t": pixel_mse_per_t.cpu().tolist(),
        "psnr_per_t": psnr_per_t.cpu().tolist(),
        "ssim_per_t": ssim_per_t,
        "lpips_per_t": lpips_per_t,
        "floor_mse_per_t": floor_mse_per_t.cpu().tolist(),
        "floor_psnr_per_t": floor_psnr_per_t.cpu().tolist(),
        "floor_ssim_per_t": floor_ssim_per_t,
        "floor_lpips_per_t": floor_lpips_per_t,
        "horizon": horizon,
        "ctx_length": ctx_length,
        "B": B,
        # Keep sample frames for qualitative figure
        "gt_frames": frames_eval.cpu(),
        "pred_frames": pred_frames.cpu(),
    }


def aggregate_batch_metrics(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average per-timestep metrics across batches (weighted by B)."""
    keys_per_t = [
        "latent_mse_per_t", "cos_sim_per_t",
        "pixel_mse_per_t", "psnr_per_t", "ssim_per_t", "lpips_per_t",
        "floor_mse_per_t", "floor_psnr_per_t", "floor_ssim_per_t", "floor_lpips_per_t",
    ]
    total_B = sum(b["B"] for b in batch_list)
    horizon = batch_list[0]["horizon"]

    agg = {"horizon": horizon, "ctx_length": batch_list[0]["ctx_length"]}
    for key in keys_per_t:
        weighted = np.zeros(horizon)
        for b in batch_list:
            weighted += np.array(b[key]) * b["B"]
        agg[key] = (weighted / total_B).tolist()

    # Per-token MSE
    weighted_tok = np.zeros(len(batch_list[0]["per_token_mse"]))
    for b in batch_list:
        weighted_tok += np.array(b["per_token_mse"]) * b["B"]
    agg["per_token_mse"] = (weighted_tok / total_B).tolist()

    # Scalar aggregates
    for key in ["latent_mse_per_t", "pixel_mse_per_t", "psnr_per_t",
                "ssim_per_t", "lpips_per_t", "floor_mse_per_t", "floor_psnr_per_t"]:
        vals = np.array(agg[key])
        scalar_key = key.replace("_per_t", "")
        agg[scalar_key] = float(vals.mean())

    agg["cos_sim"] = float(np.mean(agg["cos_sim_per_t"]))
    agg["floor_ssim"] = float(np.mean(agg["floor_ssim_per_t"]))
    agg["floor_lpips"] = float(np.mean(agg["floor_lpips_per_t"]))

    return agg


# ============================================================
# Action shuffle metric
# ============================================================
@torch.no_grad()
def compute_action_shuffle_ratio(
    dyn: Dynamics,
    encoder: Encoder,
    frames: torch.Tensor,
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    *,
    patch: int, packing_factor: int, k_max: int,
) -> float:
    patches = temporal_patchify(frames, patch)
    z_btLd, _ = encoder(patches)
    n_spatial = z_btLd.shape[2] // packing_factor
    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)

    loss_real, _ = dynamics_pretrain_loss(
        dyn, z1=z1, actions=actions, act_mask=act_mask,
        k_max=k_max, B_self=0, step=100000, bootstrap_start=100001,
    )
    perm = torch.randperm(actions.shape[0], device=actions.device)
    loss_shuffled, _ = dynamics_pretrain_loss(
        dyn, z1=z1, actions=actions[perm], act_mask=act_mask,
        k_max=k_max, B_self=0, step=100000, bootstrap_start=100001,
    )
    ratio = float((loss_shuffled / loss_real.clamp_min(1e-12)).item())
    return ratio


# ============================================================
# Full evaluation
# ============================================================
def evaluate_checkpoint(
    dynamics_ckpt_path: str,
    tokenizer_ckpt_path: str,
    data_dirs: List[str],
    frame_dirs: List[str],
    tasks_json: str,
    *,
    eval_ctx: int = 8,
    eval_horizon: int = 16,
    eval_schedule: str = "shortcut",
    eval_d: float = 0.25,
    num_seqs_per_task: int = 16,
    batch_size: int = 4,
    device: torch.device,
    seed: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate one checkpoint. Returns (results_dict, sample_data_for_figures)."""
    seed_everything(seed)

    # Load models
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        tokenizer_ckpt_path, device=device,
    )
    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))

    dyn, dyn_args, ckpt_step = load_dynamics_from_ckpt(
        dynamics_ckpt_path, tok_args, device,
    )
    packing_factor = int(dyn_args["packing_factor"])
    k_max = int(dyn_args["k_max"])

    sched = make_tau_schedule(k_max=k_max, schedule=eval_schedule, d=eval_d)

    # LPIPS
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # Dataset
    seq_len = eval_ctx + eval_horizon
    dataset = WMDataset(
        data_dir=data_dirs,
        frames_dir=frame_dirs,
        seq_len=seq_len,
        img_size=128,
        action_dim=16,
        tasks_json=tasks_json,
        tasks=TASK_SET,
        verbose=True,
        strict_tasks=False,
    )

    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint: {dynamics_ckpt_path} (step {ckpt_step})")
    print(f"Tasks: {len(dataset.tasks)}, Schedule: {eval_schedule} (d={eval_d}, K={sched['K']})")
    print(f"Context: {eval_ctx}, Horizon: {eval_horizon}, Seqs/task: {num_seqs_per_task}")
    print(f"{'='*60}\n")

    results = {
        "checkpoint": str(dynamics_ckpt_path),
        "step": ckpt_step,
        "per_task": {},
        "per_timestep": {},
    }
    sample_data = {}  # for qualitative figures

    rng = np.random.default_rng(seed)

    # Action shuffle: collect a batch for the metric
    action_shuffle_ratios = []

    for task_idx, task_name in enumerate(dataset.tasks):
        prev = 0 if task_idx == 0 else dataset._cum_counts[task_idx - 1]
        count = dataset._cum_counts[task_idx] - prev

        n_eval = min(num_seqs_per_task, count)
        sample_local = rng.choice(count, size=n_eval, replace=False)

        batch_metrics = []

        for batch_start in range(0, n_eval, batch_size):
            batch_end = min(batch_start + batch_size, n_eval)
            batch_indices = sample_local[batch_start:batch_end]
            global_indices = batch_indices + prev

            items = [dataset[int(gi)] for gi in global_indices]
            batch = collate_batch(items)

            obs_u8 = batch["obs"].to(device)
            act = batch["act"].to(device)
            mask = batch["act_mask"].to(device)

            act = act.clamp(-1, 1) * mask
            frames = obs_u8[:, :-1].float() / 255.0
            actions = torch.zeros_like(act)
            actions[:, 1:] = act[:, :-1]
            act_mask_shifted = torch.zeros_like(mask)
            act_mask_shifted[:, 1:] = mask[:, :-1]

            metrics = evaluate_batch(
                encoder, decoder, dyn, frames, actions, act_mask_shifted,
                H=H, W=W, C=C, patch=patch,
                packing_factor=packing_factor, k_max=k_max,
                ctx_length=eval_ctx, horizon=eval_horizon,
                sched=sched, lpips_fn=lpips_fn,
            )
            batch_metrics.append(metrics)

            # Action shuffle on first batch of first few tasks
            if batch_start == 0 and task_idx < 10:
                ratio = compute_action_shuffle_ratio(
                    dyn, encoder, frames, actions, act_mask_shifted,
                    patch=patch, packing_factor=packing_factor, k_max=k_max,
                )
                action_shuffle_ratios.append(ratio)

            # Save sample for qualitative figure (first batch of select tasks)
            if batch_start == 0 and task_name not in sample_data:
                sample_data[task_name] = {
                    "gt_frames": metrics["gt_frames"][:2],
                    "pred_frames": metrics["pred_frames"][:2],
                    "ctx_length": metrics["ctx_length"],
                    "horizon": metrics["horizon"],
                }

        task_agg = aggregate_batch_metrics(batch_metrics)
        results["per_task"][task_name] = task_agg

        print(f"  {task_name:30s} | Latent MSE={task_agg['latent_mse']:.5f} "
              f"| PSNR={task_agg['psnr']:.2f} dB | SSIM={task_agg['ssim']:.4f} "
              f"| LPIPS={task_agg['lpips']:.4f}")

    # Aggregate across tasks
    all_tasks_metrics = list(results["per_task"].values())
    agg = {}
    for key in all_tasks_metrics[0]:
        vals = [m[key] for m in all_tasks_metrics]
        if isinstance(vals[0], list):
            agg[key] = np.mean(vals, axis=0).tolist()
        elif isinstance(vals[0], (int, float)):
            agg[key] = float(np.mean(vals))
    results["aggregate"] = agg
    results["aggregate"]["action_shuffle_ratio"] = float(np.mean(action_shuffle_ratios)) if action_shuffle_ratios else 0.0

    # Also store the per-timestep from aggregate for easy access
    results["per_timestep"] = {
        k: agg[k] for k in agg if k.endswith("_per_t") or k == "per_token_mse"
    }

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS (step {ckpt_step})")
    print(f"  Latent MSE:           {agg.get('latent_mse', 0):.6f}")
    print(f"  Latent Cosine Sim:    {agg.get('cos_sim', 0):.4f}")
    print(f"  Pixel MSE:            {agg.get('pixel_mse', 0):.6f}")
    print(f"  PSNR:                 {agg.get('psnr', 0):.2f} dB")
    print(f"  SSIM:                 {agg.get('ssim', 0):.4f}")
    print(f"  LPIPS:                {agg.get('lpips', 0):.4f}")
    print(f"  Floor PSNR:           {agg.get('floor_psnr', 0):.2f} dB")
    print(f"  PSNR Gain:            {agg.get('psnr', 0) - agg.get('floor_psnr', 0):.2f} dB")
    print(f"  Action Shuffle Ratio: {results['aggregate']['action_shuffle_ratio']:.3f}")
    print(f"  Per-token MSE:        {agg.get('per_token_mse', [])}")
    print(f"{'='*60}\n")

    return results, sample_data


# ============================================================
# Figure generation
# ============================================================
def plot_horizon_curves(results: Dict, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = results["aggregate"]
    horizon = int(agg["horizon"])
    ts = list(range(1, horizon + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Latent MSE
    ax = axes[0, 0]
    ax.plot(ts, agg["latent_mse_per_t"], "o-", label="Model", color="#2196F3", markersize=4)
    ax.set_xlabel("Horizon Step")
    ax.set_ylabel("Latent MSE")
    ax.set_title("Latent-Space MSE over Horizon")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Pixel MSE
    ax = axes[0, 1]
    ax.plot(ts, agg["pixel_mse_per_t"], "o-", label="Model", color="#2196F3", markersize=4)
    ax.plot(ts, agg["floor_mse_per_t"], "s--", label="Repeat-last", color="#FF9800", markersize=4)
    ax.set_xlabel("Horizon Step")
    ax.set_ylabel("Pixel MSE")
    ax.set_title("Pixel-Space MSE over Horizon")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # PSNR
    ax = axes[1, 0]
    ax.plot(ts, agg["psnr_per_t"], "o-", label="Model", color="#2196F3", markersize=4)
    ax.plot(ts, agg["floor_psnr_per_t"], "s--", label="Repeat-last", color="#FF9800", markersize=4)
    ax.set_xlabel("Horizon Step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR over Horizon")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # SSIM
    ax = axes[1, 1]
    ax.plot(ts, agg["ssim_per_t"], "o-", label="Model", color="#2196F3", markersize=4)
    ax.plot(ts, agg["floor_ssim_per_t"], "s--", label="Repeat-last", color="#FF9800", markersize=4)
    ax.set_xlabel("Horizon Step")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM over Horizon")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"Dynamics Evaluation — Step {results['step']}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "horizon_curves.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_qualitative_grid(
    sample_data: Dict[str, Any],
    output_dir: Path,
    select_tasks: Optional[List[str]] = None,
    step: int = 0,
):
    from PIL import Image

    if select_tasks is None:
        preferred = ["walker-walk", "cheetah-run", "cup-catch", "cartpole-swingup",
                      "acrobot-swingup", "hopper-hop", "reacher-easy", "finger-spin"]
        select_tasks = [t for t in preferred if t in sample_data][:4]
        if len(select_tasks) < 4:
            remaining = [t for t in sample_data if t not in select_tasks]
            select_tasks += remaining[:4 - len(select_tasks)]

    if not select_tasks:
        print("  No tasks available for qualitative grid.")
        return

    gap_px = 8
    rows = []

    for task_name in select_tasks:
        data = sample_data[task_name]
        gt = data["gt_frames"][0]      # (T, 3, H, W)
        pred = data["pred_frames"][0]
        ctx = data["ctx_length"]
        T, C_img, H_img, W_img = gt.shape

        # Select frames: context[-2,-1] + horizon[0,3,7,11,15]
        ctx_indices = list(range(max(0, ctx - 2), ctx))
        hz_indices = [ctx + i for i in [0, 3, 7, 11, 15] if ctx + i < T]
        indices = ctx_indices + hz_indices

        def make_row(frames, indices, is_pred=False):
            imgs = []
            for i in indices:
                img = (frames[i].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                imgs.append(img)
            # Add separator between context and horizon
            return imgs

        gt_imgs = make_row(gt, indices)
        pred_imgs = make_row(pred, indices, is_pred=True)

        sep_idx = len(ctx_indices)

        def concat_with_sep(imgs, sep_idx):
            parts = []
            for i, img in enumerate(imgs):
                if i == sep_idx and i > 0:
                    sep = np.ones((img.shape[0], gap_px, 3), dtype=np.uint8) * 200
                    parts.append(sep)
                parts.append(img)
            return np.concatenate(parts, axis=1)

        gt_strip = concat_with_sep(gt_imgs, sep_idx)
        pred_strip = concat_with_sep(pred_imgs, sep_idx)

        # Add row separator
        row_sep = np.ones((2, gt_strip.shape[1], 3), dtype=np.uint8) * 128
        rows.append(gt_strip)
        rows.append(pred_strip)
        rows.append(np.ones((gap_px, gt_strip.shape[1], 3), dtype=np.uint8) * 255)

    if rows:
        rows = rows[:-1]  # remove last gap
    panel = np.concatenate(rows, axis=0)

    path = output_dir / "qualitative_grid.png"
    Image.fromarray(panel).save(str(path))
    print(f"  Saved {path}")


def generate_videos(
    sample_data: Dict[str, Any],
    output_dir: Path,
    select_tasks: Optional[List[str]] = None,
    fps: int = 4,
):
    """Generate side-by-side GT vs Predicted GIF videos for each task."""
    from PIL import Image, ImageDraw, ImageFont

    if select_tasks is None:
        preferred = ["walker-walk", "cheetah-run", "cup-catch", "cartpole-swingup",
                      "acrobot-swingup", "hopper-hop", "reacher-easy", "finger-spin"]
        select_tasks = [t for t in preferred if t in sample_data][:6]
        if len(select_tasks) < 4:
            remaining = [t for t in sample_data if t not in select_tasks]
            select_tasks += remaining[:6 - len(select_tasks)]

    vid_dir = output_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    for task_name in select_tasks:
        data = sample_data[task_name]
        gt = data["gt_frames"][0]      # (T, 3, H, W)
        pred = data["pred_frames"][0]
        ctx = data["ctx_length"]
        T = gt.shape[0]

        frames_pil = []
        for t in range(T):
            gt_img = (gt[t].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            pred_img = (pred[t].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            H_img, W_img = gt_img.shape[:2]

            # Side-by-side: GT | gap | Pred
            gap = 4
            label_h = 20
            canvas_w = W_img * 2 + gap
            canvas_h = H_img + label_h
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

            # Place images
            canvas[label_h:label_h + H_img, :W_img] = gt_img
            canvas[label_h:label_h + H_img, W_img + gap:] = pred_img

            pil_frame = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil_frame)

            # Labels
            phase = "context" if t < ctx else "predicted"
            step_label = f"t={t}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except OSError:
                font = ImageFont.load_default()
                font_small = font

            draw.text((2, 2), f"GT ({step_label})", fill=(0, 0, 0), font=font_small)
            draw.text((W_img + gap + 2, 2), f"Pred ({step_label})", fill=(0, 0, 0), font=font_small)

            # Red border on predicted horizon frames
            if t >= ctx:
                draw.rectangle(
                    [W_img + gap, label_h, canvas_w - 1, canvas_h - 1],
                    outline=(255, 80, 80), width=2,
                )
            # Blue border on context frames
            else:
                draw.rectangle(
                    [0, label_h, W_img - 1, canvas_h - 1],
                    outline=(80, 80, 255), width=2,
                )
                draw.rectangle(
                    [W_img + gap, label_h, canvas_w - 1, canvas_h - 1],
                    outline=(80, 80, 255), width=2,
                )

            frames_pil.append(pil_frame)

        # Save GIF
        gif_path = vid_dir / f"{task_name}.gif"
        frames_pil[0].save(
            str(gif_path),
            save_all=True,
            append_images=frames_pil[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"  Saved {gif_path}")

    print(f"  Videos saved to {vid_dir}/")


def plot_per_task_bars(results: Dict, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_task = results["per_task"]
    tasks = sorted(per_task.keys(), key=lambda t: per_task[t].get("psnr", 0), reverse=True)

    model_psnr = [per_task[t].get("psnr", 0) for t in tasks]
    floor_psnr = [per_task[t].get("floor_psnr", 0) for t in tasks]
    model_lpips = [per_task[t].get("lpips", 0) for t in tasks]
    floor_lpips = [per_task[t].get("floor_lpips", 0) for t in tasks]

    x = np.arange(len(tasks))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # PSNR
    ax1.bar(x - w/2, model_psnr, w, label="Model", color="#2196F3", alpha=0.85)
    ax1.bar(x + w/2, floor_psnr, w, label="Repeat-last", color="#FF9800", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Per-Task PSNR (higher is better)")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # LPIPS
    ax2.bar(x - w/2, model_lpips, w, label="Model", color="#2196F3", alpha=0.85)
    ax2.bar(x + w/2, floor_lpips, w, label="Repeat-last", color="#FF9800", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("LPIPS")
    ax2.set_title("Per-Task LPIPS (lower is better)")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Per-Task Dynamics Evaluation — Step {results['step']}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "per_task_bars.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_training_progression(
    all_results: List[Tuple[int, Dict]],
    output_dir: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_results = sorted(all_results, key=lambda x: x[0])
    steps = [s for s, _ in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    def get_vals(key):
        return [r["aggregate"].get(key, 0) for _, r in all_results]

    # Latent MSE
    ax = axes[0, 0]
    ax.plot(steps, get_vals("latent_mse"), "o-", color="#2196F3", markersize=5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Latent MSE")
    ax.set_title("Latent MSE vs Training Step")
    ax.grid(True, alpha=0.3)

    # PSNR
    ax = axes[0, 1]
    ax.plot(steps, get_vals("psnr"), "o-", color="#4CAF50", markersize=5, label="Model")
    ax.plot(steps, get_vals("floor_psnr"), "s--", color="#FF9800", markersize=5, label="Repeat-last")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs Training Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # LPIPS
    ax = axes[1, 0]
    ax.plot(steps, get_vals("lpips"), "o-", color="#9C27B0", markersize=5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("LPIPS")
    ax.set_title("LPIPS vs Training Step")
    ax.grid(True, alpha=0.3)

    # Action shuffle ratio
    ax = axes[1, 1]
    ratios = [r["aggregate"].get("action_shuffle_ratio", 0) for _, r in all_results]
    ax.plot(steps, ratios, "o-", color="#F44336", markersize=5)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss Ratio (shuffled / real)")
    ax.set_title("Action Shuffle Ratio vs Training Step")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training Progression", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "training_progression.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Dreamer4 dynamics evaluation")

    p.add_argument("--dynamics_ckpt", type=str, required=True,
                   help="Path to dynamics checkpoint .pt file")
    p.add_argument("--tokenizer_ckpt", type=str, required=True,
                   help="Path to tokenizer checkpoint .pt file")

    p.add_argument("--data_dirs", type=str, nargs="+", default=["../data/expert"])
    p.add_argument("--frame_dirs", type=str, nargs="+", default=["../data/expert-shards"])
    p.add_argument("--tasks_json", type=str, default="../tasks.json")

    p.add_argument("--eval_ctx", type=int, default=8)
    p.add_argument("--eval_horizon", type=int, default=16)
    p.add_argument("--eval_schedule", type=str, default="shortcut",
                   choices=["shortcut", "finest"])
    p.add_argument("--eval_d", type=float, default=0.25)
    p.add_argument("--num_seqs_per_task", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)

    p.add_argument("--output_dir", type=str, default="./eval_output")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--all_ckpts", action="store_true",
                   help="Also evaluate all step_*.pt checkpoints for training progression plot")
    p.add_argument("--progression_seqs", type=int, default=4,
                   help="Seqs per task for progression (fewer for speed)")
    p.add_argument("--progression_tasks", type=int, default=5,
                   help="Number of tasks to use for progression (fewer for speed)")

    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t0 = time.time()

    # --- Main checkpoint evaluation ---
    results, sample_data = evaluate_checkpoint(
        dynamics_ckpt_path=args.dynamics_ckpt,
        tokenizer_ckpt_path=args.tokenizer_ckpt,
        data_dirs=args.data_dirs,
        frame_dirs=args.frame_dirs,
        tasks_json=args.tasks_json,
        eval_ctx=args.eval_ctx,
        eval_horizon=args.eval_horizon,
        eval_schedule=args.eval_schedule,
        eval_d=args.eval_d,
        num_seqs_per_task=args.num_seqs_per_task,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )

    # Save JSON
    json_path = output_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {json_path}")

    # Generate figures into subfolder
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating figures...")
    plot_horizon_curves(results, fig_dir)
    plot_qualitative_grid(sample_data, fig_dir, step=results["step"])
    plot_per_task_bars(results, fig_dir)

    # Generate videos
    print("\nGenerating videos...")
    generate_videos(sample_data, fig_dir)

    # --- Training progression (optional) ---
    if args.all_ckpts:
        ckpt_dir = Path(args.dynamics_ckpt).parent
        all_ckpt_paths = sorted(glob_mod.glob(str(ckpt_dir / "step_*.pt")))
        # Exclude latest.pt which is a duplicate
        all_ckpt_paths = [p for p in all_ckpt_paths if "latest" not in Path(p).name]

        if len(all_ckpt_paths) > 10:
            # Sample ~8 evenly spaced checkpoints
            indices = np.linspace(0, len(all_ckpt_paths) - 1, 8, dtype=int)
            all_ckpt_paths = [all_ckpt_paths[i] for i in indices]

        print(f"\nTraining progression: evaluating {len(all_ckpt_paths)} checkpoints...")

        # Use a subset of tasks for speed
        prog_tasks = TASK_SET[:args.progression_tasks]

        all_prog_results = []
        for ckpt_path in all_ckpt_paths:
            try:
                prog_result, _ = evaluate_checkpoint(
                    dynamics_ckpt_path=ckpt_path,
                    tokenizer_ckpt_path=args.tokenizer_ckpt,
                    data_dirs=args.data_dirs,
                    frame_dirs=args.frame_dirs,
                    tasks_json=args.tasks_json,
                    eval_ctx=args.eval_ctx,
                    eval_horizon=args.eval_horizon,
                    eval_schedule=args.eval_schedule,
                    eval_d=args.eval_d,
                    num_seqs_per_task=args.progression_seqs,
                    batch_size=args.batch_size,
                    device=device,
                    seed=args.seed,
                )
                all_prog_results.append((prog_result["step"], prog_result))
            except Exception as e:
                print(f"  Skipping {ckpt_path}: {e}")

        if all_prog_results:
            plot_training_progression(all_prog_results, fig_dir)

            prog_json_path = output_dir / "progression_results.json"
            prog_data = [(s, r) for s, r in all_prog_results]
            with open(prog_json_path, "w") as f:
                json.dump(prog_data, f, indent=2)
            print(f"Saved progression metrics to {prog_json_path}")

    elapsed = time.time() - t0
    print(f"\nTotal evaluation time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
