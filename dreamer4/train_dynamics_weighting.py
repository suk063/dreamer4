# train_dynamics_weighting.py
# Reward Gradient Saliency-Weighted Dynamics Training for Dreamer4
# Based on train_dynamics.py with pixel-level importance weighting
import os
import time
import math
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler

import wandb

from task_set import TASK_SET
from sharded_frame_dataset import ShardedFrameDataset

from model import (
    Encoder, Decoder, Tokenizer,
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck,
    Dynamics,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Reward prediction head for saliency computation
# ---------------------------------------------------------------------------

class RewardHead(nn.Module):
    """Auxiliary reward predictor. Takes mean-pooled spatial tokens, predicts scalar reward."""

    def __init__(self, d_input: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return self.net(x).squeeze(-1)  # (B, T)


# ---------------------------------------------------------------------------
# Saliency utilities (spatial-token level — memory-efficient)
# ---------------------------------------------------------------------------

def compute_spatial_saliency(
    reward_head: RewardHead,
    z1: torch.Tensor,               # (B, T, Sz, Dz) detached packed spatial tokens
    *,
    delta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-spatial-token importance weights via reward gradient saliency.
    Only backward through the lightweight RewardHead MLP — negligible memory overhead.

    Returns:
        saliency: (B, T, Sz) detached weights in [delta, 1]
        r_hat:    (B, T)     reward predictions (detached)
    """
    z1_sal = z1.detach().requires_grad_(True)                    # (B,T,Sz,Dz)

    # Reward prediction from mean-pooled spatial tokens
    r_hat = reward_head(z1_sal.mean(dim=2))                      # (B, T)

    # Gradient of reward prediction w.r.t. spatial tokens
    grad = torch.autograd.grad(r_hat.sum(), z1_sal, create_graph=False)[0]  # (B,T,Sz,Dz)

    # Per-token saliency: aggregate over feature dimension
    sal = grad.abs().mean(dim=-1)                                # (B, T, Sz)

    # Min-max normalize per (batch, time) to [delta, 1.0]
    sal_min = sal.amin(dim=-1, keepdim=True)                     # (B, T, 1)
    sal_max = sal.amax(dim=-1, keepdim=True)                     # (B, T, 1)
    sal_range = (sal_max - sal_min).clamp_min(1e-8)
    sal_norm = (sal - sal_min) / sal_range
    sal_norm = sal_norm.clamp(min=delta)

    return sal_norm.detach(), r_hat.detach()


# ---------------------------------------------------------------------------
# Utilities (same as train_dynamics.py)
# ---------------------------------------------------------------------------

def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def seed_everything(seed: int):
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    seed_everything(info.seed)


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return ddp, rank, world_size, local_rank


def save_ckpt(path: Path, *, step: int, epoch: int, dyn_model, reward_head, opt, scaler, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "epoch": epoch,
        "dynamics": (dyn_model.module.state_dict() if hasattr(dyn_model, "module") else dyn_model.state_dict()),
        "reward_head": reward_head.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, *, dyn_model, reward_head, opt, scaler) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["dynamics"]
    (dyn_model.module if hasattr(dyn_model, "module") else dyn_model).load_state_dict(state, strict=True)
    if "reward_head" in ckpt:
        reward_head.load_state_dict(ckpt["reward_head"], strict=True)
    opt.load_state_dict(ckpt["opt"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


@torch.no_grad()
def load_frozen_tokenizer_from_pt_ckpt(
    ckpt_path: str,
    *,
    device: torch.device,
    override: Optional[Dict[str, Any]] = None,
) -> tuple[Encoder, Decoder, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tok_args = dict(ckpt.get("args", {}))
    if override:
        tok_args.update(override)

    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C

    enc = Encoder(
        patch_dim=d_patch,
        d_model=int(tok_args.get("d_model", 256)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_patches=n_patches,
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        dropout=0.0,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        time_every=int(tok_args.get("time_every", 1)),
        latents_only_time=bool(tok_args.get("latents_only_time", True)),
        mae_p_min=0.0,
        mae_p_max=0.0,
    )
    dec = Decoder(
        d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        d_model=int(tok_args.get("d_model", 256)),
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=0.0,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        time_every=int(tok_args.get("time_every", 1)),
        latents_only_time=bool(tok_args.get("latents_only_time", True)),
    )

    tok = Tokenizer(enc, dec)
    tok.load_state_dict(ckpt["model"], strict=True)

    tok = tok.to(device)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    return tok.encoder, tok.decoder, tok_args


# ---------------------------------------------------------------------------
# Shortcut forcing helpers (same as train_dynamics.py)
# ---------------------------------------------------------------------------

def _emax_from_kmax(k_max: int) -> int:
    emax = int(round(math.log2(k_max)))
    assert (1 << emax) == k_max, "k_max must be power of two"
    return emax


def _sample_step_excluding_dmin(device: torch.device, B: int, T: int, k_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    emax = _emax_from_kmax(k_max)
    step_idx = torch.randint(low=0, high=max(1, emax), size=(B, T), device=device, dtype=torch.long)
    d = 1.0 / (1 << step_idx).to(torch.float32)
    return d, step_idx


def _sample_tau_for_step(device: torch.device, B: int, T: int, k_max: int, step_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    K = (1 << step_idx).to(torch.long)
    u = torch.rand((B, T), device=device, dtype=torch.float32)
    j_idx = torch.floor(u * K.to(torch.float32)).to(torch.long)
    tau = j_idx.to(torch.float32) / K.to(torch.float32)
    scale = torch.div(torch.tensor(k_max, device=device), K, rounding_mode="floor")
    tau_idx = j_idx * scale
    return tau, tau_idx


def dynamics_pretrain_loss(
    dynamics: torch.nn.Module,
    *,
    z1: torch.Tensor,                    # (B,T,Sz,Dz) packed clean targets
    actions: Optional[torch.Tensor],     # (B,T) or None
    act_mask: Optional[torch.Tensor],    # (A,) or None
    k_max: int,
    B_self: int,
    step: int,
    bootstrap_start: int,
    agent_tokens: Optional[torch.Tensor] = None,
    spatial_weights: Optional[torch.Tensor] = None,  # (B,T,Sz) per-token saliency weights
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Shortcut forcing loss for dynamics model.
    When spatial_weights is provided, the per-token MSE is weighted by saliency
    instead of uniform averaging over the spatial dimension.
    """
    device = z1.device
    B, T = z1.shape[:2]
    assert 0 <= B_self < B
    B_emp = B - B_self
    emax = _emax_from_kmax(k_max)

    act_mask_full = act_mask
    act_mask_self = None if act_mask_full is None else act_mask_full[B_emp:]

    step_idx_emp = torch.full((B_emp, T), emax, device=device, dtype=torch.long)
    if B_self > 0:
        d_self, step_idx_self = _sample_step_excluding_dmin(device, B_self, T, k_max)
        step_idx_full = torch.cat([step_idx_emp, step_idx_self], dim=0)
    else:
        d_self = torch.zeros((0, T), device=device, dtype=torch.float32)
        step_idx_self = torch.zeros((0, T), device=device, dtype=torch.long)
        step_idx_full = step_idx_emp

    sigma_full, sigma_idx_full = _sample_tau_for_step(device, B, T, k_max, step_idx_full)
    sigma_emp = sigma_full[:B_emp]
    sigma_self = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    z0_full = torch.randn_like(z1)
    z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * z1
    z_tilde_self = z_tilde_full[B_emp:]

    w_emp = 0.9 * sigma_emp + 0.1
    w_self = 0.9 * sigma_self + 0.1

    z1_hat_full, _ = dynamics(actions, step_idx_full, sigma_idx_full, z_tilde_full, act_mask=act_mask_full, agent_tokens=agent_tokens)
    z1_hat_emp = z1_hat_full[:B_emp]
    z1_hat_self = z1_hat_full[B_emp:]

    # Per-token MSE with optional saliency weighting
    flow_per_token = (z1_hat_emp.float() - z1[:B_emp].float()).pow(2).mean(dim=3)  # (B_emp,T,Sz)
    if spatial_weights is not None:
        w_sal_emp = spatial_weights[:B_emp]                                         # (B_emp,T,Sz)
        flow_per = (flow_per_token * w_sal_emp).sum(dim=2) / w_sal_emp.sum(dim=2).clamp_min(1e-8)
    else:
        flow_per = flow_per_token.mean(dim=2)                                       # (B_emp,T)
    loss_emp = (flow_per * w_emp).mean()

    boot_mse = torch.zeros((), device=device, dtype=torch.float32)
    loss_self = torch.zeros((), device=device, dtype=torch.float32)

    do_boot = (B_self > 0) and (step >= bootstrap_start)
    if do_boot:
        d_half = d_self / 2.0
        step_idx_half = step_idx_self + 1
        sigma_plus = sigma_self + d_half
        sigma_idx_plus = sigma_idx_self + (torch.tensor(k_max, device=device, dtype=torch.float32) * d_half).to(torch.long)

        z1_hat_half1, _ = dynamics(actions[B_emp:] if actions is not None else None, step_idx_half, sigma_idx_self, z_tilde_self, act_mask=act_mask_self, agent_tokens=agent_tokens[B_emp:] if agent_tokens is not None else None)
        b_prime = (z1_hat_half1.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
        z_prime = z_tilde_self.float() + b_prime * d_half[..., None, None]

        z1_hat_half2, _ = dynamics(actions[B_emp:] if actions is not None else None, step_idx_half, sigma_idx_plus, z_prime.to(z_tilde_self.dtype), act_mask=act_mask_self, agent_tokens=agent_tokens[B_emp:] if agent_tokens is not None else None)
        b_doubleprime = (z1_hat_half2.float() - z_prime.float()) / (1.0 - sigma_plus).clamp_min(1e-6)[..., None, None]

        vhat_sigma = (z1_hat_self.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
        vbar_target = ((b_prime + b_doubleprime) / 2.0).detach()

        # Bootstrap loss with optional saliency weighting
        boot_per_token = (1.0 - sigma_self)[..., None].pow(2) * (vhat_sigma - vbar_target).pow(2).mean(dim=3)  # (B_self,T,Sz)
        if spatial_weights is not None:
            w_sal_self = spatial_weights[B_emp:]                                     # (B_self,T,Sz)
            boot_per = (boot_per_token * w_sal_self).sum(dim=2) / w_sal_self.sum(dim=2).clamp_min(1e-8)
        else:
            boot_per = boot_per_token.mean(dim=2)
        loss_self = (boot_per * w_self).mean()
        boot_mse = boot_per.mean()

    loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B

    aux = {
        "flow_mse": flow_per.mean().detach(),
        "bootstrap_mse": boot_mse.detach(),
        "loss_emp": loss_emp.detach(),
        "loss_self": loss_self.detach(),
        "sigma_mean": sigma_full.mean().detach(),
    }

    return loss, aux


# ---------------------------------------------------------------------------
# Sampling / evaluation (same as train_dynamics.py)
# ---------------------------------------------------------------------------

def _is_pow2(n: int) -> bool:
    return (n > 0) and ((n & (n - 1)) == 0)


def make_tau_schedule(*, k_max: int, schedule: str, d: Optional[float] = None) -> Dict[str, Any]:
    assert _is_pow2(k_max), "k_max must be power of two"
    if schedule == "finest":
        K = k_max
    elif schedule == "shortcut":
        assert d is not None, "shortcut schedule requires --eval_d"
        inv = int(round(1.0 / float(d)))
        assert _is_pow2(inv), "eval_d must be 1/(power of two)"
        assert inv <= k_max, "eval_d must be >= 1/k_max"
        assert (k_max % inv) == 0, "k_max must be divisible by 1/eval_d"
        K = inv
    else:
        raise ValueError(f"unknown schedule: {schedule}")

    e = int(round(math.log2(K)))
    scale = k_max // K
    tau = [i / K for i in range(K)] + [1.0]
    tau_idx = [i * scale for i in range(K)] + [k_max]
    return dict(K=K, e=e, scale=scale, tau=tau, tau_idx=tau_idx, dt=1.0 / K, schedule=schedule, d=1.0 / K)


@torch.no_grad()
def sample_one_timestep_packed(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = past_packed.device
    dtype = past_packed.dtype
    B, t = past_packed.shape[:2]
    n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]

    K = int(sched["K"])
    e = int(sched["e"])
    tau = sched["tau"]
    tau_idx = sched["tau_idx"]
    dt = float(sched["dt"])

    z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)
    emax = int(round(math.log2(k_max)))

    step_idxs_full = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
    step_idxs_full[:, -1] = e

    signal_idxs_full = torch.full((B, t + 1), k_max - 1, device=device, dtype=torch.long)

    if act_mask is not None and act_mask.dim() == 1:
        act_mask = act_mask.view(1, 1, -1)

    for i in range(K):
        tau_i = float(tau[i])
        sig_i = int(tau_idx[i])

        signal_idxs_full[:, -1] = sig_i
        packed_seq = torch.cat([past_packed, z], dim=1)

        actions_in = None if actions is None else actions[:, : t + 1]
        actmask_in = None if act_mask is None else act_mask[:, : t + 1]

        x1_hat_full, _ = dyn(
            actions_in,
            step_idxs_full,
            signal_idxs_full,
            packed_seq,
            act_mask=actmask_in,
            agent_tokens=None,
        )
        x1_hat = x1_hat_full[:, -1:, :, :]

        denom = max(1e-4, 1.0 - tau_i)
        b = (x1_hat.float() - z.float()) / denom
        z = (z.float() + b * dt).to(dtype)

    return z[:, 0]


@torch.no_grad()
def sample_autoregressive_packed_sequence(
    dyn: Dynamics,
    *,
    z_gt_packed: torch.Tensor,
    ctx_length: int,
    horizon: int,
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, T = z_gt_packed.shape[:2]
    L = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, L - 1)
    horizon = min(horizon, L - ctx_length)

    outs = [z_gt_packed[:, t] for t in range(ctx_length)]

    for t in range(ctx_length, ctx_length + horizon):
        past = torch.stack(outs, dim=1)
        z_next = sample_one_timestep_packed(
            dyn,
            past_packed=past,
            k_max=k_max,
            sched=sched,
            actions=actions,
            act_mask=act_mask,
        )
        outs.append(z_next)

    return torch.stack(outs, dim=1)


@torch.no_grad()
def decode_packed_to_frames(
    decoder: Decoder,
    *,
    z_packed: torch.Tensor,
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
) -> torch.Tensor:
    z_btLd = unpack_spatial_to_bottleneck(z_packed, k=packing_factor)
    patches_btnd = decoder(z_btLd)
    frames = temporal_unpatchify(patches_btnd, H, W, C, patch)
    return frames.clamp(0, 1)


@torch.no_grad()
def log_dynamics_eval_wandb(
    *,
    gt: torch.Tensor,
    pred: torch.Tensor,
    ctx_length: int,
    step: int,
    tag: str,
    max_items: int = 4,
    gap_px: int = 16,
):
    B, T, C, H, W = gt.shape
    Bv = min(B, max_items)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        x = x[:Bv]
        B_, T_, C_, H_, W_ = x.shape
        ctx = int(max(0, min(ctx_length, T_)))
        y = x.permute(0, 2, 3, 1, 4).contiguous().view(B_, C_, H_, T_ * W_)
        if gap_px > 0 and 0 < ctx < T_:
            split = ctx * W_
            left = y[..., :split]
            right = y[..., split:]
            gap = torch.zeros((B_, C_, H_, gap_px), device=y.device, dtype=y.dtype)
            y = torch.cat([left, gap, right], dim=-1)
        return y

    gt_t = tile_time(gt)
    pr_t = tile_time(pred)
    panel = torch.cat([gt_t, pr_t], dim=2)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)

    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log(
        {f"{tag}/viz": wandb.Image(big_hwc, caption=f"rows=GT/Pred | ctx={ctx_length} | T={T}")},
        step=step,
    )


@torch.no_grad()
def log_saliency_viz_wandb(
    *,
    saliency: torch.Tensor,        # (B,T,Sz) float [0,1] spatial-token saliency
    step: int,
    max_items: int = 4,
    max_t: int = 8,
):
    """Log spatial-token saliency as a table to wandb."""
    B, T, Sz = saliency.shape
    Bv = min(B, max_items)
    Tv = min(T, max_t)
    sal = saliency[:Bv, :Tv]  # (Bv, Tv, Sz)

    # Log as wandb table for clear visualization
    columns = ["batch", "time"] + [f"token_{i}" for i in range(Sz)]
    data = []
    for b in range(Bv):
        for t in range(Tv):
            row = [b, t] + [float(sal[b, t, s].item()) for s in range(Sz)]
            data.append(row)

    table = wandb.Table(columns=columns, data=data)
    wandb.log({"eval/saliency_table": table}, step=step)


@torch.no_grad()
def run_dynamics_eval(
    *,
    encoder: Encoder,
    decoder: Decoder,
    dyn: Dynamics,
    frames: torch.Tensor,
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
    k_max: int,
    ctx_length: int,
    horizon: int,
    sched: Dict[str, Any],
    max_items: int,
    step: int,
):
    dyn_was_training = dyn.training
    dyn.eval()

    B, T = frames.shape[:2]
    T_eval = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, T_eval - 1)
    horizon = min(horizon, T_eval - ctx_length)

    frames_eval = frames[:, :T_eval]

    patches = temporal_patchify(frames_eval, patch)
    z_btLd, _ = encoder(patches)
    assert z_btLd.shape[2] % packing_factor == 0
    n_spatial = z_btLd.shape[2] // packing_factor
    z_gt_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)

    actions_eval = None if actions is None else actions[:, :T_eval]
    act_mask_eval = None if act_mask is None else act_mask[:, :T_eval] if act_mask.dim() == 3 else act_mask

    z_pred_packed = sample_autoregressive_packed_sequence(
        dyn,
        z_gt_packed=z_gt_packed,
        ctx_length=ctx_length,
        horizon=horizon,
        k_max=k_max,
        sched=sched,
        actions=actions_eval,
        act_mask=act_mask_eval,
    )

    pred_frames = decode_packed_to_frames(
        decoder,
        z_packed=z_pred_packed,
        H=H, W=W, C=C, patch=patch,
        packing_factor=packing_factor,
    )

    floor = frames_eval.clone()
    if horizon > 0:
        floor[:, ctx_length:ctx_length + horizon] = frames_eval[:, ctx_length - 1:ctx_length].expand(-1, horizon, -1, -1, -1)

    gt_h    = frames_eval[:, ctx_length:ctx_length + horizon]
    pred_h  = pred_frames[:, ctx_length:ctx_length + horizon]
    floor_h = floor[:, ctx_length:ctx_length + horizon]

    mse_pred  = (pred_h.float()  - gt_h.float()).pow(2).mean()
    mse_floor = (floor_h.float() - gt_h.float()).pow(2).mean()

    psnr_pred  = 10.0 * torch.log10(1.0 / mse_pred.clamp_min(1e-12))
    psnr_floor = 10.0 * torch.log10(1.0 / mse_floor.clamp_min(1e-12))

    mse_ratio = mse_pred / mse_floor.clamp_min(1e-12)
    psnr_gain = psnr_pred - psnr_floor

    per_t_pred  = (pred_h.float()  - gt_h.float()).pow(2).mean(dim=(0,2,3,4))
    per_t_floor = (floor_h.float() - gt_h.float()).pow(2).mean(dim=(0,2,3,4))

    if horizon > 0:
        i0 = 0
        im = (horizon - 1) // 2
        i1 = horizon - 1

        wandb.log(
            {
                "eval/mse_pred": float(mse_pred.item()),
                "eval/mse_floor": float(mse_floor.item()),
                "eval/mse_ratio_pred_over_floor": float(mse_ratio.item()),
                "eval/psnr_pred": float(psnr_pred.item()),
                "eval/psnr_floor": float(psnr_floor.item()),
                "eval/psnr_gain_over_floor_db": float(psnr_gain.item()),
                "eval/mse_pred_t1": float(per_t_pred[i0].item()),
                "eval/mse_pred_tmid": float(per_t_pred[im].item()),
                "eval/mse_pred_tend": float(per_t_pred[i1].item()),
                "eval/mse_floor_t1": float(per_t_floor[i0].item()),
                "eval/mse_floor_tmid": float(per_t_floor[im].item()),
                "eval/mse_floor_tend": float(per_t_floor[i1].item()),
            },
            step=step,
        )

    log_dynamics_eval_wandb(
        gt=frames_eval,
        pred=pred_frames,
        ctx_length=ctx_length,
        step=step,
        tag="eval",
        max_items=max_items,
    )

    if dyn_was_training:
        dyn.train()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed + rank)

    # Dataset — always use WMDataset (need rewards for saliency)
    from wm_dataset import WMDataset, collate_batch
    dataset = WMDataset(
        data_dir=args.data_dirs,
        frames_dir=args.frame_dirs,
        seq_len=args.seq_len,
        img_size=128,
        action_dim=16,
        tasks_json=args.tasks_json,
        tasks=TASK_SET,
        verbose=is_rank0(),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
        collate_fn=collate_batch,
    )

    # Load frozen tokenizer
    tok_override = {}
    if args.H is not None: tok_override["H"] = args.H
    if args.W is not None: tok_override["W"] = args.W
    if args.C is not None: tok_override["C"] = args.C
    if args.patch is not None: tok_override["patch"] = args.patch

    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        args.tokenizer_ckpt, device=device, override=tok_override
    )

    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))

    assert H % patch == 0 and W % patch == 0
    assert n_latents % args.packing_factor == 0
    n_spatial = n_latents // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor

    # Build dynamics model
    dyn = Dynamics(
        d_model=args.d_model_dyn,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=args.n_register,
        n_agent=args.n_agent,
        n_heads=args.n_heads,
        depth=args.dyn_depth,
        k_max=args.k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        space_mode=args.space_mode,
    ).to(device)

    # Build reward prediction head
    reward_head = RewardHead(d_input=d_spatial, d_hidden=args.reward_hidden).to(device)

    if is_rank0():
        print(dyn)
        dyn_params = sum(p.numel() for p in dyn.parameters() if p.requires_grad)
        rh_params = sum(p.numel() for p in reward_head.parameters() if p.requires_grad)
        print(f"Learnable parameters (dynamics): {dyn_params:,}")
        print(f"Learnable parameters (reward head): {rh_params:,}")
        print(f"[tokenizer] H={H} W={W} C={C} patch={patch} n_lat={n_latents} d_b={d_bottleneck} packing={args.packing_factor}")
        print(f"[saliency] lambda_rew={args.lambda_rew} warmup={args.warmup_steps} delta={args.delta} every={args.saliency_every}")

    if args.compile:
        dyn = torch.compile(dyn)

    if ddp:
        dyn = torch.nn.parallel.DistributedDataParallel(
            dyn, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    # Optimizer: dynamics + reward head
    opt = torch.optim.AdamW(
        list(dyn.parameters()) + list(reward_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(device="cuda", enabled=use_amp)

    # Initialize wandb
    if is_rank0():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode="online",
            config=vars(args),
        )

    # Resume from checkpoint
    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume is not None:
        step, start_epoch = load_ckpt(Path(args.resume), dyn_model=dyn, reward_head=reward_head, opt=opt, scaler=scaler)
        if is_rank0():
            print(f"[rank0] Resumed from {args.resume} (step={step}, epoch={start_epoch})")

    # Training loop
    dyn.train()
    reward_head.train()
    t0 = time.time()
    grad_accum = max(1, int(args.grad_accum))

    # Cached saliency
    cached_saliency = None

    while step < args.max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch in loader:
                if step >= args.max_steps:
                    break

                # --- Data preparation ---
                obs_u8 = batch["obs"].to(device, non_blocking=True)          # (B,T+1,3,H,W) uint8
                act    = batch["act"].to(device, non_blocking=True)          # (B,T,16) float
                mask   = batch["act_mask"].to(device, non_blocking=True)     # (B,T,16) float
                rew    = batch["rew"].to(device, non_blocking=True)          # (B,T) float

                act = act.clamp(-1, 1) * mask

                # Keep obs[0..T-1], align action[t] as action that produced obs[t]
                frames = obs_u8[:, :-1].float() / 255.0                      # (B,T,3,H,W)
                actions = torch.zeros_like(act)
                actions[:, 1:] = act[:, :-1]
                act_mask = torch.zeros_like(mask)
                act_mask[:, 1:] = mask[:, :-1]

                # Rewards: rew[t] corresponds to transition from frame[t] to next
                rewards = torch.nan_to_num(rew, nan=0.0)                     # (B,T)

                # Safeguard
                if frames.dtype == torch.uint8:
                    frames = frames.float() / 255.0

                # --- Frozen encoder → packed spatial tokens z1 ---
                with torch.no_grad():
                    patches = temporal_patchify(frames, patch)  # (B,T,Np,Dp)
                    z_btLd, _ = encoder(patches)                # (B,T,n_latents,d_b)
                    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=args.packing_factor)  # (B,T,Sz,Dz)

                # --- Compute spatial-token saliency (every saliency_every steps) ---
                if step % args.saliency_every == 0:
                    with torch.enable_grad():
                        saliency, _ = compute_spatial_saliency(
                            reward_head, z1,
                            delta=args.delta,
                        )
                    cached_saliency = saliency  # (B,T,Sz)
                else:
                    # Reuse cached saliency; handle batch size mismatch
                    if cached_saliency is None or cached_saliency.shape[0] != z1.shape[0]:
                        # Fallback: uniform weights
                        cached_saliency = torch.ones(
                            z1.shape[0], z1.shape[1], z1.shape[2],
                            device=device, dtype=torch.float32,
                        )

                # Apply warm-up schedule
                alpha = min(1.0, step / max(1, args.warmup_steps))
                saliency_final = (1.0 - alpha) + alpha * cached_saliency  # (B,T,Sz)

                # --- Reward prediction loss ---
                # Use z1 (detached from encoder, but reward_head params have grad)
                r_hat = reward_head(z1.detach().mean(dim=2))  # (B,T)
                reward_loss = (r_hat - rewards).pow(2).mean()

                # --- Dynamics loss with spatial-token saliency weighting ---
                if actions is not None:
                    actions = actions.to(device, non_blocking=True)

                B = z1.shape[0]
                B_self = int(round(args.self_fraction * B))
                B_self = max(0, min(B - 1, B_self))

                with autocast(device_type="cuda", enabled=use_amp):
                    loss_dyn, aux = dynamics_pretrain_loss(
                        dyn.module if hasattr(dyn, "module") else dyn,
                        z1=z1,
                        actions=actions,
                        act_mask=act_mask,
                        k_max=args.k_max,
                        B_self=B_self,
                        step=step,
                        bootstrap_start=args.bootstrap_start,
                        agent_tokens=None,
                        spatial_weights=saliency_final,
                    )

                # --- Combine losses ---
                loss = loss_dyn + args.lambda_rew * reward_loss

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: loss={loss}")

                loss_to_backprop = loss / grad_accum
                scaler.scale(loss_to_backprop).backward()

                do_step = ((step + 1) % grad_accum == 0)
                if do_step:
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        all_params = list((dyn.module if hasattr(dyn, "module") else dyn).parameters()) + list(reward_head.parameters())
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)

                    if use_amp:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                # --- Evaluation / visualization ---
                if is_rank0() and args.eval_every > 0 and (step % args.eval_every == 0):
                    B_eval = min(frames.shape[0], args.eval_batch_size)
                    frames_eval = frames[:B_eval]

                    actions_eval = actions[:B_eval]
                    act_mask_eval = act_mask[:B_eval]

                    sched = make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)

                    run_dynamics_eval(
                        encoder=encoder,
                        decoder=decoder,
                        dyn=(dyn.module if hasattr(dyn, "module") else dyn),
                        frames=frames_eval,
                        actions=actions_eval,
                        act_mask=act_mask_eval,
                        H=H, W=W, C=C, patch=patch,
                        packing_factor=args.packing_factor,
                        k_max=args.k_max,
                        ctx_length=args.eval_ctx,
                        horizon=args.eval_horizon,
                        sched=sched,
                        max_items=args.eval_max_items,
                        step=step,
                    )

                    # Log saliency visualization
                    log_saliency_viz_wandb(
                        saliency=saliency_final[:B_eval],
                        step=step,
                    )

                # --- Logging ---
                if is_rank0() and (step % args.log_every == 0):

                    # Action shuffle loss ratio
                    with torch.no_grad():
                        loss_real, _ = dynamics_pretrain_loss(
                            dyn.module if hasattr(dyn, "module") else dyn,
                            z1=z1,
                            actions=actions,
                            act_mask=act_mask,
                            k_max=args.k_max,
                            B_self=B_self,
                            step=step,
                            bootstrap_start=args.bootstrap_start,
                            agent_tokens=None,
                        )
                        perm = torch.randperm(actions.shape[0], device=actions.device)
                        loss_shuffled, _ = dynamics_pretrain_loss(
                            dyn.module if hasattr(dyn, "module") else dyn,
                            z1=z1,
                            actions=actions[perm],
                            act_mask=act_mask,
                            k_max=args.k_max,
                            B_self=B_self,
                            step=step,
                            bootstrap_start=args.bootstrap_start,
                            agent_tokens=None,
                        )
                    action_shuffle_loss_ratio = loss_shuffled / loss_dyn.clamp_min(1e-12)

                    wandb.log(
                        {
                            "loss/total": float(loss.item()),
                            "loss/dynamics": float(loss_dyn.item()),
                            "loss/reward_pred": float(reward_loss.item()),
                            "loss/flow_mse": float(aux["flow_mse"].item()),
                            "loss/bootstrap_mse": float(aux["bootstrap_mse"].item()),
                            "loss/loss_emp": float(aux["loss_emp"].item()),
                            "loss/loss_self": float(aux["loss_self"].item()),
                            "stats/action_shuffle_loss_ratio": float(action_shuffle_loss_ratio.item()),
                            "stats/sigma_mean": float(aux["sigma_mean"].item()),
                            "stats/B_self": float(B_self),
                            "stats/alpha_warmup": float(alpha),
                            "stats/saliency_mean": float(cached_saliency.mean().item()),
                            "stats/saliency_std": float(cached_saliency.std().item()),
                            "lr": float(opt.param_groups[0]["lr"]),
                            "time/hrs": (time.time() - t0) / 3600.0,
                        },
                        step=step,
                    )

                    print(
                        f"step {step:07d} | loss={loss.item():.6f} "
                        f"| dyn={loss_dyn.item():.6f} "
                        f"| rew={reward_loss.item():.6f} "
                        f"| alpha={alpha:.3f} "
                        f"| flow_mse={aux['flow_mse'].item():.6f} "
                        f"| boot_mse={aux['bootstrap_mse'].item():.6f} "
                        f"| sigma={aux['sigma_mean'].item():.3f} | B_self={B_self}"
                    )

                # --- Checkpointing ---
                if is_rank0() and args.save_every > 0 and (step % args.save_every == 0) and do_step:
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    save_ckpt(ckpt_path, step=step, epoch=epoch, dyn_model=dyn, reward_head=reward_head, opt=opt, scaler=scaler, args=args)
                    latest = ckpt_dir / "latest.pt"
                    save_ckpt(latest, step=step, epoch=epoch, dyn_model=dyn, reward_head=reward_head, opt=opt, scaler=scaler, args=args)

                step += 1

            start_epoch = epoch + 1

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dirs", type=str, nargs="+", default=[
        "/<path>/expert",
        "/<path>/mixed-small",
        "/<path>/mixed-large",
    ])
    p.add_argument("--frame_dirs", type=str, nargs="+", default=[
        "/<path>/expert-shards",
        "/<path>/mixed-small-shards",
        "/<path>/mixed-large-shards",
    ])
    p.add_argument("--tasks_json", type=str, default="../tasks.json")
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=24)

    # tokenizer restore
    p.add_argument("--tokenizer_ckpt", type=str, default="./logs/tokenizer_ckpts/latest.pt")
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--W", type=int, default=None)
    p.add_argument("--C", type=int, default=None)
    p.add_argument("--patch", type=int, default=None)

    # dynamics arch
    p.add_argument("--d_model_dyn", type=int, default=512)
    p.add_argument("--dyn_depth", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)

    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--n_register", type=int, default=4)
    p.add_argument("--n_agent", type=int, default=1)
    p.add_argument("--space_mode", type=str, default="wm_agent_isolated", choices=["wm_agent_isolated", "wm_agent"])

    # shortcut / schedule
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--bootstrap_start", type=int, default=5_000)
    p.add_argument("--self_fraction", type=float, default=0.25)

    # saliency weighting (NEW)
    p.add_argument("--lambda_rew", type=float, default=0.01, help="Weight for reward prediction loss")
    p.add_argument("--warmup_steps", type=int, default=5_000, help="Linear warm-up steps for saliency")
    p.add_argument("--saliency_every", type=int, default=5, help="Recompute saliency every N steps")
    p.add_argument("--delta", type=float, default=0.1, help="Minimum saliency weight (floor)")
    p.add_argument("--reward_hidden", type=int, default=128, help="Hidden dim of reward head MLP")

    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10_000_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # eval / viz
    p.add_argument("--eval_every", type=int, default=1_000)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_max_items", type=int, default=4)
    p.add_argument("--eval_ctx", type=int, default=8)
    p.add_argument("--eval_horizon", type=int, default=16)
    p.add_argument("--eval_schedule", type=str, default="shortcut", choices=["finest", "shortcut"])
    p.add_argument("--eval_d", type=float, default=0.25)

    # logging
    p.add_argument("--log_every", type=int, default=200)

    # wandb
    p.add_argument("--wandb_project", type=str, default="dreamer4-dynamics-saliency")
    p.add_argument("--wandb_run_name", type=str, default="saliency-weighted")
    p.add_argument("--wandb_entity", type=str, default=None)

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./logs/dynamics_ckpts_saliency")
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--resume", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")

    train(p.parse_args())
