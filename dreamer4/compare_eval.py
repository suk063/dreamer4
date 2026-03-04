#!/usr/bin/env python3
"""
compare_eval.py -- Compare baseline vs saliency dynamics evaluation results.

Loads eval_results.json from two evaluation runs and produces:
  1. Per-task comparison table (printed + saved as CSV)
  2. Aggregate summary
  3. Side-by-side horizon curves (baseline vs saliency)
  4. Per-task grouped bar charts
  5. Side-by-side qualitative grids (GT / Baseline / Saliency)

Usage:
    python compare_eval.py \
        --baseline_dir ./eval_output_baseline \
        --saliency_dir ./eval_output_saliency \
        --output_dir ./eval_output_comparison
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(result_dir: Path) -> Dict[str, Any]:
    path = result_dir / "eval_results.json"
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# Table output
# ============================================================
def print_comparison_table(base: Dict, sal: Dict, output_dir: Path):
    base_tasks = base["per_task"]
    sal_tasks = sal["per_task"]
    all_tasks = sorted(set(base_tasks.keys()) | set(sal_tasks.keys()))

    metrics = [
        ("PSNR", "psnr", True),       # higher is better
        ("SSIM", "ssim", True),
        ("LPIPS", "lpips", False),     # lower is better
        ("Latent MSE", "latent_mse", False),
        ("Cos Sim", "cos_sim", True),
    ]

    header = f"{'Task':<35s}"
    for name, _, _ in metrics:
        header += f" | {name+' (B)':>12s} {name+' (S)':>12s} {'Delta':>8s}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    wins = {m[1]: 0 for m in metrics}
    total = 0

    rows = []
    for task in all_tasks:
        b = base_tasks.get(task, {})
        s = sal_tasks.get(task, {})
        if not b or not s:
            continue
        total += 1
        row = f"{task:<35s}"
        csv_row = [task]
        for name, key, higher_better in metrics:
            bv = b.get(key, 0)
            sv = s.get(key, 0)
            delta = sv - bv
            if (higher_better and delta > 0) or (not higher_better and delta < 0):
                wins[key] += 1
                marker = "+"
            else:
                marker = " "
            row += f" | {bv:>12.4f} {sv:>12.4f} {delta:>+8.4f}{marker}"
            csv_row.extend([f"{bv:.6f}", f"{sv:.6f}", f"{delta:+.6f}"])
        print(row)
        rows.append(csv_row)

    print("=" * len(header))

    # Aggregate
    ba = base["aggregate"]
    sa = sal["aggregate"]
    agg_row = f"{'AGGREGATE':<35s}"
    for name, key, higher_better in metrics:
        bv = ba.get(key, 0)
        sv = sa.get(key, 0)
        delta = sv - bv
        agg_row += f" | {bv:>12.4f} {sv:>12.4f} {delta:>+8.4f} "
    print(agg_row)
    print()

    print("Win counts (saliency better):")
    for name, key, _ in metrics:
        print(f"  {name}: {wins[key]}/{total}")
    print()

    # Action shuffle
    base_asr = ba.get("action_shuffle_ratio", 0)
    sal_asr = sa.get("action_shuffle_ratio", 0)
    print(f"Action Shuffle Ratio:  Baseline={base_asr:.3f}  Saliency={sal_asr:.3f}")
    print()

    # Save CSV
    csv_path = output_dir / "comparison_table.csv"
    csv_header = ["task"]
    for name, _, _ in metrics:
        csv_header.extend([f"{name}_baseline", f"{name}_saliency", f"{name}_delta"])
    with open(csv_path, "w") as f:
        f.write(",".join(csv_header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"Saved {csv_path}")


# ============================================================
# Horizon curves (side-by-side)
# ============================================================
def plot_horizon_comparison(base: Dict, sal: Dict, output_dir: Path):
    ba = base["aggregate"]
    sa = sal["aggregate"]
    horizon = int(ba["horizon"])
    ts = list(range(1, horizon + 1))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_specs = [
        ("latent_mse_per_t", "Latent MSE", None, False),
        ("pixel_mse_per_t", "Pixel MSE", "floor_mse_per_t", False),
        ("psnr_per_t", "PSNR (dB)", "floor_psnr_per_t", True),
        ("ssim_per_t", "SSIM", "floor_ssim_per_t", True),
        ("lpips_per_t", "LPIPS", "floor_lpips_per_t", False),
        ("cos_sim_per_t", "Cosine Similarity", None, True),
    ]

    for idx, (key, ylabel, floor_key, higher_better) in enumerate(plot_specs):
        ax = axes[idx // 3, idx % 3]

        bvals = ba.get(key, [])
        svals = sa.get(key, [])
        if not bvals or not svals:
            ax.set_visible(False)
            continue

        ax.plot(ts[:len(bvals)], bvals, "o-", label="Baseline", color="#2196F3", markersize=3, linewidth=1.5)
        ax.plot(ts[:len(svals)], svals, "s-", label="Saliency", color="#E91E63", markersize=3, linewidth=1.5)

        if floor_key and floor_key in ba:
            fvals = ba[floor_key]
            ax.plot(ts[:len(fvals)], fvals, "^--", label="Repeat-last", color="#FF9800", markersize=3, linewidth=1, alpha=0.7)

        ax.set_xlabel("Horizon Step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Baseline vs Saliency — Horizon Metrics (Step 95000)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "horizon_comparison.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Per-task bar comparison
# ============================================================
def plot_per_task_comparison(base: Dict, sal: Dict, output_dir: Path):
    base_tasks = base["per_task"]
    sal_tasks = sal["per_task"]
    common_tasks = sorted(set(base_tasks.keys()) & set(sal_tasks.keys()),
                          key=lambda t: base_tasks[t].get("psnr", 0), reverse=True)

    if not common_tasks:
        return

    x = np.arange(len(common_tasks))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    specs = [
        ("psnr", "PSNR (dB)", "higher is better"),
        ("ssim", "SSIM", "higher is better"),
        ("lpips", "LPIPS", "lower is better"),
        ("latent_mse", "Latent MSE", "lower is better"),
    ]

    for idx, (key, ylabel, note) in enumerate(specs):
        ax = axes[idx // 2, idx % 2]
        bvals = [base_tasks[t].get(key, 0) for t in common_tasks]
        svals = [sal_tasks[t].get(key, 0) for t in common_tasks]

        ax.bar(x - w/2, bvals, w, label="Baseline", color="#2196F3", alpha=0.85)
        ax.bar(x + w/2, svals, w, label="Saliency", color="#E91E63", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(common_tasks, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Per-Task {ylabel} ({note})")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Baseline vs Saliency — Per-Task Metrics (Step 95000)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "per_task_comparison.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Qualitative comparison grid (GT / Baseline / Saliency)
# ============================================================
def plot_qualitative_comparison(
    baseline_dir: Path,
    saliency_dir: Path,
    output_dir: Path,
):
    """Build side-by-side qualitative grids using saved sample frames."""
    from PIL import Image, ImageDraw, ImageFont

    # Load sample frames from both evaluations
    # eval.py saves gt_frames and pred_frames in the batch metrics, but
    # these are not in eval_results.json. Instead, load the qualitative_grid.png
    # from each directory for a simple side-by-side comparison.

    base_grid = baseline_dir / "figures" / "qualitative_grid.png"
    sal_grid = saliency_dir / "figures" / "qualitative_grid.png"

    if not base_grid.exists() or not sal_grid.exists():
        print("  Qualitative grids not found in one or both eval directories.")
        print(f"    Baseline: {base_grid} exists={base_grid.exists()}")
        print(f"    Saliency: {sal_grid} exists={sal_grid.exists()}")
        return

    base_img = Image.open(base_grid)
    sal_img = Image.open(sal_grid)

    # Resize to same height if needed
    h = max(base_img.height, sal_img.height)
    if base_img.height != h:
        ratio = h / base_img.height
        base_img = base_img.resize((int(base_img.width * ratio), h), Image.LANCZOS)
    if sal_img.height != h:
        ratio = h / sal_img.height
        sal_img = sal_img.resize((int(sal_img.width * ratio), h), Image.LANCZOS)

    # Add labels
    label_h = 30
    gap = 16

    total_w = base_img.width + gap + sal_img.width
    total_h = h + label_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    canvas.paste(base_img, (0, label_h))
    canvas.paste(sal_img, (base_img.width + gap, label_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    draw.text((base_img.width // 2 - 50, 5), "Baseline", fill=(33, 150, 243), font=font)
    draw.text((base_img.width + gap + sal_img.width // 2 - 50, 5), "Saliency", fill=(233, 30, 99), font=font)

    path = output_dir / "qualitative_comparison.png"
    canvas.save(str(path))
    print(f"Saved {path}")

    # Also create side-by-side GIF comparison for select tasks
    base_vid_dir = baseline_dir / "figures" / "videos"
    sal_vid_dir = saliency_dir / "figures" / "videos"

    if base_vid_dir.exists() and sal_vid_dir.exists():
        compare_vid_dir = output_dir / "videos"
        compare_vid_dir.mkdir(parents=True, exist_ok=True)

        base_gifs = {p.stem: p for p in base_vid_dir.glob("*.gif")}
        sal_gifs = {p.stem: p for p in sal_vid_dir.glob("*.gif")}
        common = sorted(set(base_gifs.keys()) & set(sal_gifs.keys()))

        for task in common[:8]:
            bg = Image.open(base_gifs[task])
            sg = Image.open(sal_gifs[task])

            frames = []
            n_frames = min(getattr(bg, 'n_frames', 1), getattr(sg, 'n_frames', 1))

            for fi in range(n_frames):
                bg.seek(fi)
                sg.seek(fi)

                bf = bg.copy().convert("RGB")
                sf = sg.copy().convert("RGB")

                # Match heights
                mh = max(bf.height, sf.height)
                w_total = bf.width + 8 + sf.width
                frame = Image.new("RGB", (w_total, mh + 20), (255, 255, 255))
                frame.paste(bf, (0, 20))
                frame.paste(sf, (bf.width + 8, 20))

                d = ImageDraw.Draw(frame)
                try:
                    fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
                except OSError:
                    fnt = ImageFont.load_default()
                d.text((2, 2), "Baseline", fill=(33, 150, 243), font=fnt)
                d.text((bf.width + 10, 2), "Saliency", fill=(233, 30, 99), font=fnt)

                frames.append(frame)

            if frames:
                gif_path = compare_vid_dir / f"{task}_comparison.gif"
                frames[0].save(
                    str(gif_path), save_all=True,
                    append_images=frames[1:], duration=250, loop=0,
                )
                print(f"  Saved {gif_path}")


# ============================================================
# Aggregate summary figure
# ============================================================
def plot_aggregate_summary(base: Dict, sal: Dict, output_dir: Path):
    ba = base["aggregate"]
    sa = sal["aggregate"]

    metrics = ["psnr", "ssim", "lpips", "latent_mse", "cos_sim"]
    labels = ["PSNR (dB)", "SSIM", "LPIPS", "Latent MSE", "Cos Sim"]
    higher_better = [True, True, False, False, True]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x = np.arange(len(metrics))
    w = 0.35

    bvals = [ba.get(m, 0) for m in metrics]
    svals = [sa.get(m, 0) for m in metrics]

    bars_b = ax.bar(x - w/2, bvals, w, label="Baseline", color="#2196F3", alpha=0.85)
    bars_s = ax.bar(x + w/2, svals, w, label="Saliency", color="#E91E63", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Aggregate Metrics — Baseline vs Saliency (Step 95000)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars_b, bars_s]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = output_dir / "aggregate_summary.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Compare baseline vs saliency eval results")
    p.add_argument("--baseline_dir", type=str, required=True)
    p.add_argument("--saliency_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./eval_output_comparison")
    args = p.parse_args()

    baseline_dir = Path(args.baseline_dir)
    saliency_dir = Path(args.saliency_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    base = load_results(baseline_dir)
    sal = load_results(saliency_dir)

    print("\n" + "=" * 80)
    print("COMPARISON: Baseline vs Saliency (Step 95000)")
    print("=" * 80 + "\n")

    print_comparison_table(base, sal, output_dir)

    print("\nGenerating comparison figures...")
    plot_horizon_comparison(base, sal, output_dir)
    plot_per_task_comparison(base, sal, output_dir)
    plot_aggregate_summary(base, sal, output_dir)

    print("\nGenerating qualitative comparison...")
    plot_qualitative_comparison(baseline_dir, saliency_dir, output_dir)

    # Save combined JSON
    combined = {
        "baseline": base["aggregate"],
        "saliency": sal["aggregate"],
        "per_task_baseline": base["per_task"],
        "per_task_saliency": sal["per_task"],
    }
    json_path = output_dir / "combined_results.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined results to {json_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
