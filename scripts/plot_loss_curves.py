"""
Plot training loss curves for ablation runs.

Parses loss_log.txt from each run's checkpoint directory and generates
per-loss-component plots comparing all runs.

Usage:
    python scripts/plot_loss_curves.py \
        --checkpoints_dir cyclegan_repo/checkpoints \
        --runs ablation_baseline ablation_perceptual ablation_no_identity ablation_perc_no_idt \
        --output_dir results/loss_curves
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def parse_loss_log(log_path):
    """Parse loss_log.txt into {loss_name: [(epoch, iter, value), ...]}."""
    losses = defaultdict(list)

    pattern = re.compile(
        r"\(epoch:\s*(\d+),\s*iters:\s*(\d+),.*?\)\s*,\s*(.*)"
    )

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            iters = int(m.group(2))
            loss_str = m.group(3)

            # Parse "D_A: 0.355, G_A: 0.484, ..."
            for pair in loss_str.split(","):
                pair = pair.strip()
                if ":" not in pair:
                    continue
                name, val = pair.split(":", 1)
                name = name.strip()
                try:
                    val = float(val.strip())
                except ValueError:
                    continue
                losses[name].append((epoch, iters, val))

    return dict(losses)


def compute_epoch_averages(loss_entries):
    """Average loss values per epoch."""
    epoch_sums = defaultdict(lambda: [0.0, 0])
    for epoch, iters, val in loss_entries:
        epoch_sums[epoch][0] += val
        epoch_sums[epoch][1] += 1

    epochs = sorted(epoch_sums.keys())
    avgs = [epoch_sums[e][0] / epoch_sums[e][1] for e in epochs]
    return epochs, avgs


def plot_loss_component(all_runs_data, loss_name, output_dir, smooth_window=5):
    """Plot a single loss component across all runs."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for run_name, losses in all_runs_data.items():
        if loss_name not in losses:
            continue
        epochs, avgs = compute_epoch_averages(losses[loss_name])

        # Skip if all zeros
        if max(abs(v) for v in avgs) < 1e-8:
            continue

        # Smooth with moving average
        if len(avgs) > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            avgs_smooth = np.convolve(avgs, kernel, mode="valid")
            epochs_smooth = epochs[smooth_window - 1:]
        else:
            avgs_smooth = avgs
            epochs_smooth = epochs

        label = run_name.replace("ablation_", "")
        ax.plot(epochs_smooth, avgs_smooth, label=label, linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(f"Loss ({loss_name})", fontsize=12)
    ax.set_title(f"Training Loss: {loss_name}", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"loss_{loss_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_total_generator_loss(all_runs_data, output_dir, smooth_window=5):
    """Plot total generator loss (G_A + G_B + cycle_A + cycle_B + idt_A + idt_B + perc_A + perc_B + fft_A + fft_B)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    g_components = ["G_A", "G_B", "cycle_A", "cycle_B", "idt_A", "idt_B",
                    "perc_A", "perc_B", "fft_A", "fft_B"]

    for run_name, losses in all_runs_data.items():
        # Collect all iteration points from G_A
        if "G_A" not in losses:
            continue

        # Sum all G components per (epoch, iter)
        point_sums = defaultdict(float)
        point_counts = defaultdict(int)

        for comp in g_components:
            if comp not in losses:
                continue
            for epoch, iters, val in losses[comp]:
                point_sums[epoch] += val / len(losses[comp]) * len(losses["G_A"])

        # Average per epoch
        epoch_sums = defaultdict(lambda: [0.0, 0])
        for comp in g_components:
            if comp not in losses:
                continue
            for epoch, iters, val in losses[comp]:
                epoch_sums[epoch][0] += val
                epoch_sums[epoch][1] = max(epoch_sums[epoch][1], 1)

        # Simpler: just sum epoch averages of each component
        comp_epoch_avgs = {}
        for comp in g_components:
            if comp not in losses:
                continue
            epochs, avgs = compute_epoch_averages(losses[comp])
            comp_epoch_avgs[comp] = dict(zip(epochs, avgs))

        if not comp_epoch_avgs:
            continue

        all_epochs = sorted(set().union(*[d.keys() for d in comp_epoch_avgs.values()]))
        total = [sum(comp_epoch_avgs.get(c, {}).get(e, 0) for c in g_components) for e in all_epochs]

        # Smooth
        if len(total) > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            total_smooth = np.convolve(total, kernel, mode="valid")
            epochs_smooth = all_epochs[smooth_window - 1:]
        else:
            total_smooth = total
            epochs_smooth = all_epochs

        label = run_name.replace("ablation_", "")
        ax.plot(epochs_smooth, total_smooth, label=label, linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Generator Loss", fontsize=12)
    ax.set_title("Total Generator Loss (all components)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "loss_total_generator.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined_overview(all_runs_data, output_dir, smooth_window=5):
    """Plot a 2x3 grid of the most important losses."""
    key_losses = ["G_A", "G_B", "D_A", "D_B", "cycle_A", "cycle_B"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, loss_name in enumerate(key_losses):
        ax = axes[idx]
        for run_name, losses in all_runs_data.items():
            if loss_name not in losses:
                continue
            epochs, avgs = compute_epoch_averages(losses[loss_name])
            if max(abs(v) for v in avgs) < 1e-8:
                continue

            if len(avgs) > smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                avgs_smooth = np.convolve(avgs, kernel, mode="valid")
                epochs_smooth = epochs[smooth_window - 1:]
            else:
                avgs_smooth = avgs
                epochs_smooth = epochs

            label = run_name.replace("ablation_", "")
            ax.plot(epochs_smooth, avgs_smooth, label=label, linewidth=1.2)

        ax.set_title(loss_name, fontsize=12)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=1)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.suptitle("Training Loss Curves — Ablation Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "loss_overview_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument("--checkpoints_dir", default="cyclegan_repo/checkpoints")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Run names. If not specified, auto-detect ablation_* runs.")
    parser.add_argument("--output_dir", default="results/loss_curves")
    parser.add_argument("--smooth", type=int, default=5, help="Moving average window")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect runs
    if args.runs is None:
        args.runs = sorted([
            d for d in os.listdir(args.checkpoints_dir)
            if d.startswith("ablation_") and os.path.isfile(
                os.path.join(args.checkpoints_dir, d, "loss_log.txt")
            )
        ])

    print(f"Runs: {args.runs}")

    # Parse all loss logs
    all_runs_data = {}
    for run in args.runs:
        log_path = os.path.join(args.checkpoints_dir, run, "loss_log.txt")
        if not os.path.isfile(log_path):
            print(f"  SKIP {run}: no loss_log.txt")
            continue
        all_runs_data[run] = parse_loss_log(log_path)
        print(f"  Loaded {run}: {len(all_runs_data[run])} loss components")

    # Collect all unique loss names (excluding wgangp if it diverged)
    all_loss_names = set()
    for losses in all_runs_data.values():
        all_loss_names.update(losses.keys())
    all_loss_names = sorted(all_loss_names)
    print(f"\nLoss components found: {all_loss_names}\n")

    # Plot each loss component
    print("=== Individual loss plots ===")
    for loss_name in all_loss_names:
        plot_loss_component(all_runs_data, loss_name, args.output_dir, args.smooth)

    # Plot total generator loss
    print("\n=== Total generator loss ===")
    plot_total_generator_loss(all_runs_data, args.output_dir, args.smooth)

    # Plot combined overview grid
    print("\n=== Overview grid ===")
    plot_combined_overview(all_runs_data, args.output_dir, args.smooth)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
