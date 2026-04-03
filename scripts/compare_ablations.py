"""
Compare ablation study results across multiple runs.

Reads metrics.json from each run's results directory and produces:
  1. A formatted table printed to the console
  2. A CSV file for import into Excel/LaTeX
  3. A LaTeX-ready table for thesis inclusion

Usage:
    python scripts/compare_ablations.py \
        --results_root results \
        --runs ablation_baseline ablation_perceptual ablation_no_identity ablation_perc_no_idt ablation_wgangp

    # Or auto-detect all runs that have metrics.json:
    python scripts/compare_ablations.py --results_root results --auto
"""

import argparse
import json
import os
import sys
from pathlib import Path


# Metrics to display, in order. (key, display_name, direction, format)
METRICS_TABLE = [
    ("fid_fakeB_realB",             "FID(fB,rB)",       "low",  ".2f"),
    ("fid_fakeA_realA",             "FID(fA,rA)",       "low",  ".2f"),
    ("ssim_recA_realA_mean",        "SSIM(recA)",       "high", ".4f"),
    ("ssim_recB_realB_mean",        "SSIM(recB)",       "high", ".4f"),
    ("psnr_recA_realA_mean",        "PSNR(recA)",       "high", ".2f"),
    ("psnr_recB_realB_mean",        "PSNR(recB)",       "high", ".2f"),
    ("lpips_recA_realA_mean",       "LPIPS(recA)",      "low",  ".4f"),
    ("lpips_recB_realB_mean",       "LPIPS(recB)",      "low",  ".4f"),
    ("sharpness_ratio_fakeB_realB", "Sharp ratio(B)",   "~1.0", ".3f"),
    ("cnr_fakeB_mean",              "CNR(fakeB)",       "high", ".3f"),
    ("cnr_realB_mean",              "CNR(realB)",       "",     ".3f"),
    ("cnr_ratio_fakeB_realB",       "CNR ratio(B)",     "~1.0", ".3f"),
    ("hist_kl_fakeA_realA",         "KL(fA||rA)",       "low",  ".4f"),
]


def load_metrics(results_root: str, run_name: str) -> dict:
    path = Path(results_root) / run_name / "metrics.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping {run_name}")
        return None
    with open(path) as f:
        return json.load(f)


def find_best(all_metrics: list, key: str, direction: str):
    """Return the index of the best run for a given metric."""
    values = []
    for m in all_metrics:
        if m is None:
            values.append(None)
            continue
        v = m.get(key)
        if v is None or (isinstance(v, float) and (v != v)):  # NaN check
            values.append(None)
        else:
            values.append(v)

    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return -1

    if direction == "low":
        return min(valid, key=lambda x: x[1])[0]
    elif direction == "high":
        return max(valid, key=lambda x: x[1])[0]
    elif direction == "~1.0":
        return min(valid, key=lambda x: abs(x[1] - 1.0))[0]
    return -1


def format_val(val, fmt: str) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "N/A"
    return f"{val:{fmt}}"


def print_console_table(run_names: list, all_metrics: list):
    """Print a formatted comparison table to console."""
    # Column widths
    name_w = max(len(n) for n in run_names) + 2
    col_w = 14

    # Header
    header = f"  {'Run':<{name_w}}"
    for _, display, direction, _ in METRICS_TABLE:
        label = f"{display}"
        header += f" {label:>{col_w}}"
    print(header)
    print(f"  {'-' * (name_w + col_w * len(METRICS_TABLE) + len(METRICS_TABLE))}")

    # Find best for each metric
    best_indices = {}
    for key, _, direction, _ in METRICS_TABLE:
        best_indices[key] = find_best(all_metrics, key, direction)

    # Rows
    for i, (name, metrics) in enumerate(zip(run_names, all_metrics)):
        if metrics is None:
            row = f"  {name:<{name_w}}"
            row += f" {'(no data)':>{col_w}}"
            print(row)
            continue

        row = f"  {name:<{name_w}}"
        for key, _, _, fmt in METRICS_TABLE:
            val = metrics.get(key)
            val_str = format_val(val, fmt)
            if best_indices.get(key) == i:
                val_str = f"*{val_str}"  # mark best with asterisk
            row += f" {val_str:>{col_w}}"
        print(row)

    print(f"\n  * = best in column")


def save_csv(run_names: list, all_metrics: list, out_path: str):
    """Save comparison as CSV."""
    headers = ["run"] + [display for _, display, _, _ in METRICS_TABLE]
    rows = []
    for name, metrics in zip(run_names, all_metrics):
        if metrics is None:
            rows.append([name] + ["N/A"] * len(METRICS_TABLE))
            continue
        row = [name]
        for key, _, _, fmt in METRICS_TABLE:
            row.append(format_val(metrics.get(key), fmt))
        rows.append(row)

    with open(out_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"  CSV saved: {out_path}")


def save_latex(run_names: list, all_metrics: list, out_path: str):
    """Save comparison as a LaTeX table."""
    n_cols = len(METRICS_TABLE)

    best_indices = {}
    for key, _, direction, _ in METRICS_TABLE:
        best_indices[key] = find_best(all_metrics, key, direction)

    with open(out_path, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Ablation study results}}\n")
        f.write(f"\\label{{tab:ablation}}\n")
        col_spec = "l" + "r" * n_cols
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        # Header
        headers = ["Run"] + [display for _, display, _, _ in METRICS_TABLE]
        # Add direction arrows
        header_with_dir = ["Run"]
        for _, display, direction, _ in METRICS_TABLE:
            if direction == "low":
                header_with_dir.append(f"{display} $\\downarrow$")
            elif direction == "high":
                header_with_dir.append(f"{display} $\\uparrow$")
            else:
                header_with_dir.append(f"{display} $\\approx 1$")

        f.write(" & ".join(header_with_dir) + " \\\\\n")
        f.write("\\midrule\n")

        # Rows
        for i, (name, metrics) in enumerate(zip(run_names, all_metrics)):
            if metrics is None:
                f.write(f"{name} & " + " & ".join(["N/A"] * n_cols) + " \\\\\n")
                continue

            cells = [name.replace("_", "\\_")]
            for key, _, _, fmt in METRICS_TABLE:
                val = metrics.get(key)
                val_str = format_val(val, fmt)
                if best_indices.get(key) == i:
                    val_str = f"\\textbf{{{val_str}}}"
                cells.append(val_str)
            f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"  LaTeX saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare ablation study results")
    parser.add_argument("--results_root", default="results", help="Path to results/ folder")
    parser.add_argument("--runs", nargs="+", default=None, help="Run names to compare")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect all runs with metrics.json")
    parser.add_argument("--output", default=None,
                        help="Output basename (default: results_root/ablation_comparison)")
    args = parser.parse_args()

    if args.auto:
        runs = []
        for d in sorted(Path(args.results_root).iterdir()):
            if d.is_dir() and (d / "metrics.json").exists():
                runs.append(d.name)
        if not runs:
            print(f"No runs with metrics.json found in {args.results_root}/")
            sys.exit(1)
    elif args.runs:
        runs = args.runs
    else:
        print("Specify --runs or --auto. Use -h for help.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  ABLATION COMPARISON — {len(runs)} runs")
    print(f"{'=' * 70}\n")

    all_metrics = []
    for run in runs:
        m = load_metrics(args.results_root, run)
        all_metrics.append(m)
        if m:
            print(f"  Loaded: {run}")

    print()
    print_console_table(runs, all_metrics)

    # Save outputs
    out_base = args.output or str(Path(args.results_root) / "ablation_comparison")
    save_csv(runs, all_metrics, f"{out_base}.csv")
    save_latex(runs, all_metrics, f"{out_base}.tex")

    print()


if __name__ == "__main__":
    main()
