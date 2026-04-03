"""
Generate publication-ready visual comparison grids.

Two modes:
  per_run:    For each run, show N rows of [real_A | fake_B | real_B]
  cross_run:  For each sample, compare fake_B across all runs

Usage:
    # Both modes, auto-detect phases
    python scripts/visual_comparison.py \
        --results_root results \
        --runs ablation_baseline ablation_perceptual ablation_no_identity ablation_perc_no_idt \
        --num_samples 5 --mode both

    # Cross-run comparison only
    python scripts/visual_comparison.py \
        --results_root results \
        --runs ablation_baseline ablation_perc_no_idt \
        --mode cross_run --num_samples 3
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Layout constants
IMG_SIZE = 256
PAD = 4
LABEL_H = 28
BG_COLOR = (30, 30, 30)
LABEL_COLOR = (220, 220, 220)


def find_phase_dir(results_root: str, run_name: str) -> str:
    """Auto-detect the test_latest_* directory for a run."""
    run_dir = Path(results_root) / run_name
    candidates = sorted(run_dir.glob("test_latest_*"))
    if not candidates:
        raise FileNotFoundError(f"No test_latest_* dir in {run_dir}")
    return str(candidates[-1])  # most recent


def get_samples(images_dir: str):
    """Parse image directory into sample dict: prefix -> {type: path}."""
    samples = defaultdict(dict)
    for f in sorted(os.listdir(images_dir)):
        if not f.endswith(".png"):
            continue
        for suffix in ["_real_A", "_fake_B", "_real_B", "_fake_A", "_rec_A", "_rec_B"]:
            if f.endswith(suffix + ".png"):
                prefix = f[: -len(suffix) - 4]
                samples[prefix][suffix[1:]] = os.path.join(images_dir, f)
                break
    return dict(samples)


def get_font():
    """Try to get a readable font, fall back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except (OSError, IOError):
            return ImageFont.load_default()


def draw_label(draw, x, y, w, text, font):
    """Draw centered text label."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    tx = x + (w - tw) // 2
    draw.text((tx, y + 4), text, fill=LABEL_COLOR, font=font)


def per_run_grid(run_name: str, images_dir: str, prefixes: list, output_path: str):
    """Create grid: N rows x 3 cols [real_A | fake_B | real_B]."""
    samples = get_samples(images_dir)
    font = get_font()

    cols = 3
    col_labels = ["Input (Telemed)", "Enhanced (fake_B)", "Reference (Philips)"]
    keys = ["real_A", "fake_B", "real_B"]

    W = cols * IMG_SIZE + (cols + 1) * PAD
    H = LABEL_H * 2 + len(prefixes) * (IMG_SIZE + PAD) + PAD

    canvas = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Title
    draw_label(draw, 0, 0, W, run_name, font)

    # Column headers
    for j, label in enumerate(col_labels):
        x = PAD + j * (IMG_SIZE + PAD)
        draw_label(draw, x, LABEL_H, IMG_SIZE, label, font)

    # Image rows
    y_offset = LABEL_H * 2
    for i, prefix in enumerate(prefixes):
        sample = samples.get(prefix, {})
        for j, key in enumerate(keys):
            x = PAD + j * (IMG_SIZE + PAD)
            y = y_offset + i * (IMG_SIZE + PAD)
            if key in sample:
                img = Image.open(sample[key]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                canvas.paste(img, (x, y))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"  Saved: {output_path}")


def cross_run_grid(run_data: dict, prefixes: list, output_dir: str):
    """For each sample, show real_A | fake_B per run | real_B."""
    font = get_font()
    run_names = list(run_data.keys())
    cols = 2 + len(run_names)  # real_A + N runs + real_B

    for prefix in prefixes:
        W = cols * IMG_SIZE + (cols + 1) * PAD
        H = LABEL_H * 2 + IMG_SIZE + PAD * 2

        canvas = Image.new("RGB", (W, H), BG_COLOR)
        draw = ImageDraw.Draw(canvas)

        # Title
        draw_label(draw, 0, 0, W, f"Sample: {prefix}", font)

        # Column headers and images
        col_labels = ["Input (Telemed)"] + [rn.replace("ablation_", "") for rn in run_names] + ["Reference (Philips)"]
        col_keys = ["real_A"] + ["fake_B"] * len(run_names) + ["real_B"]

        y_img = LABEL_H * 2 + PAD
        for j, (label, key) in enumerate(zip(col_labels, col_keys)):
            x = PAD + j * (IMG_SIZE + PAD)
            draw_label(draw, x, LABEL_H, IMG_SIZE, label, font)

            # Pick image source
            if j == 0:  # real_A from first run that has it
                for rn in run_names:
                    sample = run_data[rn].get(prefix, {})
                    if "real_A" in sample:
                        img = Image.open(sample["real_A"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                        canvas.paste(img, (x, y_img))
                        break
            elif j == len(col_labels) - 1:  # real_B from first run that has it
                for rn in run_names:
                    sample = run_data[rn].get(prefix, {})
                    if "real_B" in sample:
                        img = Image.open(sample["real_B"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                        canvas.paste(img, (x, y_img))
                        break
            else:  # fake_B from specific run
                rn = run_names[j - 1]
                sample = run_data[rn].get(prefix, {})
                if "fake_B" in sample:
                    img = Image.open(sample["fake_B"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    canvas.paste(img, (x, y_img))

        out_path = os.path.join(output_dir, f"cross_run_{prefix}.png")
        canvas.save(out_path)
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visual comparison grids for thesis figures")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_dir", default="results/visual_comparisons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["per_run", "cross_run", "both"], default="both")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all runs
    run_data = {}
    for rn in args.runs:
        phase_dir = find_phase_dir(args.results_root, rn)
        images_dir = os.path.join(phase_dir, "images")
        run_data[rn] = get_samples(images_dir)
        print(f"Loaded {rn}: {len(run_data[rn])} samples from {images_dir}")

    # Find common sample prefixes across all runs
    common = set.intersection(*(set(d.keys()) for d in run_data.values()))
    common = sorted(common)
    print(f"\nCommon samples across all runs: {len(common)}")

    # Select samples
    random.seed(args.seed)
    selected = random.sample(common, min(args.num_samples, len(common)))
    print(f"Selected {len(selected)} samples: {selected}\n")

    if args.mode in ("per_run", "both"):
        print("=== Per-run grids ===")
        for rn in args.runs:
            phase_dir = find_phase_dir(args.results_root, rn)
            images_dir = os.path.join(phase_dir, "images")
            out_path = os.path.join(args.output_dir, f"per_run_{rn}.png")
            per_run_grid(rn, images_dir, selected, out_path)

    if args.mode in ("cross_run", "both"):
        print("\n=== Cross-run comparisons ===")
        cross_run_grid(run_data, selected, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
