"""
Crop black borders (vertical strips of near-black pixels) from ultrasound images.

Walks inward from left and right edges column-by-column until the column mean
exceeds a threshold, then crops to the content region and resizes back to the
original dimensions.

Usage:
    # Preview mode (default) — shows before/after for first N images
    python scripts/crop_black_borders.py \
        --input_dir datasets/us_cyclegan_v4/testB \
        --preview 3

    # Run on full directory, save to separate output dir (originals untouched)
    python scripts/crop_black_borders.py \
        --input_dir datasets/us_cyclegan_v4/testB \
        --output_dir datasets/us_cyclegan_v4/testB_cropped \
        --run

    # Run in-place (overwrites originals)
    python scripts/crop_black_borders.py \
        --input_dir datasets/us_cyclegan_v4/testB \
        --run --inplace
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image


def detect_borders(img_gray: np.ndarray, threshold: float = 10.0):
    """Walk inward from left/right edges; return (left_crop, right_crop) column indices.

    Returns the slice [left_crop : right_crop] that contains the content region.
    """
    h, w = img_gray.shape

    left = 0
    for c in range(w):
        if img_gray[:, c].mean() >= threshold:
            left = c
            break

    right = w
    for c in range(w - 1, -1, -1):
        if img_gray[:, c].mean() >= threshold:
            right = c + 1
            break

    return left, right


def crop_image(img: Image.Image, threshold: float = 10.0):
    """Detect and crop black left/right borders. Returns (cropped_img, left, right, original_width)."""
    gray = np.array(img.convert("L"))
    left, right = detect_borders(gray, threshold)
    w = img.width

    if left == 0 and right == w:
        return img, left, right, w  # no border detected

    # Crop content region (full height, trimmed width)
    cropped = img.crop((left, 0, right, img.height))
    return cropped, left, right, w


def preview(input_dir: str, n: int, threshold: float):
    """Save a side-by-side before/after comparison for the first n images."""
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        print("No images found.")
        return

    preview_dir = os.path.join(os.path.dirname(input_dir.rstrip("/")), "border_crop_preview")
    os.makedirs(preview_dir, exist_ok=True)

    for fname in files[:n]:
        path = os.path.join(input_dir, fname)
        img = Image.open(path).convert("RGB")
        cropped, left, right, orig_w = crop_image(img, threshold)

        # Resize cropped back to original size for visual comparison
        cropped_resized = cropped.resize((img.width, img.height), Image.LANCZOS)

        # Side-by-side canvas
        canvas = Image.new("RGB", (img.width * 2 + 10, img.height), (40, 40, 40))
        canvas.paste(img, (0, 0))
        canvas.paste(cropped_resized, (img.width + 10, 0))

        out_path = os.path.join(preview_dir, f"preview_{fname}")
        canvas.save(out_path)
        print(f"{fname}: crop [{left}:{right}] of {orig_w}px  →  {right - left}px content  |  saved {out_path}")

    print(f"\n{n} previews saved to {preview_dir}/")


def run(input_dir: str, output_dir: str, inplace: bool, threshold: float, resize_back: bool):
    """Crop all images in input_dir."""
    if not inplace and output_dir is None:
        print("Error: specify --output_dir or --inplace")
        sys.exit(1)

    if not inplace:
        os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    cropped_count = 0
    skipped_count = 0

    for fname in files:
        path = os.path.join(input_dir, fname)
        img = Image.open(path).convert("RGB")
        orig_size = (img.width, img.height)
        cropped, left, right, orig_w = crop_image(img, threshold)

        if left == 0 and right == orig_w:
            skipped_count += 1
            if not inplace and output_dir:
                # Copy unchanged
                img.save(os.path.join(output_dir, fname))
            print(f"  SKIP  {fname} (no border detected)")
            continue

        if resize_back:
            cropped = cropped.resize(orig_size, Image.LANCZOS)

        save_path = path if inplace else os.path.join(output_dir, fname)
        cropped.save(save_path)
        cropped_count += 1
        print(f"  CROP  {fname}: [{left}:{right}] of {orig_w}px → {right - left}px content")

    print(f"\nDone. Cropped: {cropped_count}, Skipped (no border): {skipped_count}, Total: {len(files)}")


def main():
    parser = argparse.ArgumentParser(description="Crop black left/right borders from ultrasound images")
    parser.add_argument("--input_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_dir", default=None, help="Output directory (if not --inplace)")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Column mean threshold to consider as border (default: 10)")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="Preview mode: show before/after for first N images, then exit")
    parser.add_argument("--run", action="store_true", help="Actually crop and save images")
    parser.add_argument("--inplace", action="store_true", help="Overwrite originals (use with --run)")
    parser.add_argument("--no_resize", action="store_true",
                        help="Do NOT resize cropped images back to original dimensions")

    args = parser.parse_args()

    if args.preview > 0:
        preview(args.input_dir, args.preview, args.threshold)
    elif args.run:
        run(args.input_dir, args.output_dir, args.inplace, args.threshold, resize_back=not args.no_resize)
    else:
        print("Specify --preview N or --run. Use -h for help.")


if __name__ == "__main__":
    main()
