import argparse
import os
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def find_bbox(img_u8: np.ndarray, thr: int) -> tuple[int, int, int, int] | None:
    """
    Return bbox (x0,y0,x1,y1) inclusive pixel indices for pixels > thr.
    None if no pixels exceed thr.
    """
    mask = img_u8 > thr
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, x1, y1


def pad_bbox(bbox, pad, H, W):
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return x0, y0, x1, y1


def combine_bboxes(bboxes, H, W, mode="quantile", q=0.02):
    """
    Combine per-image bboxes into one global bbox.
    mode:
      - "union": full union (can be too big if any outlier exists)
      - "quantile": robust aggregation (recommended)
    """
    xs0, ys0, xs1, ys1 = zip(*bboxes)
    xs0 = np.array(xs0); ys0 = np.array(ys0); xs1 = np.array(xs1); ys1 = np.array(ys1)

    if mode == "union":
        x0, y0 = int(xs0.min()), int(ys0.min())
        x1, y1 = int(xs1.max()), int(ys1.max())
        return x0, y0, x1, y1

    # quantile-based robust bbox
    # left/top: take small quantile (ignore rare far-left/top bright pixels)
    # right/bottom: take large quantile
    x0 = int(np.quantile(xs0, q))
    y0 = int(np.quantile(ys0, q))
    x1 = int(np.quantile(xs1, 1.0 - q))
    y1 = int(np.quantile(ys1, 1.0 - q))

    # clamp and sanity
    x0 = max(0, min(x0, W - 2))
    y0 = max(0, min(y0, H - 2))
    x1 = max(x0 + 1, min(x1, W - 1))
    y1 = max(y0 + 1, min(y1, H - 1))
    return x0, y0, x1, y1


def compute_global_crop(paths, thr, pad, sample_n=400, mode="quantile", q=0.02):
    """
    Compute a robust global crop bbox for a domain.
    """
    if len(paths) == 0:
        raise RuntimeError("No images found.")

    # sample for speed + robustness
    if sample_n and len(paths) > sample_n:
        paths = random.sample(paths, sample_n)

    bboxes = []
    H = W = None

    for p in tqdm(paths, desc="Scanning for bbox"):
        im = Image.open(p).convert("L")
        arr = np.array(im, dtype=np.uint8)
        if H is None:
            H, W = arr.shape
        bb = find_bbox(arr, thr)
        if bb is None:
            continue
        bb = pad_bbox(bb, pad, H, W)
        bboxes.append(bb)

    if len(bboxes) < 10:
        raise RuntimeError(f"Too few valid bboxes found ({len(bboxes)}). Try lowering --thr.")

    x0, y0, x1, y1 = combine_bboxes(bboxes, H, W, mode=mode, q=q)
    return (x0, y0, x1, y1), (H, W), len(bboxes)


def crop_and_resize(in_path, out_path, crop_box, out_size):
    im = Image.open(in_path).convert("L")
    arr = np.array(im, dtype=np.uint8)
    x0, y0, x1, y1 = crop_box

    # PIL crop uses (left, upper, right, lower) with right/lower EXCLUSIVE
    cropped = im.crop((x0, y0, x1 + 1, y1 + 1))
    resized = cropped.resize((out_size, out_size), resample=Image.BILINEAR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(out_path)


def process_split(indir, outdir, split, cropA, cropB, out_size):
    for domain, crop in [("A", cropA), ("B", cropB)]:
        src = Path(indir) / f"{split}{domain}"
        dst = Path(outdir) / f"{split}{domain}"
        paths = sorted(glob.glob(str(src / "*.png")))
        if not paths:
            raise RuntimeError(f"No PNGs in {src}")

        for p in tqdm(paths, desc=f"Writing {split}{domain}"):
            in_path = Path(p)
            out_path = dst / in_path.name
            crop_and_resize(in_path, out_path, crop, out_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Input dataset root (has trainA/trainB/testA/testB)")
    ap.add_argument("--outdir", required=True, help="Output dataset root")
    ap.add_argument("--thr", type=int, default=10, help="Threshold for non-black mask (0-255)")
    ap.add_argument("--pad", type=int, default=6, help="Padding added to crop bbox (pixels, in 256x256 space)")
    ap.add_argument("--sample_n", type=int, default=400, help="How many images to sample per domain for bbox estimation")
    ap.add_argument("--mode", choices=["quantile", "union"], default="quantile", help="How to combine bboxes")
    ap.add_argument("--q", type=float, default=0.02, help="Quantile for robust bbox (mode=quantile)")
    ap.add_argument("--out_size", type=int, default=256, help="Output size (square)")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    # Compute global crops from training sets only (more stable)
    trainA = sorted(glob.glob(str(indir / "trainA" / "*.png")))
    trainB = sorted(glob.glob(str(indir / "trainB" / "*.png")))

    print(f"Found trainA={len(trainA)}, trainB={len(trainB)}")

    cropA, (HA, WA), nA = compute_global_crop(trainA, args.thr, args.pad, args.sample_n, args.mode, args.q)
    cropB, (HB, WB), nB = compute_global_crop(trainB, args.thr, args.pad, args.sample_n, args.mode, args.q)

    print(f"\nA global crop (x0,y0,x1,y1): {cropA}  (H,W)=({HA},{WA})  from {nA} samples")
    print(f"B global crop (x0,y0,x1,y1): {cropB}  (H,W)=({HB},{WB})  from {nB} samples\n")

    # Write crops used (for reproducibility)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "crop_boxes.txt", "w") as f:
        f.write(f"thr={args.thr} pad={args.pad} mode={args.mode} q={args.q}\n")
        f.write(f"A_crop={cropA}\n")
        f.write(f"B_crop={cropB}\n")

    for split in ["train", "test"]:
        process_split(indir, outdir, split, cropA, cropB, args.out_size)

    print("\nDone.")
    print(f"Output written to: {outdir}")


if __name__ == "__main__":
    main()