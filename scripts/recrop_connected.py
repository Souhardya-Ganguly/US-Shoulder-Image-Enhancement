import argparse
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm


def largest_component_bbox(img, thr=30):
    mask = img > thr
    lbl = label(mask)
    props = regionprops(lbl)

    if len(props) == 0:
        return None

    # pick largest region
    largest = max(props, key=lambda x: x.area)
    minr, minc, maxr, maxc = largest.bbox  # note row/col order
    return minc, minr, maxc - 1, maxr - 1  # convert to x0,y0,x1,y1


def crop_resize(in_path, out_path, thr, pad, out_size):
    im = Image.open(in_path).convert("L")
    arr = np.array(im)

    bbox = largest_component_bbox(arr, thr=thr)
    if bbox is None:
        return

    x0, y0, x1, y1 = bbox

    # padding
    H, W = arr.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)

    cropped = im.crop((x0, y0, x1 + 1, y1 + 1))
    resized = cropped.resize((out_size, out_size), Image.BILINEAR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(out_path)


def process_split(indir, outdir, split, thr, pad, out_size):
    for domain in ["A", "B"]:
        src = Path(indir) / f"{split}{domain}"
        dst = Path(outdir) / f"{split}{domain}"

        paths = glob.glob(str(src / "*.png"))

        for p in tqdm(paths, desc=f"{split}{domain}"):
            out_path = dst / Path(p).name
            crop_resize(p, out_path, thr, pad, out_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thr", type=int, default=30)
    ap.add_argument("--pad", type=int, default=5)
    ap.add_argument("--out_size", type=int, default=256)
    args = ap.parse_args()

    for split in ["train", "test"]:
        process_split(args.indir, args.outdir, split, args.thr, args.pad, args.out_size)

    print("Done.")


if __name__ == "__main__":
    main()
