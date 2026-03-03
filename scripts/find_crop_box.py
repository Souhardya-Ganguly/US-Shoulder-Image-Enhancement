import os, glob, argparse
import numpy as np
import cv2

def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1, y1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Folder with *_Y.png frames")
    ap.add_argument("--thr", type=int, default=20, help="threshold for non-background")
    ap.add_argument("--pad", type=int, default=10, help="padding around bbox")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*_Y.png")))
    if not files:
        raise SystemExit(f"No *_Y.png files found in {args.indir}")

    bboxes = []
    shapes = set()

    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        shapes.add((h,w))

        # mask of "non-black" pixels
        mask = (img > args.thr).astype(np.uint8)

        # clean small speckles
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=1)

        bb = bbox_from_mask(mask)
        if bb is not None:
            bboxes.append(bb)

    if not bboxes:
        raise SystemExit("Could not find any bounding boxes. Try lowering --thr.")

    bboxes = np.array(bboxes)
    # robust aggregate crop: take percentile bounds (avoids outliers)
    x0 = int(np.percentile(bboxes[:,0], 10))
    y0 = int(np.percentile(bboxes[:,1], 10))
    x1 = int(np.percentile(bboxes[:,2], 90))
    y1 = int(np.percentile(bboxes[:,3], 90))

    # apply padding + clamp
    h, w = list(shapes)[0]
    x0 = max(0, x0 - args.pad)
    y0 = max(0, y0 - args.pad)
    x1 = min(w-1, x1 + args.pad)
    y1 = min(h-1, y1 + args.pad)

    print("Shapes seen:", shapes)
    print("Recommended crop box (x0,y0,x1,y1):", (x0,y0,x1,y1))
    print("Crop size (H,W):", (y1-y0+1, x1-x0+1))

if __name__ == "__main__":
    main()
