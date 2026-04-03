import os
import re
import glob
import argparse
import random
from pathlib import Path

import numpy as np
import cv2
import pydicom
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Your computed crop for Philips (x0,y0,x1,y1)
PHILIPS_CROP = (228, 102, 788, 516)

# set this to your Telemed crop (x0,y0,x1,y1)
TELEMED_CROP = (120, 0, 900, 679)

import numpy as np

def has_telemed_black_side_borders(img: np.ndarray,
                                   side_width=80,
                                   dark_threshold=20,
                                   min_dark_fraction=0.85):
    """
    Detect whether a grayscale Telemed frame has strong black margins
    on the left/right sides.

    Parameters
    ----------
    img : np.ndarray
        Grayscale uint8 image (H, W)
    side_width : int
        Width of left/right strips to inspect.
    dark_threshold : int
        Pixel intensity threshold below which a pixel is considered dark.
    min_dark_fraction : float
        Minimum fraction of dark pixels in a side strip to call it a border.

    Returns
    -------
    bool
    """
    if img.ndim != 2:
        raise ValueError(f"Expected grayscale image, got {img.shape}")

    h, w = img.shape
    sw = min(side_width, w // 4)

    left_strip = img[:, :sw]
    right_strip = img[:, w - sw:]

    left_dark_frac = (left_strip <= dark_threshold).mean()
    right_dark_frac = (right_strip <= dark_threshold).mean()

    return (left_dark_frac >= min_dark_fraction) and (right_dark_frac >= min_dark_fraction)

def percentile_normalize_uint8(y: np.ndarray, low=1, high=99, mask_thr=5):
    """
    y: uint8 (H,W). Returns uint8 (H,W) normalized to 0..255 using robust percentiles.
    mask_thr: ignore pixels <= mask_thr when computing percentiles (helps with black background).
    """
    y_f = y.astype(np.float32)

    mask = y_f > mask_thr
    if mask.sum() < 100:  # fallback if almost all background
        mask = np.ones_like(y_f, dtype=bool)

    lo = np.percentile(y_f[mask], low)
    hi = np.percentile(y_f[mask], high)

    if hi <= lo + 1e-6:
        return y

    y_f = np.clip(y_f, lo, hi)
    y_f = (y_f - lo) * (255.0 / (hi - lo))
    return np.clip(y_f, 0, 255).astype(np.uint8)


def auto_crop_black_border(img: np.ndarray,
                           dark_threshold=18,
                           frac_dark_required=0.97,
                           min_border_px=12,
                           pad=3):
    """
    Remove black borders from grayscale ultrasound images.

    Strategy:
    - A column is considered 'dark' if at least frac_dark_required of its pixels
      are <= dark_threshold.
    - Same idea for rows.
    - Crop contiguous dark strips only from the outer edges.
    - If no substantial border is found, return the original image.

    Parameters
    ----------
    img : np.ndarray
        Grayscale uint8 image of shape (H, W)
    dark_threshold : int
        Pixel intensity threshold for considering a pixel dark/background.
    frac_dark_required : float
        Fraction of pixels in a row/column that must be dark to count as border.
    min_border_px : int
        Minimum border width to trigger cropping.
    pad : int
        Padding kept after cropping.

    Returns
    -------
    cropped_img : np.ndarray
    did_crop : bool
    bbox : tuple
        (x0, y0, x1, y1)
    """
    if img.ndim != 2:
        raise ValueError(f"Expected grayscale image (H,W), got shape={img.shape}")

    h, w = img.shape

    # Fraction of dark pixels in each column and row
    col_dark_frac = (img <= dark_threshold).mean(axis=0)
    row_dark_frac = (img <= dark_threshold).mean(axis=1)

    def edge_run_length(arr, from_start=True, threshold=frac_dark_required):
        n = len(arr)
        count = 0
        idxs = range(n) if from_start else range(n - 1, -1, -1)
        for i in idxs:
            if arr[i] >= threshold:
                count += 1
            else:
                break
        return count

    left_border = edge_run_length(col_dark_frac, from_start=True)
    right_border = edge_run_length(col_dark_frac, from_start=False)
    top_border = edge_run_length(row_dark_frac, from_start=True)
    bottom_border = edge_run_length(row_dark_frac, from_start=False)

    # Only crop meaningful borders
    crop_left = left_border if left_border >= min_border_px else 0
    crop_right = right_border if right_border >= min_border_px else 0
    crop_top = top_border if top_border >= min_border_px else 0
    crop_bottom = bottom_border if bottom_border >= min_border_px else 0

    if crop_left == 0 and crop_right == 0 and crop_top == 0 and crop_bottom == 0:
        return img, False, (0, 0, w - 1, h - 1)

    x0 = max(0, crop_left - pad)
    x1 = min(w - 1, w - 1 - crop_right + pad)
    y0 = max(0, crop_top - pad)
    y1 = min(h - 1, h - 1 - crop_bottom + pad)

    # Safety check
    if x1 <= x0 or y1 <= y0:
        return img, False, (0, 0, w - 1, h - 1)

    cropped = img[y0:y1 + 1, x0:x1 + 1]
    return cropped, True, (x0, y0, x1, y1)

def ybr_full_to_rgb(frame_uint8_hwc3: np.ndarray) -> np.ndarray:
    # frame is HxWx3 uint8 in YBR_FULL; convert via PIL YCbCr -> RGB
    return np.array(Image.fromarray(frame_uint8_hwc3, mode="YCbCr").convert("RGB"), dtype=np.uint8)

def rgb_to_luma(rgb_uint8_hwc3: np.ndarray) -> np.ndarray:
    r = rgb_uint8_hwc3[..., 0].astype(np.float32)
    g = rgb_uint8_hwc3[..., 1].astype(np.float32)
    b = rgb_uint8_hwc3[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)

def apply_crop(img: np.ndarray, crop):
    x0, y0, x1, y1 = crop
    return img[y0:y1+1, x0:x1+1]

def resize(img: np.ndarray, size: int) -> np.ndarray:
    # size x size
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def ensure_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)

def read_dicom_frames_as_rgb(ds) -> np.ndarray:
    """
    Returns frames as uint8 RGB array of shape (T,H,W,3).
    Handles RGB and YBR_FULL.
    """
    arr = ds.pixel_array
    if arr.ndim == 3:
        arr = arr[None, ...]  # T=1

    arr = ensure_uint8(arr)
    phot = str(getattr(ds, "PhotometricInterpretation", "")).upper()

    if phot == "RGB":
        return arr
    if phot == "YBR_FULL":
        rgb_frames = []
        for f in arr:
            rgb_frames.append(ybr_full_to_rgb(f))
        return np.stack(rgb_frames, axis=0)

    # Fallback: attempt to treat as RGB-like
    print(f"[WARN] Unhandled PhotometricInterpretation={phot}. Proceeding as-is.")
    return arr

def infer_domain(path_str: str):
    s = path_str.lower()
    if "philips" in s:
        return "A"  # Domain A
    if "telemed" in s:
        return "B"  # Domain B
    return None

def infer_patient_id(path: Path):
    """
    Patient-level split is critical.
    For your paths like:
      POCUS Shoulder Data/Final US Data/1/1_Philips/001_left.dcm
    This returns '1' as patient_id.
    If your structure changes, adjust this function.
    """
    parts = path.parts
    # find ".../Final US Data/<patient_id>/..."
    for i in range(len(parts)-1):
        if parts[i].lower() == "final us data":
            if i+1 < len(parts):
                return parts[i+1]
    # fallback: first integer-like folder in path
    for p in parts:
        if re.fullmatch(r"\d+", p):
            return p
    return "unknown"

def sample_frame_indices(T, max_frames_per_dicom, strategy="uniform"):
    if max_frames_per_dicom <= 0 or max_frames_per_dicom >= T:
        return list(range(T))
    if strategy == "uniform":
        idxs = np.linspace(0, T-1, num=max_frames_per_dicom, dtype=int)
        return sorted(set(map(int, idxs)))
    # random
    return sorted(random.sample(range(T), k=max_frames_per_dicom))

def save_frame(out_path: Path, img_uint8_hw: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_uint8_hw, mode="L").save(str(out_path))

def read_dicom_frames_as_gray(ds) -> np.ndarray:
    """
    Returns frames as uint8 grayscale array of shape (T,H,W).
    - If YBR_FULL (Telemed), uses Y channel directly (no conversion).
    - If RGB, converts to grayscale.
    - If already grayscale, uses as-is.
    """
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None, ...]  # (1,H,W)

    phot = str(getattr(ds, "PhotometricInterpretation", "")).upper()

    # Multi-frame color: (T,H,W,3) OR sometimes (H,W,3)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[None, ...]  # (1,H,W,3)

    if arr.ndim == 4 and arr.shape[-1] == 3:
        if phot == "YBR_FULL":
            y = arr[..., 0]  # take luma directly
            return y.astype(np.uint8)
        elif phot == "RGB":
            # convert each frame to gray
            grays = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in arr]
            return np.stack(grays, axis=0).astype(np.uint8)
        else:
            # fallback: take channel 0
            return arr[..., 0].astype(np.uint8)

    # If grayscale multi-frame already: (T,H,W)
    if arr.ndim == 3:
        return arr.astype(np.uint8)

    raise ValueError(f"Unexpected pixel_array shape: {arr.shape}, photometric={phot}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root folder that contains DICOMs (searches recursively)")
    ap.add_argument("--out_root", required=True, help="Output dataset root (e.g., datasets/us_cyclegan)")
    ap.add_argument("--img_size", type=int, default=256, help="Final size (square)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Patient-level test split fraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_frames_per_dicom", type=int, default=30, help="Limit frames extracted per DICOM (0=all)")
    ap.add_argument("--frame_strategy", choices=["uniform", "random"], default="uniform")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dicoms = [Path(p) for p in glob.glob(os.path.join(args.data_root, "**", "*.dcm"), recursive=True)]
    dicoms = sorted(dicoms)

    # Group by patient, and keep only files that are clearly Philips/Telemed
    per_patient = {}
    for p in dicoms:
        dom = infer_domain(str(p))
        if dom is None:
            continue
        pid = infer_patient_id(p)
        per_patient.setdefault(pid, []).append(p)

    patient_ids = sorted(per_patient.keys())
    if len(patient_ids) < 2:
        raise SystemExit(f"Found only {len(patient_ids)} patient ids. Check --data_root and infer_patient_id().")

    train_pids, test_pids = train_test_split(patient_ids, test_size=args.test_size, random_state=args.seed, shuffle=True)

    out_root = Path(args.out_root)
    (out_root / "trainA").mkdir(parents=True, exist_ok=True)
    (out_root / "trainB").mkdir(parents=True, exist_ok=True)
    (out_root / "testA").mkdir(parents=True, exist_ok=True)
    (out_root / "testB").mkdir(parents=True, exist_ok=True)

    def split_of(pid):  # patient-level split
        return "test" if pid in set(test_pids) else "train"

    # Counters for unique filenames
    counters = {"trainA": 0, "trainB": 0, "testA": 0, "testB": 0}

    for pid in tqdm(patient_ids, desc="Patients"):
        split = split_of(pid)
        files = per_patient[pid]

        for dcm_path in files:
            dom = infer_domain(str(dcm_path))  # A or B
            ds = pydicom.dcmread(str(dcm_path))
            frames_gray = read_dicom_frames_as_gray(ds)  # (T,H,W)

            T = frames_gray.shape[0]
            idxs = sample_frame_indices(T, args.max_frames_per_dicom, strategy=args.frame_strategy)

            for t in idxs:
                y = frames_gray[t]  # already grayscale uint8

                # ✅ FIX 1: domain crop logic is INSIDE the frame loop
                if dom == "A":
                    y = apply_crop(y, PHILIPS_CROP)
                elif dom == "B":
                    left_strip = y[:, :60]
                    right_strip = y[:, -60:]
                    left_dark_frac = (left_strip <= 15).mean()
                    right_dark_frac = (right_strip <= 15).mean()
                    print(f"[DEBUG] {dcm_path.name} frame={t} | H={y.shape[0]} W={y.shape[1]} | left_dark={left_dark_frac:.2f} right_dark={right_dark_frac:.2f}")

                    if has_telemed_black_side_borders(y, side_width=60, dark_threshold=15, min_dark_fraction=0.90):
                        print(f"[DEBUG] → Applying TELEMED_CROP {TELEMED_CROP}")
                        y = apply_crop(y, TELEMED_CROP)
                    else:
                        print(f"[DEBUG] → No crop applied")

                y = resize(y, args.img_size)

                folder = f"{split}{dom}"
                counters[folder] += 1

                stem = dcm_path.stem
                fname = f"{pid}_{stem}_f{t:04d}_{counters[folder]:07d}.png"
                save_frame(out_root / folder / fname, y)

    # Write a tiny manifest for reproducibility
    (out_root / "manifest.txt").write_text(
        f"data_root={args.data_root}\n"
        f"img_size={args.img_size}\n"
        f"test_size={args.test_size}\n"
        f"seed={args.seed}\n"
        f"max_frames_per_dicom={args.max_frames_per_dicom}\n"
        f"frame_strategy={args.frame_strategy}\n"
        f"PHILIPS_CROP={PHILIPS_CROP}\n"
        f"n_patients={len(patient_ids)}\n"
        f"n_train_patients={len(train_pids)}\n"
        f"n_test_patients={len(test_pids)}\n"
    )

    print("\nDone.")
    print("Output:", out_root)
    print("Example folders:", out_root/"trainA", out_root/"trainB", out_root/"testA", out_root/"testB")

if __name__ == "__main__":
    main()
