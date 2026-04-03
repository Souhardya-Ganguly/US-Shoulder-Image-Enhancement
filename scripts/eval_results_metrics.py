#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import inception_v3

import cv2
import lpips

from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.models import Inception_V3_Weights


# ----------------------------
# Helpers: image loading
# ----------------------------
def load_grayscale_uint8(path: str) -> np.ndarray:
    """Load PNG as grayscale uint8 array (H,W)."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def list_cycle_gan_images(images_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse CycleGAN test images folder. Returns:
      samples[prefix] = {"real_A": path, "fake_A": path, "rec_A": path, "real_B": path, ...}
    It matches filenames like:
      <prefix>_real_A.png
      <prefix>_fake_A.png
      <prefix>_rec_A.png
      <prefix>_real_B.png
      <prefix>_fake_B.png
      <prefix>_rec_B.png
    """
    patterns = ["*_real_A.png", "*_fake_A.png", "*_rec_A.png",
                "*_real_B.png", "*_fake_B.png", "*_rec_B.png"]
    files = []
    for p in patterns:
        files.extend(images_dir.glob(p))

    samples: Dict[str, Dict[str, str]] = {}
    for f in files:
        name = f.name
        # Find tag suffix
        tag = None
        for t in ["real_A", "fake_A", "rec_A", "real_B", "fake_B", "rec_B"]:
            if name.endswith(f"_{t}.png"):
                tag = t
                prefix = name[:-(len(t) + 5)]  # remove _{tag}.png
                break
        if tag is None:
            continue
        samples.setdefault(prefix, {})[tag] = str(f)

    return samples


# ----------------------------
# Metrics: SSIM / PSNR on recon
# ----------------------------
def compute_recon_metrics(samples: Dict[str, Dict[str, str]],
                          max_pairs: Optional[int] = None) -> Dict[str, float]:
    """
    Compute SSIM/PSNR for recon pairs:
      rec_A vs real_A and rec_B vs real_B (paired, meaningful).
    """
    ssim_A, psnr_A = [], []
    ssim_B, psnr_B = [], []

    keys = sorted(samples.keys())
    if max_pairs is not None:
        keys = keys[:max_pairs]

    for k in tqdm(keys, desc="Recon SSIM/PSNR"):
        d = samples[k]

        if "real_A" in d and "rec_A" in d:
            a = load_grayscale_uint8(d["real_A"])
            ra = load_grayscale_uint8(d["rec_A"])
            # SSIM expects floats or uint8 ok if data_range set
            ssim_A.append(ssim(a, ra, data_range=255))
            psnr_A.append(psnr(a, ra, data_range=255))

        if "real_B" in d and "rec_B" in d:
            b = load_grayscale_uint8(d["real_B"])
            rb = load_grayscale_uint8(d["rec_B"])
            ssim_B.append(ssim(b, rb, data_range=255))
            psnr_B.append(psnr(b, rb, data_range=255))

    out = {}
    if ssim_A:
        out["ssim_recA_realA_mean"] = float(np.mean(ssim_A))
        out["psnr_recA_realA_mean"] = float(np.mean(psnr_A))
        out["n_pairs_A"] = int(len(ssim_A))
    else:
        out["ssim_recA_realA_mean"] = float("nan")
        out["psnr_recA_realA_mean"] = float("nan")
        out["n_pairs_A"] = 0

    if ssim_B:
        out["ssim_recB_realB_mean"] = float(np.mean(ssim_B))
        out["psnr_recB_realB_mean"] = float(np.mean(psnr_B))
        out["n_pairs_B"] = int(len(ssim_B))
    else:
        out["ssim_recB_realB_mean"] = float("nan")
        out["psnr_recB_realB_mean"] = float("nan")
        out["n_pairs_B"] = 0

    return out


# ----------------------------
# Histogram KL divergence (fake_A vs real_A)
# ----------------------------
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P || Q) with small epsilon smoothing."""
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_hist_kl(samples: Dict[str, Dict[str, str]],
                    bins: int = 256,
                    max_images: Optional[int] = None) -> Dict[str, float]:
    """
    Compute histogram KL divergence between distributions:
      KL( hist(fake_A) || hist(real_A) ) and KL( hist(real_A) || hist(fake_A) )
    using aggregated histograms over all pixels.
    """
    fakeA_paths, realA_paths = [], []
    for k in sorted(samples.keys()):
        d = samples[k]
        if "fake_A" in d and "real_A" in d:
            fakeA_paths.append(d["fake_A"])
            realA_paths.append(d["real_A"])

    if max_images is not None:
        fakeA_paths = fakeA_paths[:max_images]
        realA_paths = realA_paths[:max_images]

    if not fakeA_paths:
        return {
            "hist_kl_fakeA_realA": float("nan"),
            "hist_kl_realA_fakeA": float("nan"),
            "n_hist_images": 0
        }

    hist_fake = np.zeros((bins,), dtype=np.float64)
    hist_real = np.zeros((bins,), dtype=np.float64)

    for fp, rp in tqdm(list(zip(fakeA_paths, realA_paths)), desc="Histogram KL"):
        fa = load_grayscale_uint8(fp).ravel()
        ra = load_grayscale_uint8(rp).ravel()

        hf, _ = np.histogram(fa, bins=bins, range=(0, 255))
        hr, _ = np.histogram(ra, bins=bins, range=(0, 255))
        hist_fake += hf
        hist_real += hr

    kl_f_r = kl_divergence(hist_fake, hist_real)
    kl_r_f = kl_divergence(hist_real, hist_fake)

    return {
        "hist_kl_fakeA_realA": float(kl_f_r),
        "hist_kl_realA_fakeA": float(kl_r_f),
        "n_hist_images": int(len(fakeA_paths))
    }


# ----------------------------
# FID (fake_A vs real_A) using Inception features
# ----------------------------
class InceptionV3Features(nn.Module):
    """
    InceptionV3 feature extractor returning 2048-d pooled features.
    Uses pretrained ImageNet weights. With aux_logits=True (required by torchvision when weights are set).
    """
    def __init__(self):
        super().__init__()
        m = inception_v3(
            weights=Inception_V3_Weights.DEFAULT,
            transform_input=False,
            aux_logits=True,   # REQUIRED when weights are provided in newer torchvision
        )
        # Replace the classifier with identity so forward() returns 2048-d features
        m.fc = nn.Identity()
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,H,W) -> (N,2048)
        return self.model(x)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(fid)


@torch.no_grad()
def extract_inception_features(paths: List[str],
                               device: str,
                               batch_size: int = 16,
                               resize: int = 299) -> np.ndarray:
    model = InceptionV3Features().to(device)

    tfm = T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),  # [0,1]
        # Inception expects approx ImageNet normalized inputs
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Inception feats"):
        batch_paths = paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            im = Image.open(p).convert("L")       # grayscale
            im = im.convert("RGB")                # replicate to 3-channel
            imgs.append(tfm(im))
        x = torch.stack(imgs, dim=0).to(device)
        f = model(x).cpu().numpy()
        feats.append(f)

    return np.concatenate(feats, axis=0)


def compute_fid_pair(fake_paths: List[str], real_paths: List[str],
                     device: str, batch_size: int, label: str) -> Dict[str, float]:
    """Compute FID between two sets of image paths."""
    if len(fake_paths) < 10:
        return {f"fid_{label}": float("nan"), f"n_fid_{label}": int(len(fake_paths))}

    feats_fake = extract_inception_features(fake_paths, device=device, batch_size=batch_size)
    feats_real = extract_inception_features(real_paths, device=device, batch_size=batch_size)

    mu_f, sig_f = compute_stats(feats_fake)
    mu_r, sig_r = compute_stats(feats_real)

    fid = frechet_distance(mu_f, sig_f, mu_r, sig_r)
    return {f"fid_{label}": float(fid), f"n_fid_{label}": int(len(fake_paths))}


def compute_fid(samples: Dict[str, Dict[str, str]],
                device: str,
                max_images: Optional[int] = None,
                batch_size: int = 16) -> Dict[str, float]:
    """
    Compute FID for both directions:
      fake_A vs real_A  (backward: B->A quality)
      fake_B vs real_B  (forward: A->B quality — the enhancement direction)
    """
    fakeA_paths, realA_paths = [], []
    fakeB_paths, realB_paths = [], []
    for k in sorted(samples.keys()):
        d = samples[k]
        if "fake_A" in d and "real_A" in d:
            fakeA_paths.append(d["fake_A"])
            realA_paths.append(d["real_A"])
        if "fake_B" in d and "real_B" in d:
            fakeB_paths.append(d["fake_B"])
            realB_paths.append(d["real_B"])

    if max_images is not None:
        fakeA_paths = fakeA_paths[:max_images]
        realA_paths = realA_paths[:max_images]
        fakeB_paths = fakeB_paths[:max_images]
        realB_paths = realB_paths[:max_images]

    out = {}
    out.update(compute_fid_pair(fakeA_paths, realA_paths, device, batch_size, "fakeA_realA"))
    out.update(compute_fid_pair(fakeB_paths, realB_paths, device, batch_size, "fakeB_realB"))
    return out


# ----------------------------
# LPIPS (perceptual similarity on reconstructions)
# ----------------------------
@torch.no_grad()
def compute_lpips(samples: Dict[str, Dict[str, str]],
                  device: str,
                  max_pairs: Optional[int] = None) -> Dict[str, float]:
    """
    Compute LPIPS (lower = more similar) on cycle reconstruction pairs:
      rec_A vs real_A and rec_B vs real_B.
    """
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    lpips_A, lpips_B = [], []

    keys = sorted(samples.keys())
    if max_pairs is not None:
        keys = keys[:max_pairs]

    for k in tqdm(keys, desc="LPIPS"):
        d = samples[k]

        if "real_A" in d and "rec_A" in d:
            a = Image.open(d["real_A"]).convert("L").convert("RGB")
            ra = Image.open(d["rec_A"]).convert("L").convert("RGB")
            tfm = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])  # [-1, 1]
            ta = tfm(a).unsqueeze(0).to(device)
            tra = tfm(ra).unsqueeze(0).to(device)
            lpips_A.append(loss_fn(ta, tra).item())

        if "real_B" in d and "rec_B" in d:
            b = Image.open(d["real_B"]).convert("L").convert("RGB")
            rb = Image.open(d["rec_B"]).convert("L").convert("RGB")
            tfm = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
            tb = tfm(b).unsqueeze(0).to(device)
            trb = tfm(rb).unsqueeze(0).to(device)
            lpips_B.append(loss_fn(tb, trb).item())

    out = {}
    out["lpips_recA_realA_mean"] = float(np.mean(lpips_A)) if lpips_A else float("nan")
    out["lpips_recB_realB_mean"] = float(np.mean(lpips_B)) if lpips_B else float("nan")
    return out


# ----------------------------
# Sharpness (Tenengrad — Sobel gradient magnitude)
# ----------------------------
def tenengrad_sharpness(img_gray: np.ndarray) -> float:
    """Tenengrad sharpness: mean of squared Sobel gradient magnitudes."""
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def compute_sharpness(samples: Dict[str, Dict[str, str]],
                      max_images: Optional[int] = None) -> Dict[str, float]:
    """
    Compute Tenengrad sharpness for fake_B, real_B, and their ratio.
    Ratio close to 1.0 means the enhanced images preserve edge sharpness.
    > 1.0 means sharper than reference, < 1.0 means blurrier.
    """
    sharp_fakeB, sharp_realB = [], []
    sharp_fakeA, sharp_realA = [], []

    keys = sorted(samples.keys())
    if max_images is not None:
        keys = keys[:max_images]

    for k in tqdm(keys, desc="Sharpness"):
        d = samples[k]
        if "fake_B" in d:
            sharp_fakeB.append(tenengrad_sharpness(load_grayscale_uint8(d["fake_B"])))
        if "real_B" in d:
            sharp_realB.append(tenengrad_sharpness(load_grayscale_uint8(d["real_B"])))
        if "fake_A" in d:
            sharp_fakeA.append(tenengrad_sharpness(load_grayscale_uint8(d["fake_A"])))
        if "real_A" in d:
            sharp_realA.append(tenengrad_sharpness(load_grayscale_uint8(d["real_A"])))

    out = {}
    out["sharpness_fakeB_mean"] = float(np.mean(sharp_fakeB)) if sharp_fakeB else float("nan")
    out["sharpness_realB_mean"] = float(np.mean(sharp_realB)) if sharp_realB else float("nan")
    out["sharpness_fakeA_mean"] = float(np.mean(sharp_fakeA)) if sharp_fakeA else float("nan")
    out["sharpness_realA_mean"] = float(np.mean(sharp_realA)) if sharp_realA else float("nan")

    if sharp_fakeB and sharp_realB:
        out["sharpness_ratio_fakeB_realB"] = out["sharpness_fakeB_mean"] / out["sharpness_realB_mean"]
    else:
        out["sharpness_ratio_fakeB_realB"] = float("nan")

    return out


# ----------------------------
# CNR (Contrast-to-Noise Ratio)
# ----------------------------
def compute_cnr_single(img_gray: np.ndarray,
                       bright_pct: float = 90,
                       dark_low_pct: float = 20,
                       dark_high_pct: float = 40) -> float:
    """
    Automatic CNR for ultrasound images.

    Bright ROI: pixels above the bright_pct percentile (humerus cortex / bright structures).
    Dark ROI: pixels between dark_low_pct and dark_high_pct percentiles (soft tissue).

    CNR = |mean_bright - mean_dark| / sqrt((var_bright + var_dark) / 2)
    """
    img = img_gray.astype(np.float64)
    bright_thresh = np.percentile(img, bright_pct)
    dark_low = np.percentile(img, dark_low_pct)
    dark_high = np.percentile(img, dark_high_pct)

    bright_pixels = img[img >= bright_thresh]
    dark_pixels = img[(img >= dark_low) & (img <= dark_high)]

    if len(bright_pixels) < 10 or len(dark_pixels) < 10:
        return float("nan")

    mean_b = bright_pixels.mean()
    mean_d = dark_pixels.mean()
    var_b = bright_pixels.var()
    var_d = dark_pixels.var()

    denom = np.sqrt((var_b + var_d) / 2.0)
    if denom < 1e-8:
        return float("nan")

    return float(abs(mean_b - mean_d) / denom)


def compute_cnr(samples: Dict[str, Dict[str, str]],
                max_images: Optional[int] = None) -> Dict[str, float]:
    """
    Compute CNR for fake_B (enhanced), real_B (Philips reference),
    real_A (Telemed source), and fake_A (degraded Philips).
    Higher CNR = better contrast between cortex and tissue.
    """
    cnr_fakeB, cnr_realB, cnr_realA, cnr_fakeA = [], [], [], []

    keys = sorted(samples.keys())
    if max_images is not None:
        keys = keys[:max_images]

    for k in tqdm(keys, desc="CNR"):
        d = samples[k]
        if "fake_B" in d:
            cnr_fakeB.append(compute_cnr_single(load_grayscale_uint8(d["fake_B"])))
        if "real_B" in d:
            cnr_realB.append(compute_cnr_single(load_grayscale_uint8(d["real_B"])))
        if "real_A" in d:
            cnr_realA.append(compute_cnr_single(load_grayscale_uint8(d["real_A"])))
        if "fake_A" in d:
            cnr_fakeA.append(compute_cnr_single(load_grayscale_uint8(d["fake_A"])))

    out = {}
    out["cnr_fakeB_mean"] = float(np.nanmean(cnr_fakeB)) if cnr_fakeB else float("nan")
    out["cnr_realB_mean"] = float(np.nanmean(cnr_realB)) if cnr_realB else float("nan")
    out["cnr_realA_mean"] = float(np.nanmean(cnr_realA)) if cnr_realA else float("nan")
    out["cnr_fakeA_mean"] = float(np.nanmean(cnr_fakeA)) if cnr_fakeA else float("nan")

    if cnr_fakeB and cnr_realB:
        out["cnr_ratio_fakeB_realB"] = out["cnr_fakeB_mean"] / out["cnr_realB_mean"]
    else:
        out["cnr_ratio_fakeB_realB"] = float("nan")

    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results", help="Path to results/ folder")
    ap.add_argument("--run_name", required=True, help="CycleGAN run name under results/")
    ap.add_argument("--phase", default="test_latest", help="e.g. test_latest or test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_images", type=int, default=None, help="Limit images for faster eval")
    ap.add_argument("--fid_batch", type=int, default=16, help="Batch size for FID feature extraction")
    ap.add_argument("--hist_bins", type=int, default=256)
    args = ap.parse_args()

    run_dir = Path(args.results_root) / args.run_name / args.phase / "images"
    if not run_dir.exists():
        raise FileNotFoundError(f"Could not find: {run_dir}")

    samples = list_cycle_gan_images(run_dir)
    if not samples:
        raise RuntimeError(f"No CycleGAN images found in: {run_dir}")

    metrics = {}
    metrics["run_name"] = args.run_name
    metrics["images_dir"] = str(run_dir)
    metrics["max_images"] = args.max_images

    # recon SSIM/PSNR (paired)
    metrics.update(compute_recon_metrics(samples, max_pairs=args.max_images))

    # histogram KL (distribution)
    metrics.update(compute_hist_kl(samples, bins=args.hist_bins, max_images=args.max_images))

    # FID (distribution — both directions)
    metrics.update(compute_fid(samples, device=args.device, max_images=args.max_images, batch_size=args.fid_batch))

    # LPIPS (perceptual similarity on reconstructions)
    metrics.update(compute_lpips(samples, device=args.device, max_pairs=args.max_images))

    # Sharpness (Tenengrad)
    metrics.update(compute_sharpness(samples, max_images=args.max_images))

    # CNR (Contrast-to-Noise Ratio)
    metrics.update(compute_cnr(samples, max_images=args.max_images))

    # Save outputs
    out_json = Path(args.results_root) / args.run_name / "metrics.json"
    out_csv = Path(args.results_root) / args.run_name / "metrics.csv"

    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {out_json}")

    # simple 1-row csv
    keys = sorted(metrics.keys())
    with open(out_csv, "w") as f:
        f.write(",".join(keys) + "\n")
        f.write(",".join(str(metrics[k]) for k in keys) + "\n")
    print(f"Saved: {out_csv}\n")

    # Print results table
    print(f"\n{'=' * 70}")
    print(f"  EVALUATION RESULTS: {args.run_name}")
    print(f"{'=' * 70}")

    def row(metric, value, direction, description):
        val_str = f"{value:.4f}" if isinstance(value, float) and not np.isnan(value) else str(value)
        print(f"  {metric:<35s} {val_str:>10s}  {direction:<6s}  {description}")

    print(f"\n  {'Metric':<35s} {'Value':>10s}  {'Goal':<6s}  Description")
    print(f"  {'-' * 66}")

    # --- Translation Quality (A->B: the enhancement direction) ---
    print(f"\n  ** Translation Quality (A->B Enhancement) **")
    row("FID (fake_B vs real_B)",       metrics.get("fid_fakeB_realB", float("nan")),     "(low)", "Distribution similarity to Philips")
    row("Sharpness fake_B",             metrics.get("sharpness_fakeB_mean", float("nan")), "",      "Tenengrad on enhanced images")
    row("Sharpness real_B",             metrics.get("sharpness_realB_mean", float("nan")), "",      "Tenengrad on real Philips")
    row("Sharpness ratio (fB/rB)",      metrics.get("sharpness_ratio_fakeB_realB", float("nan")), "(~1.0)", "Edge preservation")
    row("CNR fake_B",                   metrics.get("cnr_fakeB_mean", float("nan")),         "(high)", "Contrast-to-noise on enhanced")
    row("CNR real_B",                   metrics.get("cnr_realB_mean", float("nan")),         "",       "Contrast-to-noise on real Philips")
    row("CNR real_A",                   metrics.get("cnr_realA_mean", float("nan")),         "",       "Contrast-to-noise on real Telemed")
    row("CNR ratio (fB/rB)",            metrics.get("cnr_ratio_fakeB_realB", float("nan")), "(~1.0)", "CNR preservation vs Philips")

    # --- Backward Translation (B->A) ---
    print(f"\n  ** Backward Translation (B->A) **")
    row("FID (fake_A vs real_A)",       metrics.get("fid_fakeA_realA", float("nan")),     "(low)", "Distribution similarity to Telemed")
    row("Hist KL (fake_A || real_A)",   metrics.get("hist_kl_fakeA_realA", float("nan")), "(low)", "Intensity distribution match")

    # --- Cycle Consistency ---
    print(f"\n  ** Cycle Consistency (Reconstruction) **")
    row("SSIM (rec_A vs real_A)",       metrics.get("ssim_recA_realA_mean", float("nan")), "(high)", "Structural similarity after A->B->A")
    row("PSNR (rec_A vs real_A)",       metrics.get("psnr_recA_realA_mean", float("nan")), "(high)", "Pixel fidelity after A->B->A")
    row("LPIPS (rec_A vs real_A)",      metrics.get("lpips_recA_realA_mean", float("nan")), "(low)", "Perceptual distance after A->B->A")
    row("SSIM (rec_B vs real_B)",       metrics.get("ssim_recB_realB_mean", float("nan")), "(high)", "Structural similarity after B->A->B")
    row("PSNR (rec_B vs real_B)",       metrics.get("psnr_recB_realB_mean", float("nan")), "(high)", "Pixel fidelity after B->A->B")
    row("LPIPS (rec_B vs real_B)",      metrics.get("lpips_recB_realB_mean", float("nan")), "(low)", "Perceptual distance after B->A->B")

    print(f"\n  Samples: A={metrics.get('n_pairs_A', 0)}, B={metrics.get('n_pairs_B', 0)}")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()