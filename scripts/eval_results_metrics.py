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


def compute_fid(samples: Dict[str, Dict[str, str]],
                device: str,
                max_images: Optional[int] = None,
                batch_size: int = 16) -> Dict[str, float]:
    """
    Compute FID between fake_A and real_A sets.
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

    if len(fakeA_paths) < 10:
        return {"fid_fakeA_realA": float("nan"), "n_fid_images": int(len(fakeA_paths))}

    feats_fake = extract_inception_features(fakeA_paths, device=device, batch_size=batch_size)
    feats_real = extract_inception_features(realA_paths, device=device, batch_size=batch_size)

    mu_f, sig_f = compute_stats(feats_fake)
    mu_r, sig_r = compute_stats(feats_real)

    fid = frechet_distance(mu_f, sig_f, mu_r, sig_r)
    return {"fid_fakeA_realA": float(fid), "n_fid_images": int(len(fakeA_paths))}


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

    # FID (distribution)
    metrics.update(compute_fid(samples, device=args.device, max_images=args.max_images, batch_size=args.fid_batch))

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

    # Print summary
    print("=== Summary ===")
    for k in ["fid_fakeA_realA",
              "hist_kl_fakeA_realA",
              "ssim_recB_realB_mean", "psnr_recB_realB_mean",
              "ssim_recA_realA_mean", "psnr_recA_realA_mean"]:
        if k in metrics:
            print(f"{k}: {metrics[k]}")

if __name__ == "__main__":
    main()