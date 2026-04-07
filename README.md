# POCUS Shoulder Ultrasound Image Enhancement

Unpaired image-to-image domain transfer to enhance low-quality shoulder ultrasound images from a **Telemed POCUS device** to match the quality of a **Philips device** (high-quality reference), using CycleGAN with perceptual loss ablations.

**Author:** Souhardya Ganguly, University of Alberta (Computing Science)

## Requirements

- Python 3.13+
- NVIDIA GPU with CUDA support (tested on RTX 4090, 24 GB VRAM)

### Python Packages

```
torch==2.6.0+cu124
torchvision==0.21.0+cu124
numpy==2.4.2
pillow==12.1.1
opencv-python==4.13.0.92
pydicom==3.0.1
scikit-image==0.26.0
scikit-learn==1.8.0
scipy==1.17.0
matplotlib==3.10.8
tqdm==4.67.3
lpips==0.1.4
wandb==0.25.1
```

### Setup

```bash
python -m venv venv
source venv/bin/activate

# PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Remaining packages
pip install numpy pillow opencv-python pydicom scikit-image scikit-learn \
            scipy matplotlib tqdm lpips wandb
```

## Preprocessing

```bash
# Inspect DICOM metadata
python scripts/inspect_dicom.py "POCUS Shoulder Data/Final US Data/1/1_Philips/001_left.dcm"

# Extract DICOMs to PNG with device-specific border cropping
python scripts/preprocess_cyclegan.py \
    --data_root "POCUS Shoulder Data/Final US Data" \
    --out_root datasets/us_cyclegan \
    --img_size 256 \
    --test_size 0.2 \
    --max_frames_per_dicom 30 \
    --frame_strategy uniform

# Detect crop boundaries for Telemed frames
python scripts/find_crop_box.py \
    --indir /path/to/extracted_frames \
    --thr 20 --pad 10

# Crop black vertical borders from images
python scripts/crop_black_borders.py \
    --input_dir datasets/us_cyclegan/trainB \
    --preview 3

python scripts/crop_black_borders.py \
    --input_dir datasets/us_cyclegan/trainB \
    --output_dir datasets/us_cyclegan/trainB_cropped \
    --run

# Re-crop PNGs using largest connected component
python scripts/recrop_connected.py \
    --indir datasets/us_cyclegan \
    --outdir datasets/us_cyclegan_v2 \
    --thr 30 --pad 5 --out_size 256

# Batch re-crop with global robust bounding box
python scripts/recrop_png_dataset.py \
    --indir datasets/us_cyclegan \
    --outdir datasets/us_cyclegan_v2 \
    --thr 10 --pad 6 --out_size 256

# Subject-level train/test split (prevents data leakage)
python scripts/split_by_subject.py \
    --input_dir datasets/us_cyclegan \
    --output_dir datasets/us_cyclegan_v2 \
    --test_ratio 0.2 --seed 42
```

## Training Ablations

All experiments use `--direction BtoA` so that the enhancement direction (Telemed to Philips) maps to `fake_B`.

```bash
cd cyclegan_repo

# Baseline CycleGAN
python train.py \
    --dataroot ../datasets/us_cyclegan \
    --name abl_BtoA_baseline \
    --model cycle_gan \
    --direction BtoA \
    --n_epochs 100 --n_epochs_decay 100 \
    --display_id -1 --gpu_ids 0 \
    --use_wandb

# + Perceptual loss (VGG19, lambda=1.0)
python train.py \
    --dataroot ../datasets/us_cyclegan \
    --name abl_BtoA_perceptual \
    --model cycle_gan \
    --direction BtoA \
    --lambda_perceptual 1.0 \
    --n_epochs 100 --n_epochs_decay 100 \
    --display_id -1 --gpu_ids 0 \
    --use_wandb

# No identity loss
python train.py \
    --dataroot ../datasets/us_cyclegan \
    --name abl_BtoA_no_identity \
    --model cycle_gan \
    --direction BtoA \
    --lambda_identity 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    --display_id -1 --gpu_ids 0 \
    --use_wandb

# Perceptual + no identity
python train.py \
    --dataroot ../datasets/us_cyclegan \
    --name abl_BtoA_perc_no_idt \
    --model cycle_gan \
    --direction BtoA \
    --lambda_perceptual 1.0 --lambda_identity 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    --display_id -1 --gpu_ids 0 \
    --use_wandb
```

## Inference

```bash
cd cyclegan_repo

for RUN in abl_BtoA_baseline abl_BtoA_perceptual abl_BtoA_no_identity abl_BtoA_perc_no_idt; do
    python test.py \
        --dataroot ../datasets/us_cyclegan \
        --name "$RUN" \
        --model cycle_gan \
        --direction BtoA \
        --results_dir ../results \
        --num_test 660
done
```

## Evaluation

```bash
# Compute metrics (SSIM, PSNR, FID, LPIPS, CNR, sharpness) for each run
for RUN in abl_BtoA_baseline abl_BtoA_perceptual abl_BtoA_no_identity abl_BtoA_perc_no_idt; do
    PHASE=$(ls -t results/"$RUN"/ | grep test_latest | head -1)
    python scripts/eval_results_metrics.py \
        --results_root results \
        --run_name "$RUN" \
        --phase "$PHASE"
done

# Generate comparison table (console + CSV + LaTeX)
python scripts/compare_ablations.py \
    --results_root results \
    --runs abl_BtoA_baseline abl_BtoA_perceptual abl_BtoA_no_identity abl_BtoA_perc_no_idt

# Or auto-detect all runs with metrics.json
python scripts/compare_ablations.py --results_root results --auto

# Plot training loss curves
python scripts/plot_loss_curves.py \
    --checkpoints_dir cyclegan_repo/checkpoints \
    --runs abl_BtoA_baseline abl_BtoA_perceptual abl_BtoA_no_identity abl_BtoA_perc_no_idt \
    --output_dir results/loss_curves

# Generate visual comparison grids (per-run and cross-run)
python scripts/visual_comparison.py \
    --results_root results \
    --runs abl_BtoA_baseline abl_BtoA_perceptual abl_BtoA_perc_no_idt \
    --num_samples 5 --mode both \
    --output_dir results/visual_comparisons_BtoA

# Log training time from a training log
python scripts/log_training_time.py \
    --log_file /path/to/training_output.log \
    --run_name abl_BtoA_baseline
```
