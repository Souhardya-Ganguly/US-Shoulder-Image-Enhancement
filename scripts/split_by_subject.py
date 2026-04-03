"""
Create a subject-level train/test split for the CycleGAN dataset.

Ensures no subject appears in both train and test sets, preventing data leakage.
Only subjects that appear in BOTH domains are eligible for the test set.

Usage:
    python scripts/split_by_subject.py \
        --input_dir datasets/us_cyclegan \
        --output_dir datasets/us_cyclegan_v7 \
        --test_ratio 0.2 --seed 42
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def extract_subject_id(filename: str) -> int:
    """Extract subject ID from filename. First token before '_'."""
    return int(filename.split("_")[0])


def gather_images(input_dir: str):
    """Gather all images from train+test for both domains, grouped by subject."""
    subjects_A = defaultdict(list)  # subject_id -> [file paths]
    subjects_B = defaultdict(list)

    for split in ["trainA", "testA"]:
        d = os.path.join(input_dir, split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                sid = extract_subject_id(f)
                subjects_A[sid].append(os.path.join(d, f))

    for split in ["trainB", "testB"]:
        d = os.path.join(input_dir, split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                sid = extract_subject_id(f)
                subjects_B[sid].append(os.path.join(d, f))

    return subjects_A, subjects_B


def split_subjects(subjects_A, subjects_B, test_ratio, seed):
    """Split subjects into train/test, ensuring test subjects exist in both domains."""
    all_A = set(subjects_A.keys())
    all_B = set(subjects_B.keys())
    shared = sorted(all_A & all_B)  # subjects in both domains
    only_A = sorted(all_A - all_B)
    only_B = sorted(all_B - all_A)

    # Only shared subjects can be in test set (need both domains for eval)
    random.seed(seed)
    random.shuffle(shared)

    n_test = max(1, round(len(shared) * test_ratio))
    test_subjects = set(shared[:n_test])
    train_shared = set(shared[n_test:])

    # Subjects only in one domain go to train
    train_A_subjects = train_shared | set(only_A)
    train_B_subjects = train_shared | set(only_B)

    return train_A_subjects, train_B_subjects, test_subjects


def copy_images(file_list, dest_dir):
    """Copy images to destination directory."""
    os.makedirs(dest_dir, exist_ok=True)
    for src in file_list:
        fname = os.path.basename(src)
        shutil.copy2(src, os.path.join(dest_dir, fname))


def main():
    parser = argparse.ArgumentParser(description="Subject-level train/test split")
    parser.add_argument("--input_dir", default="datasets/us_cyclegan")
    parser.add_argument("--output_dir", default="datasets/us_cyclegan_v7")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Test ratio: {args.test_ratio}, Seed: {args.seed}\n")

    subjects_A, subjects_B = gather_images(args.input_dir)
    print(f"Domain A subjects: {sorted(subjects_A.keys())}")
    print(f"Domain B subjects: {sorted(subjects_B.keys())}")
    print(f"Shared subjects: {sorted(set(subjects_A.keys()) & set(subjects_B.keys()))}\n")

    train_A_subs, train_B_subs, test_subs = split_subjects(
        subjects_A, subjects_B, args.test_ratio, args.seed
    )

    print(f"Test subjects (shared): {sorted(test_subs)}")
    print(f"Train A subjects: {sorted(train_A_subs)}")
    print(f"Train B subjects: {sorted(train_B_subs)}\n")

    # Collect files
    trainA_files = [f for sid in sorted(train_A_subs) for f in subjects_A.get(sid, [])]
    testA_files = [f for sid in sorted(test_subs) for f in subjects_A.get(sid, [])]
    trainB_files = [f for sid in sorted(train_B_subs) for f in subjects_B.get(sid, [])]
    testB_files = [f for sid in sorted(test_subs) for f in subjects_B.get(sid, [])]

    print(f"trainA: {len(trainA_files)} images from {len(train_A_subs)} subjects")
    print(f"testA:  {len(testA_files)} images from {len(test_subs)} subjects")
    print(f"trainB: {len(trainB_files)} images from {len(train_B_subs)} subjects")
    print(f"testB:  {len(testB_files)} images from {len(test_subs)} subjects\n")

    # Copy
    print("Copying files...")
    copy_images(trainA_files, os.path.join(args.output_dir, "trainA"))
    copy_images(testA_files, os.path.join(args.output_dir, "testA"))
    copy_images(trainB_files, os.path.join(args.output_dir, "trainB"))
    copy_images(testB_files, os.path.join(args.output_dir, "testB"))

    # Manifest
    manifest_path = os.path.join(args.output_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write(f"Subject-level train/test split\n")
        f.write(f"Seed: {args.seed}, Test ratio: {args.test_ratio}\n\n")
        f.write(f"Test subjects: {sorted(test_subs)}\n")
        f.write(f"Train A subjects: {sorted(train_A_subs)}\n")
        f.write(f"Train B subjects: {sorted(train_B_subs)}\n\n")
        f.write(f"trainA: {len(trainA_files)} images\n")
        f.write(f"testA:  {len(testA_files)} images\n")
        f.write(f"trainB: {len(trainB_files)} images\n")
        f.write(f"testB:  {len(testB_files)} images\n")

    print(f"Manifest saved: {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    main()
