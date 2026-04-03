"""
Parse a CycleGAN training log and append timing info to results/training_times.json.

Usage:
    python scripts/log_training_time.py \
        --log_file /path/to/training_output.log \
        --run_name ablation_perceptual
"""

import argparse
import json
import re
from pathlib import Path


def parse_training_log(log_path: str) -> dict:
    """Extract epoch times from training log."""
    epoch_times = []
    total_epochs = 0
    max_epoch = 0

    with open(log_path) as f:
        for line in f:
            # Match: End of epoch 26 / 200 	 Time Taken: 177 sec
            m = re.search(r"End of epoch (\d+) / (\d+)\s+Time Taken:\s+(\d+) sec", line)
            if m:
                epoch = int(m.group(1))
                total_epochs = int(m.group(2))
                time_sec = int(m.group(3))
                epoch_times.append(time_sec)
                max_epoch = max(max_epoch, epoch)

    if not epoch_times:
        return None

    total_sec = sum(epoch_times)
    avg_sec = total_sec / len(epoch_times)

    return {
        "epochs": total_epochs,
        "completed_epochs": max_epoch,
        "avg_sec_per_epoch": round(avg_sec, 1),
        "total_sec": total_sec,
        "total_min": round(total_sec / 60, 1),
        "total_hours": round(total_sec / 3600, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True, help="Path to training output log")
    parser.add_argument("--run_name", required=True, help="Run name")
    parser.add_argument("--output", default="results/training_times.json",
                        help="JSON file to append to")
    args = parser.parse_args()

    timing = parse_training_log(args.log_file)
    if timing is None:
        print(f"No epoch timing found in {args.log_file}")
        return

    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}

    data[args.run_name] = timing

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Logged {args.run_name}: {timing['total_min']} min ({timing['total_hours']} hrs), "
          f"{timing['avg_sec_per_epoch']} sec/epoch avg")


if __name__ == "__main__":
    main()
