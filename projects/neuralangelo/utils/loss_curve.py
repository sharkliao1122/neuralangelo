'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import argparse
import csv
import os
import re
from collections import defaultdict


def _to_float(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def append_loss_history(logdir, iteration, epoch, mode, losses, filename="loss_history.csv"):
    """Append loss values to a CSV file for offline plotting."""
    if not logdir:
        return None
    os.makedirs(logdir, exist_ok=True)
    csv_path = os.path.join(logdir, filename)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["iteration", "epoch", "mode", "loss", "value"])
        if write_header:
            writer.writeheader()
        for loss_name, value in sorted(losses.items()):
            writer.writerow({
                "iteration": int(iteration),
                "epoch": int(epoch),
                "mode": mode,
                "loss": loss_name,
                "value": _to_float(value),
            })
    return csv_path


def _safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def read_loss_history(csv_path, mode="train"):
    """Read loss history CSV into a mapping from loss name to points."""
    points = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if mode != "all" and row["mode"] != mode:
                continue
            loss_name = row["loss"] if mode != "all" else f"{row['mode']}_{row['loss']}"
            points[loss_name].append((int(row["iteration"]), float(row["value"])))
    return points


def _plot_points(points, title, output_path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    points = sorted(points)
    iterations = [iteration for iteration, _ in points]
    values = [value for _, value in points]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, values)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_loss_curves(logdir, input_csv=None, output_dir=None, mode="train"):
    """Plot total loss and each individual loss curve."""
    input_csv = input_csv or os.path.join(logdir, "loss_history.csv")
    output_dir = output_dir or os.path.join(logdir, "loss_curves")
    os.makedirs(output_dir, exist_ok=True)

    points_by_loss = read_loss_history(input_csv, mode=mode)
    if not points_by_loss:
        raise ValueError(f"No loss records found in {input_csv} for mode '{mode}'.")

    output_paths = []
    total_key = "total" if mode != "all" else "train_total"
    if total_key in points_by_loss:
        output_path = os.path.join(output_dir, "loss_total.png")
        _plot_points(points_by_loss[total_key], f"{mode} total loss", output_path)
        output_paths.append(output_path)

    for loss_name, points in sorted(points_by_loss.items()):
        if loss_name == total_key:
            continue
        output_path = os.path.join(output_dir, f"loss_{_safe_filename(loss_name)}.png")
        _plot_points(points, f"{mode} {loss_name} loss", output_path)
        output_paths.append(output_path)

    return output_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss curves from loss_history.csv.")
    parser.add_argument("--logdir", required=True, help="Training log directory.")
    parser.add_argument("--input_csv", default=None, help="Path to loss_history.csv. Defaults to <logdir>/loss_history.csv.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to <logdir>/loss_curves.")
    parser.add_argument("--mode", default="train", choices=["train", "val", "all"], help="Loss mode to plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_paths = plot_loss_curves(
        logdir=args.logdir,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        mode=args.mode,
    )
    for output_path in output_paths:
        print(f"Wrote loss curve: {output_path}")


if __name__ == "__main__":
    main()
