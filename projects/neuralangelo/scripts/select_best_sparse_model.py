#!/usr/bin/env python3
"""Select the best COLMAP sparse reconstruction under a mapper output folder."""

import argparse
import math
import os
import sys


COLMAP_PYTHON_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "colmap", "scripts", "python")
)
sys.path.insert(0, COLMAP_PYTHON_DIR)

from read_write_model import read_model  # NOQA: E402


def _model_bin_size(model_dir):
    total = 0
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    return total


def _score_model(model_dir):
    cameras, images, points3d = read_model(model_dir, ext=".bin")
    registered_images = len(images)
    point_count = len(points3d)
    observations = sum(len(point.image_ids) for point in points3d.values())
    mean_track_length = observations / point_count if point_count else 0.0

    errors = [float(point.error) for point in points3d.values()]
    mean_error = sum(errors) / len(errors) if errors else math.inf
    if not math.isfinite(mean_error):
        mean_error = math.inf

    return {
        "path": model_dir,
        "registered_images": registered_images,
        "points3d": point_count,
        "observations": observations,
        "mean_track_length": mean_track_length,
        "mean_error": mean_error,
        "bin_size": _model_bin_size(model_dir),
        "score": (
            registered_images,
            point_count,
            observations,
            mean_track_length,
            -mean_error,
            _model_bin_size(model_dir),
        ),
    }


def _is_binary_model_dir(path):
    return all(os.path.isfile(os.path.join(path, name)) for name in ("cameras.bin", "images.bin", "points3D.bin"))


def select_best_sparse_model(sparse_dir):
    candidates = []
    for name in sorted(os.listdir(sparse_dir)):
        path = os.path.join(sparse_dir, name)
        if os.path.isdir(path) and _is_binary_model_dir(path):
            candidates.append(_score_model(path))

    if not candidates:
        raise RuntimeError(f"No valid COLMAP binary sparse models found under {sparse_dir}")

    return max(candidates, key=lambda item: item["score"]), candidates


def _print_report(best, candidates):
    print("Sparse model candidates:", file=sys.stderr)
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        marker = "*" if candidate["path"] == best["path"] else " "
        mean_error = "inf" if math.isinf(candidate["mean_error"]) else f"{candidate['mean_error']:.6f}"
        print(
            f"{marker} {candidate['path']} | "
            f"images={candidate['registered_images']} "
            f"points3D={candidate['points3d']} "
            f"observations={candidate['observations']} "
            f"track={candidate['mean_track_length']:.3f} "
            f"error={mean_error}",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Choose the best COLMAP mapper output. Ranking is by registered image count first, "
            "then 3D point count, observations, mean track length, lower reprojection error, and file size."
        )
    )
    parser.add_argument("--sparse_dir", required=True, help="Folder containing COLMAP mapper outputs such as 0, 1, 2.")
    parser.add_argument("--verbose", action="store_true", help="Print candidate metrics to stderr.")
    args = parser.parse_args()

    best, candidates = select_best_sparse_model(args.sparse_dir)
    if args.verbose:
        _print_report(best, candidates)
    print(best["path"])


if __name__ == "__main__":
    main()
