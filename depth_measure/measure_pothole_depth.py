"""Measure pothole-like depth from a mesh or point cloud in model units.

The script fits a reference road plane with RANSAC, then measures signed
point-to-plane distances for the candidate pothole region. It intentionally
does not apply any real-world scale conversion.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def vector_norm(vector):
    return float(np.sqrt(np.sum(vector * vector)))


def row_norms(vectors):
    return np.sqrt(np.sum(vectors * vectors, axis=1))


def smallest_eigenvector_symmetric_3x3(matrix, sweeps=24):
    a = [[float(matrix[row, col]) for col in range(3)] for row in range(3)]
    v = [[1.0 if row == col else 0.0 for col in range(3)] for row in range(3)]

    for _ in range(sweeps):
        p, q = 0, 1
        largest = abs(a[0][1])
        for i, j in ((0, 2), (1, 2)):
            value = abs(a[i][j])
            if value > largest:
                p, q = i, j
                largest = value
        if largest < 1e-18:
            break

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]
        theta = 0.5 * (aqq - app) / apq
        t = 1.0 / (abs(theta) + np.sqrt(theta * theta + 1.0))
        if theta < 0.0:
            t = -t
        c = 1.0 / np.sqrt(t * t + 1.0)
        s = t * c
        tau = s / (1.0 + c)

        a[p][q] = 0.0
        a[q][p] = 0.0
        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq

        for r in range(3):
            if r == p or r == q:
                continue
            arp = a[r][p]
            arq = a[r][q]
            a[r][p] = arp - s * (arq + tau * arp)
            a[p][r] = a[r][p]
            a[r][q] = arq + s * (arp - tau * arq)
            a[q][r] = a[r][q]

        for r in range(3):
            vrp = v[r][p]
            vrq = v[r][q]
            v[r][p] = vrp - s * (vrq + tau * vrp)
            v[r][q] = vrq + s * (vrp - tau * vrq)

    eigenvalues = [a[0][0], a[1][1], a[2][2]]
    smallest = min(range(3), key=lambda index: eigenvalues[index])
    return np.array([v[0][smallest], v[1][smallest], v[2][smallest]], dtype=np.float64)


def load_vertices(path):
    geometry = trimesh.load(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        chunks = [
            np.asarray(item.vertices, dtype=np.float64)
            for item in geometry.geometry.values()
            if hasattr(item, "vertices") and len(item.vertices)
        ]
        if not chunks:
            raise ValueError(f"No vertices found in scene: {path}")
        vertices = np.vstack(chunks)
    elif hasattr(geometry, "vertices"):
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported geometry type: {type(geometry).__name__}")

    finite = np.isfinite(vertices).all(axis=1)
    return vertices[finite]


def bounds_summary(points):
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return {
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "size": np.ptp(points, axis=0).tolist(),
    }


def plane_values(points, normal, offset):
    return (
        points[:, 0] * normal[0]
        + points[:, 1] * normal[1]
        + points[:, 2] * normal[2]
        + offset
    )


def bounds_mask(points, bounds):
    if bounds is None:
        return np.ones(len(points), dtype=bool)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )


def apply_bounds(points, bounds):
    if bounds is None:
        return points
    return points[bounds_mask(points, bounds)]


def depth_to_rgba(depths, max_depth):
    max_depth = max(float(max_depth), 1e-12)
    t = np.clip(depths / max_depth, 0.0, 1.0)
    stops = np.array([0.0, 0.20, 0.45, 0.72, 1.0], dtype=np.float64)
    colors = np.array(
        [
            [48, 84, 150],
            [44, 178, 214],
            [67, 190, 112],
            [245, 206, 66],
            [218, 62, 55],
        ],
        dtype=np.float64,
    )
    rgba = np.zeros((len(depths), 4), dtype=np.float64)
    for index in range(len(stops) - 1):
        segment = (t >= stops[index]) & (t <= stops[index + 1])
        local_t = (t[segment] - stops[index]) / (stops[index + 1] - stops[index])
        rgba[segment, :3] = colors[index] * (1.0 - local_t[:, None]) + colors[index + 1] * local_t[:, None]
    rgba[:, 3] = 255
    return np.rint(rgba).astype(np.uint8)


def export_colored_ply(input_path, output_path, normal, offset, color_mask, color_max_depth):
    geometry = trimesh.load(str(input_path), process=False)
    if isinstance(geometry, trimesh.Scene):
        raise ValueError("--colored-ply currently supports a single mesh or point cloud, not a Scene")
    if not hasattr(geometry, "vertices"):
        raise ValueError(f"Unsupported geometry type for colored export: {type(geometry).__name__}")

    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    if len(vertices) != len(color_mask):
        raise ValueError("Internal vertex count mismatch while exporting colored PLY")

    signed = plane_values(vertices, normal, offset)
    depths = np.maximum(0.0, -signed)
    neutral = np.array([155, 155, 155, 255], dtype=np.uint8)
    rgba = np.repeat(neutral[None, :], len(vertices), axis=0)
    rgba[color_mask] = depth_to_rgba(depths[color_mask], color_max_depth)

    geometry.visual.vertex_colors = rgba
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry.export(str(output_path))
    return {
        "path": str(output_path),
        "colored_vertices": int(color_mask.sum()),
        "color_max_depth": float(color_max_depth),
    }


def export_preview_json(output_path, points, normal, offset, color_mask, max_points, color_max_depth, seed):
    signed = plane_values(points, normal, offset)
    depths = np.maximum(0.0, -signed)
    eligible = np.flatnonzero(color_mask)
    rng = np.random.default_rng(seed)
    if len(eligible) > max_points:
        indices = rng.choice(eligible, size=max_points, replace=False)
    else:
        indices = eligible
    selected = points[indices]
    selected_depths = depths[indices]
    selected_rgba = depth_to_rgba(selected_depths, color_max_depth)[:, :3]
    order = np.argsort(selected_depths)
    selected = selected[order]
    selected_depths = selected_depths[order]
    selected_rgba = selected_rgba[order]

    payload = {
        "units": "model_units",
        "point_count": int(len(selected)),
        "color_max_depth": float(color_max_depth),
        "bounds": bounds_summary(selected),
        "points": [
            [
                round(float(point[0]), 4),
                round(float(point[1]), 4),
                round(float(point[2]), 4),
                round(float(depth), 6),
                int(color[0]),
                int(color[1]),
                int(color[2]),
            ]
            for point, depth, color in zip(selected, selected_depths, selected_rgba)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return {"path": str(output_path), "preview_points": int(len(selected))}


def fit_plane_covariance(points):
    if len(points) < 3:
        raise ValueError("At least 3 points are required to fit a plane")
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = np.array(
        [
            [
                np.sum(centered[:, 0] * centered[:, 0]),
                np.sum(centered[:, 0] * centered[:, 1]),
                np.sum(centered[:, 0] * centered[:, 2]),
            ],
            [
                np.sum(centered[:, 1] * centered[:, 0]),
                np.sum(centered[:, 1] * centered[:, 1]),
                np.sum(centered[:, 1] * centered[:, 2]),
            ],
            [
                np.sum(centered[:, 2] * centered[:, 0]),
                np.sum(centered[:, 2] * centered[:, 1]),
                np.sum(centered[:, 2] * centered[:, 2]),
            ],
        ],
        dtype=np.float64,
    )
    normal = smallest_eigenvector_symmetric_3x3(covariance)
    normal /= vector_norm(normal)
    offset = -float(normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2])
    return normal, offset


def subsample(points, max_points, rng):
    if max_points is None or len(points) <= max_points:
        return points
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def default_ransac_threshold(points):
    diagonal = vector_norm(np.ptp(points, axis=0))
    if diagonal <= 0:
        raise ValueError("Cannot estimate a threshold from zero-size bounds")
    return diagonal * 0.0025


def random_triples(rng, population_size, count):
    triples = []
    remaining = count
    while remaining:
        candidates = rng.integers(0, population_size, size=(remaining * 2, 3))
        distinct = (
            (candidates[:, 0] != candidates[:, 1])
            & (candidates[:, 0] != candidates[:, 2])
            & (candidates[:, 1] != candidates[:, 2])
        )
        chosen = candidates[distinct][:remaining]
        if len(chosen):
            triples.append(chosen)
            remaining -= len(chosen)
    return np.vstack(triples)


def ransac_plane(points, threshold, iterations, sample_size, refine_size, batch_size, seed):
    if len(points) < 3:
        raise ValueError("At least 3 plane points are required")

    rng = np.random.default_rng(seed)
    sample = subsample(points, sample_size, rng)
    best_count = -1
    best_error = np.inf
    best_normal = None
    best_offset = None
    best_mask = None

    completed = 0
    while completed < iterations:
        current_batch = min(batch_size, iterations - completed)
        picks = random_triples(rng, len(sample), current_batch)
        p0 = sample[picks[:, 0]]
        p1 = sample[picks[:, 1]]
        p2 = sample[picks[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)
        norms = row_norms(normals)
        valid = norms >= 1e-12
        if not np.any(valid):
            completed += current_batch
            continue

        normals = normals[valid] / norms[valid, None]
        valid_p0 = p0[valid]
        offsets = -(
            normals[:, 0] * valid_p0[:, 0]
            + normals[:, 1] * valid_p0[:, 1]
            + normals[:, 2] * valid_p0[:, 2]
        )
        distances = np.abs(
            sample[:, 0, None] * normals[None, :, 0]
            + sample[:, 1, None] * normals[None, :, 1]
            + sample[:, 2, None] * normals[None, :, 2]
            + offsets[None, :]
        )
        masks = distances <= threshold
        counts = masks.sum(axis=0)
        valid_counts = counts >= 3
        if np.any(valid_counts):
            safe_counts = np.maximum(counts, 1)
            errors = (distances * masks).sum(axis=0) / safe_counts
            scores = np.where(valid_counts, counts, -1)
            batch_best = int(np.argmax(scores))
            count = int(counts[batch_best])
            error = float(errors[batch_best])
            if count > best_count or (count == best_count and error < best_error):
                best_count = count
                best_error = error
                best_normal = normals[batch_best]
                best_offset = float(offsets[batch_best])
                best_mask = masks[:, batch_best]

        completed += current_batch

    if best_normal is None:
        raise RuntimeError(
            "RANSAC could not find a plane. Try increasing --ransac-threshold "
            "or selecting a smaller --plane-roi."
        )

    normal, offset = fit_plane_covariance(sample[best_mask])
    all_distances = np.abs(plane_values(points, normal, offset))
    inlier_mask = all_distances <= threshold
    if int(inlier_mask.sum()) >= 3:
        refine_points = subsample(points[inlier_mask], refine_size, rng)
        normal, offset = fit_plane_covariance(refine_points)
        all_distances = np.abs(plane_values(points, normal, offset))
        inlier_mask = all_distances <= threshold

    return normal, offset, inlier_mask, all_distances


def orient_for_depth(normal, offset, distances, pit_side, bottom_percent):
    if pit_side == "negative":
        return normal, offset, distances, "negative"
    if pit_side == "positive":
        return -normal, -offset, -distances, "positive"

    low = float(np.percentile(distances, bottom_percent))
    high = float(np.percentile(distances, 100.0 - bottom_percent))
    if abs(high) > abs(low):
        return -normal, -offset, -distances, "auto_positive_tail"
    return normal, offset, distances, "auto_negative_tail"


def deepest_percent_indices(distances, percent):
    count = max(1, int(np.ceil(len(distances) * percent / 100.0)))
    return np.argpartition(distances, count - 1)[:count]


def measure_depth(points, normal, offset, pit_side, bottom_percent):
    signed = plane_values(points, normal, offset)
    normal, offset, signed, chosen_side = orient_for_depth(
        normal, offset, signed, pit_side, bottom_percent
    )

    deepest_index = int(np.argmin(signed))
    deepest_distance = float(signed[deepest_index])
    bottom_indices = deepest_percent_indices(signed, bottom_percent)
    bottom_distances = signed[bottom_indices]
    bottom_points = points[bottom_indices]

    return {
        "normal": normal,
        "offset": offset,
        "signed_distances": signed,
        "chosen_side": chosen_side,
        "max_depth": max(0.0, -deepest_distance),
        "deepest_point_xyz": points[deepest_index].tolist(),
        "deepest_point_signed_distance": deepest_distance,
        "bottom_percent": bottom_percent,
        "bottom_count": int(len(bottom_indices)),
        "bottom_mean_depth": max(0.0, -float(bottom_distances.mean())),
        "bottom_percentile_depth": max(0.0, -float(bottom_distances.max())),
        "bottom_centroid_xyz": bottom_points.mean(axis=0).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a reference plane and measure pothole depth in the original "
            "mesh/point-cloud coordinate scale."
        )
    )
    parser.add_argument("input", type=Path, help="Input mesh or point-cloud file, such as .ply")
    parser.add_argument(
        "--crop",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Common 3D crop applied before plane fitting and depth measurement.",
    )
    parser.add_argument(
        "--plane-roi",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="3D bounds for points used to fit the road plane.",
    )
    parser.add_argument(
        "--depth-roi",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="3D bounds for points measured as the pothole region.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=None,
        help="Max model-unit distance from a point to the plane for RANSAC inliers.",
    )
    parser.add_argument("--ransac-iterations", type=int, default=512)
    parser.add_argument("--ransac-sample-size", type=int, default=30000)
    parser.add_argument("--ransac-batch-size", type=int, default=64)
    parser.add_argument("--refine-size", type=int, default=250000)
    parser.add_argument(
        "--bottom-percent",
        type=float,
        default=1.0,
        help="Deepest percent of measured points used for the robust mean depth.",
    )
    parser.add_argument(
        "--pit-side",
        choices=("auto", "negative", "positive"),
        default="auto",
        help=(
            "Which signed side of the fitted plane is the pothole. Use auto for "
            "the deeper tail, or force negative/positive if you know the orientation."
        ),
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--colored-ply",
        type=Path,
        default=None,
        help="Optional output PLY with vertex colors mapped to model-unit depth.",
    )
    parser.add_argument(
        "--color-max-depth",
        type=float,
        default=None,
        help=(
            "Depth value mapped to the hottest color. Defaults to the deepest "
            "--bottom-percent threshold, which preserves contrast better than a single outlier."
        ),
    )
    parser.add_argument(
        "--preview-json",
        type=Path,
        default=None,
        help="Optional sampled point/depth/color JSON for lightweight preview visualizations.",
    )
    parser.add_argument("--preview-points", type=int, default=20000)
    parser.add_argument("--print-bounds-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not (0.0 < args.bottom_percent <= 50.0):
        raise ValueError("--bottom-percent must be > 0 and <= 50")

    vertices = load_vertices(args.input)
    print(f"Input: {args.input}")
    print(f"Loaded vertices: {len(vertices):,}")
    print(f"Full bounds: {json.dumps(bounds_summary(vertices), indent=2)}")
    if args.print_bounds_only:
        return

    working = apply_bounds(vertices, args.crop)
    if len(working) < 3:
        raise ValueError("--crop left fewer than 3 points")

    plane_points = apply_bounds(working, args.plane_roi)
    depth_points = apply_bounds(working, args.depth_roi)
    if len(plane_points) < 3:
        raise ValueError("--plane-roi left fewer than 3 points")
    if len(depth_points) < 1:
        raise ValueError("--depth-roi left no points")

    threshold = args.ransac_threshold
    if threshold is None:
        threshold = default_ransac_threshold(plane_points)

    normal, offset, inlier_mask, plane_distances = ransac_plane(
        plane_points,
        threshold=threshold,
        iterations=args.ransac_iterations,
        sample_size=args.ransac_sample_size,
        refine_size=args.refine_size,
        batch_size=args.ransac_batch_size,
        seed=args.seed,
    )
    depth = measure_depth(
        depth_points,
        normal=normal,
        offset=offset,
        pit_side=args.pit_side,
        bottom_percent=args.bottom_percent,
    )
    color_mask = bounds_mask(vertices, args.crop) & bounds_mask(vertices, args.depth_roi)
    color_max_depth = args.color_max_depth
    if color_max_depth is None:
        color_max_depth = depth["bottom_percentile_depth"] or depth["max_depth"] or 1.0

    plane_rmse = float(np.sqrt(np.mean(plane_distances[inlier_mask] ** 2)))
    result = {
        "input": str(args.input),
        "units": "model_units",
        "vertices_loaded": int(len(vertices)),
        "working_vertices": int(len(working)),
        "plane_vertices": int(len(plane_points)),
        "depth_vertices": int(len(depth_points)),
        "working_bounds": bounds_summary(working),
        "plane_bounds": bounds_summary(plane_points),
        "depth_bounds": bounds_summary(depth_points),
        "ransac_threshold": float(threshold),
        "ransac_iterations": int(args.ransac_iterations),
        "plane": {
            "normal": depth["normal"].tolist(),
            "offset": float(depth["offset"]),
            "equation": "normal_x*x + normal_y*y + normal_z*z + offset = 0",
            "inliers": int(inlier_mask.sum()),
            "inlier_ratio": float(inlier_mask.mean()),
            "inlier_rmse": plane_rmse,
        },
        "depth": {
            "pit_side": depth["chosen_side"],
            "max_depth": depth["max_depth"],
            "deepest_point_xyz": depth["deepest_point_xyz"],
            "deepest_point_signed_distance": depth["deepest_point_signed_distance"],
            "bottom_percent": depth["bottom_percent"],
            "bottom_count": depth["bottom_count"],
            "bottom_mean_depth": depth["bottom_mean_depth"],
            "bottom_percentile_depth": depth["bottom_percentile_depth"],
            "bottom_centroid_xyz": depth["bottom_centroid_xyz"],
        },
        "depth_color": {
            "colored_vertices": int(color_mask.sum()),
            "color_max_depth": float(color_max_depth),
        },
    }

    if args.colored_ply:
        result["colored_ply"] = export_colored_ply(
            args.input,
            args.colored_ply,
            depth["normal"],
            depth["offset"],
            color_mask,
            color_max_depth,
        )

    if args.preview_json:
        if args.preview_points <= 0:
            raise ValueError("--preview-points must be greater than 0")
        result["preview_json"] = export_preview_json(
            args.preview_json,
            vertices,
            depth["normal"],
            depth["offset"],
            color_mask,
            args.preview_points,
            color_max_depth,
            args.seed,
        )

    print()
    print("Reference plane")
    print(f"  normal: {result['plane']['normal']}")
    print(f"  offset: {result['plane']['offset']:.9g}")
    print(
        "  inliers: "
        f"{result['plane']['inliers']:,}/{result['plane_vertices']:,} "
        f"({result['plane']['inlier_ratio']:.2%})"
    )
    print(f"  inlier RMSE: {result['plane']['inlier_rmse']:.9g}")
    print(f"  RANSAC threshold: {result['ransac_threshold']:.9g}")

    print()
    print("Depth in model units")
    print(f"  max_depth: {result['depth']['max_depth']:.9g}")
    print(
        f"  deepest_{result['depth']['bottom_percent']:.3g}%_mean_depth: "
        f"{result['depth']['bottom_mean_depth']:.9g}"
    )
    print(
        f"  deepest_{result['depth']['bottom_percent']:.3g}%_percentile_depth: "
        f"{result['depth']['bottom_percentile_depth']:.9g}"
    )
    print(f"  deepest_point_xyz: {result['depth']['deepest_point_xyz']}")
    print(f"  bottom_centroid_xyz: {result['depth']['bottom_centroid_xyz']}")
    print(f"  chosen_pit_side: {result['depth']['pit_side']}")
    print(f"  color_max_depth: {result['depth_color']['color_max_depth']:.9g}")

    if args.colored_ply:
        print()
        print(f"Saved colored depth PLY: {args.colored_ply}")

    if args.preview_json:
        print(f"Saved preview JSON: {args.preview_json}")

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print()
        print(f"Saved summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
