'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import argparse
import json
import math
import os

import numpy as np
import trimesh
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a Neuralangelo mesh using multi-view alpha support."
    )
    parser.add_argument("--mesh", required=True, help="Input mesh path.")
    parser.add_argument("--transforms", required=True, help="Path to transforms.json.")
    parser.add_argument("--output", required=True, help="Output mesh path.")
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Optional JSON file for a detailed postprocess summary.",
    )
    parser.add_argument(
        "--frame_sample_count",
        type=int,
        default=24,
        help="Use this many evenly spaced RGBA frames. Use 0 or a negative value for all frames.",
    )
    parser.add_argument(
        "--min_valid_views",
        type=int,
        default=3,
        help="Minimum projected views required before a vertex support score is trusted.",
    )
    parser.add_argument(
        "--min_face_support",
        type=float,
        default=0.28,
        help="Base average vertex support required to keep a face.",
    )
    parser.add_argument(
        "--low_z_threshold",
        type=float,
        default=-0.40,
        help="Normalized z threshold below which stricter filtering is applied.",
    )
    parser.add_argument(
        "--low_z_support",
        type=float,
        default=0.40,
        help="Required average face support below low_z_threshold.",
    )
    parser.add_argument(
        "--min_component_faces",
        type=int,
        default=1200,
        help="Directly keep components above this face count if they also have strong support.",
    )
    parser.add_argument(
        "--min_component_support",
        type=float,
        default=0.45,
        help="Directly keep large components above this support.",
    )
    parser.add_argument(
        "--near_component_faces",
        type=int,
        default=250,
        help="Minimum face count for smaller components that are close to the anchor component.",
    )
    parser.add_argument(
        "--near_component_support",
        type=float,
        default=0.28,
        help="Minimum support for smaller components that are close to the anchor component.",
    )
    parser.add_argument(
        "--near_distance",
        type=float,
        default=0.22,
        help="Maximum normalized bbox gap for a smaller component to be retained near the anchor.",
    )
    parser.add_argument(
        "--write_face_filtered",
        action="store_true",
        help="Also export the face-filtered mesh before component filtering.",
    )
    return parser.parse_args()


def load_meta(transforms_path):
    with open(transforms_path, "r", encoding="utf-8") as file:
        meta = json.load(file)
    meta_root = os.path.dirname(os.path.abspath(transforms_path))
    return meta, meta_root


def choose_frames(frames, sample_count):
    if sample_count <= 0 or sample_count >= len(frames):
        return frames
    indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
    return [frames[i] for i in indices]


def gl_to_cv(c2w_gl):
    return c2w_gl * np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)


def build_view_data(meta, meta_root, sample_count):
    frames = choose_frames(meta["frames"], sample_count)
    center = np.array(meta["sphere_center"], dtype=np.float32)
    scale = float(meta["sphere_radius"])
    intr = dict(
        fx=float(meta["fl_x"]),
        fy=float(meta["fl_y"]),
        cx=float(meta["cx"]),
        cy=float(meta["cy"]),
        width=int(meta["w"]),
        height=int(meta["h"]),
    )
    views = []
    for frame in frames:
        image_path = os.path.join(meta_root, frame["file_path"])
        image = Image.open(image_path).convert("RGBA")
        alpha = np.array(image, dtype=np.uint8)[..., 3] > 0
        c2w_gl = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w = gl_to_cv(c2w_gl)
        c2w[:3, 3] -= center
        c2w[:3, 3] /= scale
        w2c = np.linalg.inv(c2w)[:3, :]
        views.append(dict(alpha=alpha, w2c=w2c))
    return views, intr, center, scale


def compute_vertex_support(vertices_world, views, intr, center, scale, min_valid_views):
    if len(vertices_world) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    vertices_norm = (vertices_world - center) / scale
    homog = np.concatenate(
        [vertices_norm.astype(np.float32), np.ones((len(vertices_norm), 1), dtype=np.float32)],
        axis=1,
    )
    valid_counts = np.zeros(len(vertices_norm), dtype=np.int32)
    hit_counts = np.zeros(len(vertices_norm), dtype=np.int32)
    for view in views:
        cam = homog @ view["w2c"].T
        z = cam[:, 2]
        front = z > 1e-4
        x = intr["fx"] * (cam[:, 0] / z) + intr["cx"]
        y = intr["fy"] * (cam[:, 1] / z) + intr["cy"]
        inside = front & (x >= 0) & (x < intr["width"]) & (y >= 0) & (y < intr["height"])
        valid_counts += inside.astype(np.int32)
        if np.any(inside):
            indices = np.where(inside)[0]
            xi = x[inside].astype(np.int32)
            yi = y[inside].astype(np.int32)
            hit_counts[indices] += view["alpha"][yi, xi].astype(np.int32)
    support = np.divide(
        hit_counts,
        np.maximum(valid_counts, 1),
        dtype=np.float32,
    )
    support[valid_counts < min_valid_views] = 0.0
    return support, valid_counts, hit_counts


def face_filter(mesh, vertex_support, vertices_norm, min_face_support, low_z_threshold, low_z_support):
    face_support = vertex_support[mesh.faces].mean(axis=1)
    face_z = vertices_norm[mesh.faces][:, :, 2].mean(axis=1)
    keep_faces = face_support >= min_face_support
    keep_faces &= ~((face_z <= low_z_threshold) & (face_support < low_z_support))
    filtered = mesh.copy()
    filtered.update_faces(keep_faces)
    filtered.remove_unreferenced_vertices()
    filtered.update_faces(filtered.nondegenerate_faces())
    filtered.remove_unreferenced_vertices()
    return filtered, face_support, face_z, keep_faces


def bbox_gap(bounds_a, bounds_b):
    gap = np.maximum(0.0, np.maximum(bounds_a[0] - bounds_b[1], bounds_b[0] - bounds_a[1]))
    return float(np.linalg.norm(gap))


def choose_anchor(components, component_stats):
    best_idx = 0
    best_score = -math.inf
    for idx, stats in enumerate(component_stats):
        score = stats["mean_support"] * math.log1p(stats["faces"])
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def component_filter(
    components,
    views,
    intr,
    center,
    scale,
    min_valid_views,
    min_component_faces,
    min_component_support,
    near_component_faces,
    near_component_support,
    near_distance,
):
    component_stats = []
    for idx, component in enumerate(components):
        support, valid_counts, hit_counts = compute_vertex_support(
            component.vertices, views, intr, center, scale, min_valid_views
        )
        stats = dict(
            index=idx,
            verts=int(len(component.vertices)),
            faces=int(len(component.faces)),
            mean_support=float(support.mean()) if len(support) else 0.0,
            median_support=float(np.median(support)) if len(support) else 0.0,
            valid_ratio=float((valid_counts >= min_valid_views).mean()) if len(valid_counts) else 0.0,
            bounds=((component.bounds - center) / scale).tolist(),
        )
        component_stats.append(stats)

    anchor_idx = choose_anchor(components, component_stats)
    anchor_bounds = np.array(component_stats[anchor_idx]["bounds"], dtype=np.float32)
    kept_indices = []
    for stats in component_stats:
        idx = stats["index"]
        bounds = np.array(stats["bounds"], dtype=np.float32)
        dist = bbox_gap(bounds, anchor_bounds)
        stats["bbox_gap_to_anchor"] = dist
        keep = False
        if idx == anchor_idx:
            keep = True
        elif stats["faces"] >= min_component_faces and stats["mean_support"] >= min_component_support:
            keep = True
        elif (
            stats["faces"] >= near_component_faces
            and stats["mean_support"] >= near_component_support
            and dist <= near_distance
        ):
            keep = True
        stats["kept"] = keep
        if keep:
            kept_indices.append(idx)
    kept_meshes = [components[idx] for idx in kept_indices]
    if not kept_meshes:
        kept_meshes = [components[anchor_idx]]
        component_stats[anchor_idx]["kept"] = True
        kept_indices = [anchor_idx]
    merged = trimesh.util.concatenate(kept_meshes) if len(kept_meshes) > 1 else kept_meshes[0].copy()
    merged.update_faces(merged.nondegenerate_faces())
    merged.remove_unreferenced_vertices()
    return merged, component_stats, anchor_idx, kept_indices


def default_summary_path(output_path):
    stem, _ = os.path.splitext(output_path)
    return f"{stem}.summary.json"


def face_filtered_output_path(output_path):
    stem, ext = os.path.splitext(output_path)
    return f"{stem}.face_filtered{ext}"


def main():
    args = parse_args()
    mesh = trimesh.load(args.mesh, process=False)
    meta, meta_root = load_meta(args.transforms)
    views, intr, center, scale = build_view_data(meta, meta_root, args.frame_sample_count)

    vertex_support, valid_counts, hit_counts = compute_vertex_support(
        mesh.vertices, views, intr, center, scale, args.min_valid_views
    )
    vertices_norm = (mesh.vertices - center) / scale
    filtered, face_support, face_z, keep_faces = face_filter(
        mesh,
        vertex_support,
        vertices_norm,
        args.min_face_support,
        args.low_z_threshold,
        args.low_z_support,
    )

    if len(filtered.faces) == 0:
        raise RuntimeError("All faces were filtered out. Relax the face support thresholds.")

    if args.write_face_filtered:
        face_filtered_path = face_filtered_output_path(args.output)
        os.makedirs(os.path.dirname(face_filtered_path), exist_ok=True)
        filtered.export(face_filtered_path)
    else:
        face_filtered_path = None

    components = filtered.split(only_watertight=False)
    merged, component_stats, anchor_idx, kept_indices = component_filter(
        components,
        views,
        intr,
        center,
        scale,
        args.min_valid_views,
        args.min_component_faces,
        args.min_component_support,
        args.near_component_faces,
        args.near_component_support,
        args.near_distance,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.export(args.output)

    summary = dict(
        input_mesh=os.path.abspath(args.mesh),
        transforms=os.path.abspath(args.transforms),
        output_mesh=os.path.abspath(args.output),
        face_filtered_mesh=os.path.abspath(face_filtered_path) if face_filtered_path else None,
        frame_sample_count=len(views),
        min_valid_views=args.min_valid_views,
        min_face_support=args.min_face_support,
        low_z_threshold=args.low_z_threshold,
        low_z_support=args.low_z_support,
        min_component_faces=args.min_component_faces,
        min_component_support=args.min_component_support,
        near_component_faces=args.near_component_faces,
        near_component_support=args.near_component_support,
        near_distance=args.near_distance,
        original=dict(
            verts=int(len(mesh.vertices)),
            faces=int(len(mesh.faces)),
            components=int(len(mesh.split(only_watertight=False))),
            bounds=mesh.bounds.tolist(),
        ),
        vertex_support=dict(
            mean=float(vertex_support.mean()) if len(vertex_support) else 0.0,
            median=float(np.median(vertex_support)) if len(vertex_support) else 0.0,
            q05=float(np.quantile(vertex_support, 0.05)) if len(vertex_support) else 0.0,
            q95=float(np.quantile(vertex_support, 0.95)) if len(vertex_support) else 0.0,
            valid_ratio=float((valid_counts >= args.min_valid_views).mean()) if len(valid_counts) else 0.0,
        ),
        face_filter=dict(
            kept_faces=int(keep_faces.sum()),
            dropped_faces=int((~keep_faces).sum()),
            kept_ratio=float(keep_faces.mean()),
            mean_face_support=float(face_support.mean()) if len(face_support) else 0.0,
            min_face_z=float(face_z.min()) if len(face_z) else 0.0,
            max_face_z=float(face_z.max()) if len(face_z) else 0.0,
        ),
        component_filter=dict(
            anchor_component=anchor_idx,
            kept_components=kept_indices,
            total_components=int(len(components)),
            per_component=component_stats,
        ),
        final=dict(
            verts=int(len(merged.vertices)),
            faces=int(len(merged.faces)),
            components=int(len(merged.split(only_watertight=False))),
            bounds=merged.bounds.tolist(),
        ),
    )

    summary_path = args.summary_json or default_summary_path(args.output)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved filtered mesh to {args.output}")
    if face_filtered_path:
        print(f"Saved face-filtered mesh to {face_filtered_path}")
    print(f"Saved summary to {summary_path}")
    print(
        "Final mesh:",
        f"verts={len(merged.vertices)}",
        f"faces={len(merged.faces)}",
        f"components={len(merged.split(only_watertight=False))}",
    )


if __name__ == "__main__":
    main()
