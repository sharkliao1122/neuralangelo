"""
Mesh postprocessing helpers for Neuralangelo exports.
"""

import json
import math
import os

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree


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


def bbox_gap_xy(bounds_a, bounds_b):
    bounds_a_xy = bounds_a[:, :2]
    bounds_b_xy = bounds_b[:, :2]
    gap = np.maximum(0.0, np.maximum(bounds_a_xy[0] - bounds_b_xy[1], bounds_b_xy[0] - bounds_a_xy[1]))
    return float(np.linalg.norm(gap))


def choose_anchor(component_stats):
    best_idx = 0
    best_score = -math.inf
    for stats in component_stats:
        score = stats["mean_support"] * math.log1p(stats["faces"])
        if score > best_score:
            best_score = score
            best_idx = stats["index"]
    return best_idx


def compute_component_stats(components, views, intr, center, scale, min_valid_views):
    component_stats = []
    for idx, component in enumerate(components):
        support, valid_counts, hit_counts = compute_vertex_support(
            component.vertices, views, intr, center, scale, min_valid_views
        )
        bounds = (component.bounds - center) / scale
        stats = dict(
            index=idx,
            verts=int(len(component.vertices)),
            faces=int(len(component.faces)),
            mean_support=float(support.mean()) if len(support) else 0.0,
            median_support=float(np.median(support)) if len(support) else 0.0,
            valid_ratio=float((valid_counts >= min_valid_views).mean()) if len(valid_counts) else 0.0,
            bounds=bounds.tolist(),
            bottom_z=float(bounds[0, 2]),
            top_z=float(bounds[1, 2]),
            center_z=float(bounds.mean(axis=0)[2]),
        )
        component_stats.append(stats)
    return component_stats


def flag_bottom_components(component_stats, anchor_idx, bottom_z_margin, bottom_gap):
    if bottom_z_margin <= 0:
        for stats in component_stats:
            stats["xy_gap_to_anchor"] = None
            stats["z_gap_to_anchor_bottom"] = None
            stats["bottom_cleanup_removed"] = False
        return []

    anchor_bounds = np.array(component_stats[anchor_idx]["bounds"], dtype=np.float32)
    anchor_bottom_z = float(anchor_bounds[0, 2])
    removed = []
    for stats in component_stats:
        idx = stats["index"]
        bounds = np.array(stats["bounds"], dtype=np.float32)
        xy_gap = bbox_gap_xy(bounds, anchor_bounds)
        z_gap = anchor_bottom_z - float(bounds[1, 2])
        remove = idx != anchor_idx and z_gap > bottom_z_margin and xy_gap <= bottom_gap
        stats["xy_gap_to_anchor"] = xy_gap
        stats["z_gap_to_anchor_bottom"] = z_gap
        stats["bottom_cleanup_removed"] = remove
        if remove:
            removed.append(idx)
    return removed


def component_filter(
    components,
    component_stats,
    anchor_idx,
    min_component_faces,
    min_component_support,
    near_component_faces,
    near_component_support,
    near_distance,
):
    anchor_bounds = np.array(component_stats[anchor_idx]["bounds"], dtype=np.float32)
    blocked_indices = {stats["index"] for stats in component_stats if stats.get("bottom_cleanup_removed")}
    kept_indices = []
    for stats in component_stats:
        idx = stats["index"]
        bounds = np.array(stats["bounds"], dtype=np.float32)
        dist = bbox_gap(bounds, anchor_bounds)
        stats["bbox_gap_to_anchor"] = dist
        keep = False
        if idx in blocked_indices:
            keep = False
        elif idx == anchor_idx:
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
    return merged, kept_indices


def default_summary_path(output_path):
    stem, _ = os.path.splitext(output_path)
    return f"{stem}.summary.json"


def face_filtered_output_path(output_path):
    stem, ext = os.path.splitext(output_path)
    return f"{stem}.face_filtered{ext}"


def get_vertex_colors(mesh):
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is None or len(colors) != len(mesh.vertices):
        return None
    return np.asarray(colors)


def apply_nearest_vertex_colors(source_mesh, target_mesh):
    source_colors = get_vertex_colors(source_mesh)
    if source_colors is None or len(target_mesh.vertices) == 0:
        return False, None
    tree = cKDTree(np.asarray(source_mesh.vertices))
    distances, indices = tree.query(np.asarray(target_mesh.vertices), k=1)
    target_mesh.visual.vertex_colors = source_colors[indices]
    distances = np.asarray(distances, dtype=np.float32)
    stats = dict(
        mean=float(distances.mean()) if len(distances) else 0.0,
        median=float(np.median(distances)) if len(distances) else 0.0,
        q95=float(np.quantile(distances, 0.95)) if len(distances) else 0.0,
        max=float(distances.max()) if len(distances) else 0.0,
    )
    return True, stats


def cleanup_mesh(
    mesh,
    transforms_path,
    frame_sample_count=24,
    min_valid_views=3,
    min_face_support=0.28,
    low_z_threshold=-0.40,
    low_z_support=0.40,
    min_component_faces=1200,
    min_component_support=0.45,
    near_component_faces=250,
    near_component_support=0.28,
    near_distance=0.22,
    bottom_z_margin=0.0,
    bottom_gap=0.0,
):
    meta, meta_root = load_meta(transforms_path)
    views, intr, center, scale = build_view_data(meta, meta_root, frame_sample_count)

    vertex_support, valid_counts, hit_counts = compute_vertex_support(
        mesh.vertices, views, intr, center, scale, min_valid_views
    )
    vertices_norm = (mesh.vertices - center) / scale
    filtered, face_support, face_z, keep_faces = face_filter(
        mesh,
        vertex_support,
        vertices_norm,
        min_face_support,
        low_z_threshold,
        low_z_support,
    )

    if len(filtered.faces) == 0:
        raise RuntimeError("All faces were filtered out. Relax the face support thresholds.")

    components = filtered.split(only_watertight=False)
    component_stats = compute_component_stats(
        components, views, intr, center, scale, min_valid_views
    )
    anchor_idx = choose_anchor(component_stats)
    bottom_removed = flag_bottom_components(
        component_stats, anchor_idx, bottom_z_margin, bottom_gap
    )
    merged, kept_indices = component_filter(
        components,
        component_stats,
        anchor_idx,
        min_component_faces,
        min_component_support,
        near_component_faces,
        near_component_support,
        near_distance,
    )

    face_filtered_colored, face_filtered_color_transfer = apply_nearest_vertex_colors(mesh, filtered)
    merged_colored, merged_color_transfer = apply_nearest_vertex_colors(mesh, merged)
    summary = dict(
        input_mesh=None,
        transforms=os.path.abspath(transforms_path),
        output_mesh=None,
        face_filtered_mesh=None,
        frame_sample_count=len(views),
        min_valid_views=min_valid_views,
        min_face_support=min_face_support,
        low_z_threshold=low_z_threshold,
        low_z_support=low_z_support,
        min_component_faces=min_component_faces,
        min_component_support=min_component_support,
        near_component_faces=near_component_faces,
        near_component_support=near_component_support,
        near_distance=near_distance,
        bottom_cleanup=dict(
            enabled=bottom_z_margin > 0,
            bottom_z_margin=bottom_z_margin,
            bottom_gap=bottom_gap,
            removed_components=bottom_removed,
        ),
        original=dict(
            verts=int(len(mesh.vertices)),
            faces=int(len(mesh.faces)),
            components=int(len(mesh.split(only_watertight=False))),
            bounds=mesh.bounds.tolist(),
            has_vertex_colors=get_vertex_colors(mesh) is not None,
        ),
        vertex_support=dict(
            mean=float(vertex_support.mean()) if len(vertex_support) else 0.0,
            median=float(np.median(vertex_support)) if len(vertex_support) else 0.0,
            q05=float(np.quantile(vertex_support, 0.05)) if len(vertex_support) else 0.0,
            q95=float(np.quantile(vertex_support, 0.95)) if len(vertex_support) else 0.0,
            valid_ratio=float((valid_counts >= min_valid_views).mean()) if len(valid_counts) else 0.0,
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
        color_transfer=dict(
            face_filtered_applied=face_filtered_colored,
            face_filtered_nn_distance=face_filtered_color_transfer,
            final_applied=merged_colored,
            final_nn_distance=merged_color_transfer,
        ),
        final=dict(
            verts=int(len(merged.vertices)),
            faces=int(len(merged.faces)),
            components=int(len(merged.split(only_watertight=False))),
            bounds=merged.bounds.tolist(),
            has_vertex_colors=get_vertex_colors(merged) is not None,
        ),
    )
    return merged, filtered, summary


def export_cleanup_outputs(
    mesh,
    transforms_path,
    output_path,
    summary_json=None,
    write_face_filtered=False,
    **cleanup_kwargs,
):
    merged, filtered, summary = cleanup_mesh(mesh, transforms_path, **cleanup_kwargs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.export(output_path)

    face_filtered_path = None
    if write_face_filtered:
        face_filtered_path = face_filtered_output_path(output_path)
        os.makedirs(os.path.dirname(face_filtered_path), exist_ok=True)
        filtered.export(face_filtered_path)

    summary["output_mesh"] = os.path.abspath(output_path)
    summary["face_filtered_mesh"] = os.path.abspath(face_filtered_path) if face_filtered_path else None
    summary_path = summary_json or default_summary_path(output_path)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return merged, summary, face_filtered_path, summary_path
