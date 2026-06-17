"""
Mesh cleanup helpers for Neuralangelo exports.
"""

import json
import os

import numpy as np


def default_summary_path(output_path):
    stem, _ = os.path.splitext(output_path)
    return f"{stem}.summary.json"


def default_clean_output_path(input_path):
    stem, ext = os.path.splitext(input_path)
    return f"{stem}_cleaned{ext}"


def default_preclean_backup_path(input_path):
    stem, ext = os.path.splitext(input_path)
    return f"{stem}_preclean{ext}"


def get_vertex_colors(mesh):
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is None or len(colors) != len(mesh.vertices):
        return None
    return np.asarray(colors)


def get_face_rgb(mesh):
    vertex_colors = get_vertex_colors(mesh)
    if vertex_colors is None:
        raise RuntimeError("Mesh cleanup requires vertex colors. Use a textured PLY input.")
    if len(mesh.faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return vertex_colors[mesh.faces][:, :, :3].mean(axis=1).astype(np.float32)


def get_black_face_mask(face_rgb, black_rgb_threshold):
    if len(face_rgb) == 0:
        return np.zeros(0, dtype=bool)
    return face_rgb.max(axis=1) <= float(black_rgb_threshold)


def build_black_face_components(mesh, black_face_mask):
    black_indices = np.flatnonzero(black_face_mask)
    if len(black_indices) == 0:
        return []

    neighbors = {int(idx): [] for idx in black_indices}
    for face_a, face_b in np.asarray(mesh.face_adjacency, dtype=np.int64):
        if black_face_mask[face_a] and black_face_mask[face_b]:
            neighbors[int(face_a)].append(int(face_b))
            neighbors[int(face_b)].append(int(face_a))

    components = []
    remaining = set(int(idx) for idx in black_indices)
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(np.array(component, dtype=np.int64))
    return components


def filter_black_components(
    mesh,
    face_rgb,
    black_face_mask,
    black_components,
    min_black_component_faces,
    min_black_component_area_ratio,
    min_black_component_area,
):
    total_area = float(mesh.area_faces.sum()) if len(mesh.faces) else 0.0
    remove_faces = np.zeros(len(mesh.faces), dtype=bool)
    component_stats = []
    removed_components = []

    for idx, face_indices in enumerate(black_components):
        component_area = float(mesh.area_faces[face_indices].sum())
        area_ratio = component_area / total_area if total_area > 0 else 0.0
        component_rgb = face_rgb[face_indices]
        remove = (
            len(face_indices) >= int(min_black_component_faces)
            or area_ratio >= float(min_black_component_area_ratio)
            or (
                float(min_black_component_area) > 0.0
                and component_area >= float(min_black_component_area)
            )
        )
        component_bounds = mesh.triangles_center[face_indices]
        stat = dict(
            index=idx,
            face_count=int(len(face_indices)),
            area=component_area,
            area_ratio=area_ratio,
            mean_rgb=component_rgb.mean(axis=0).round(3).tolist(),
            max_rgb=component_rgb.max(axis=0).round(3).tolist(),
            bounds_min=component_bounds.min(axis=0).round(6).tolist(),
            bounds_max=component_bounds.max(axis=0).round(6).tolist(),
            removed=bool(remove),
        )
        component_stats.append(stat)
        if remove:
            remove_faces[face_indices] = True
            removed_components.append(idx)

    return remove_faces, component_stats, removed_components, total_area


def build_clean_mesh(mesh, remove_faces):
    cleaned = mesh.copy()
    cleaned.update_faces(~remove_faces)
    cleaned.remove_unreferenced_vertices()
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()
    return cleaned


def cleanup_mesh(
    mesh,
    black_rgb_threshold=16,
    min_black_component_faces=2000,
    min_black_component_area_ratio=0.015,
    min_black_component_area=0.0,
):
    face_rgb = get_face_rgb(mesh)
    black_face_mask = get_black_face_mask(face_rgb, black_rgb_threshold)
    black_components = build_black_face_components(mesh, black_face_mask)
    remove_faces, component_stats, removed_components, total_area = filter_black_components(
        mesh,
        face_rgb,
        black_face_mask,
        black_components,
        min_black_component_faces,
        min_black_component_area_ratio,
        min_black_component_area,
    )

    cleaned = build_clean_mesh(mesh, remove_faces)
    summary = dict(
        input_mesh=None,
        output_mesh=None,
        preclean_backup_mesh=None,
        black_cleanup=dict(
            black_rgb_threshold=int(black_rgb_threshold),
            min_black_component_faces=int(min_black_component_faces),
            min_black_component_area_ratio=float(min_black_component_area_ratio),
            min_black_component_area=float(min_black_component_area),
            total_black_faces=int(black_face_mask.sum()),
            total_black_face_ratio=float(black_face_mask.mean()) if len(black_face_mask) else 0.0,
            total_black_components=int(len(black_components)),
            removed_components=removed_components,
            removed_face_count=int(remove_faces.sum()),
            removed_face_ratio=float(remove_faces.mean()) if len(remove_faces) else 0.0,
            removed_area=float(mesh.area_faces[remove_faces].sum()) if len(remove_faces) else 0.0,
            removed_area_ratio=(
                float(mesh.area_faces[remove_faces].sum()) / total_area if total_area > 0 else 0.0
            ),
            per_component=component_stats,
        ),
        original=dict(
            verts=int(len(mesh.vertices)),
            faces=int(len(mesh.faces)),
            components=int(len(mesh.split(only_watertight=False))),
            area=total_area,
            bounds=mesh.bounds.tolist(),
            has_vertex_colors=get_vertex_colors(mesh) is not None,
        ),
        final=dict(
            verts=int(len(cleaned.vertices)),
            faces=int(len(cleaned.faces)),
            components=int(len(cleaned.split(only_watertight=False))),
            area=float(cleaned.area_faces.sum()) if len(cleaned.faces) else 0.0,
            bounds=cleaned.bounds.tolist() if len(cleaned.vertices) else None,
            has_vertex_colors=get_vertex_colors(cleaned) is not None,
        ),
    )
    return cleaned, summary


def export_cleanup_outputs(
    mesh,
    output_path,
    summary_json=None,
    input_mesh_path=None,
    preclean_backup_path=None,
    **cleanup_kwargs,
):
    cleaned, summary = cleanup_mesh(mesh, **cleanup_kwargs)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cleaned.export(output_path)

    summary["input_mesh"] = os.path.abspath(input_mesh_path) if input_mesh_path else None
    summary["output_mesh"] = os.path.abspath(output_path)
    summary["preclean_backup_mesh"] = os.path.abspath(preclean_backup_path) if preclean_backup_path else None

    summary_path = summary_json or default_summary_path(output_path)
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return cleaned, summary, summary_path
