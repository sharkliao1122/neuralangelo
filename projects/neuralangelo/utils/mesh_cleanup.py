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
    faces = np.asarray(mesh.faces, dtype=np.int64)
    rgb = np.asarray(vertex_colors[:, :3], dtype=np.uint16)
    face_rgb = rgb[faces[:, 0]] + rgb[faces[:, 1]] + rgb[faces[:, 2]]
    return (face_rgb.astype(np.float32) / 3.0).astype(np.float32, copy=False)


def get_black_face_mask(face_rgb, black_rgb_threshold):
    if len(face_rgb) == 0:
        return np.zeros(0, dtype=bool)
    return face_rgb.max(axis=1) <= float(black_rgb_threshold)


def build_black_face_edge_pairs(mesh, black_indices):
    faces = np.asarray(mesh.faces, dtype=np.int64)[black_indices]
    edge_count = len(faces) * 3
    edges = np.empty((edge_count, 2), dtype=np.int64)
    edges[0::3] = faces[:, [0, 1]]
    edges[1::3] = faces[:, [1, 2]]
    edges[2::3] = faces[:, [2, 0]]
    edges.sort(axis=1)

    face_refs = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_refs = face_refs[order]

    same_edge = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
    if not np.any(same_edge):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    duplicate_edge_mask = np.zeros(len(sorted_edges), dtype=bool)
    duplicate_edge_mask[:-1] |= same_edge
    duplicate_edge_mask[1:] |= same_edge

    duplicate_edges = sorted_edges[duplicate_edge_mask]
    duplicate_refs = sorted_refs[duplicate_edge_mask]
    group_breaks = np.flatnonzero(np.any(duplicate_edges[1:] != duplicate_edges[:-1], axis=1)) + 1
    group_starts = np.concatenate(([0], group_breaks))
    group_ends = np.concatenate((group_breaks, [len(duplicate_edges)]))

    rows = []
    cols = []
    for start, end in zip(group_starts, group_ends):
        refs = duplicate_refs[start:end]
        if len(refs) < 2:
            continue
        rows.append(np.full(len(refs) - 1, refs[0], dtype=np.int64))
        cols.append(refs[1:].astype(np.int64, copy=False))

    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(rows), np.concatenate(cols)


def connected_components_from_pairs(node_count, rows, cols):
    if node_count == 0:
        return np.zeros(0, dtype=np.int64)
    if len(rows) == 0:
        return np.arange(node_count, dtype=np.int64)

    try:
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components

        graph_rows = np.concatenate((rows, cols))
        graph_cols = np.concatenate((cols, rows))
        data = np.ones(len(graph_rows), dtype=bool)
        graph = sparse.coo_matrix((data, (graph_rows, graph_cols)), shape=(node_count, node_count)).tocsr()
        _, labels = connected_components(graph, directed=False, return_labels=True)
        return labels.astype(np.int64, copy=False)
    except ImportError:
        parent = np.arange(node_count, dtype=np.int64)
        rank = np.zeros(node_count, dtype=np.uint8)

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for row, col in zip(rows, cols):
            root_row = find(int(row))
            root_col = find(int(col))
            if root_row == root_col:
                continue
            if rank[root_row] < rank[root_col]:
                parent[root_row] = root_col
            elif rank[root_row] > rank[root_col]:
                parent[root_col] = root_row
            else:
                parent[root_col] = root_row
                rank[root_row] += 1

        return np.array([find(idx) for idx in range(node_count)], dtype=np.int64)


def build_black_face_components(mesh, black_face_mask):
    black_indices = np.flatnonzero(black_face_mask)
    if len(black_indices) == 0:
        return []

    rows, cols = build_black_face_edge_pairs(mesh, black_indices)
    labels = connected_components_from_pairs(len(black_indices), rows, cols)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    component_breaks = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    component_starts = np.concatenate(([0], component_breaks))
    component_ends = np.concatenate((component_breaks, [len(sorted_labels)]))
    return [black_indices[order[start:end]] for start, end in zip(component_starts, component_ends)]


def get_face_center_bounds(mesh, face_indices, chunk_size=1000000):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)

    for start in range(0, len(face_indices), chunk_size):
        chunk_indices = face_indices[start:start + chunk_size]
        centers = vertices[faces[chunk_indices]].mean(axis=1)
        bounds_min = np.minimum(bounds_min, centers.min(axis=0))
        bounds_max = np.maximum(bounds_max, centers.max(axis=0))

    return bounds_min, bounds_max


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
        bounds_min, bounds_max = get_face_center_bounds(mesh, face_indices)
        stat = dict(
            index=idx,
            face_count=int(len(face_indices)),
            area=component_area,
            area_ratio=area_ratio,
            mean_rgb=component_rgb.mean(axis=0).round(3).tolist(),
            max_rgb=component_rgb.max(axis=0).round(3).tolist(),
            bounds_min=bounds_min.round(6).tolist(),
            bounds_max=bounds_max.round(6).tolist(),
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


def get_mesh_component_count(mesh, compute_component_counts):
    if not compute_component_counts:
        return None
    return int(len(mesh.split(only_watertight=False)))


def cleanup_mesh(
    mesh,
    black_rgb_threshold=16,
    min_black_component_faces=2000,
    min_black_component_area_ratio=0.015,
    min_black_component_area=0.0,
    compute_component_counts=True,
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
            components=get_mesh_component_count(mesh, compute_component_counts),
            area=total_area,
            bounds=mesh.bounds.tolist(),
            has_vertex_colors=get_vertex_colors(mesh) is not None,
        ),
        final=dict(
            verts=int(len(cleaned.vertices)),
            faces=int(len(cleaned.faces)),
            components=get_mesh_component_count(cleaned, compute_component_counts),
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
