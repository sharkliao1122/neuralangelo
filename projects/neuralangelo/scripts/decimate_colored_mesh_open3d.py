"""Create a lower-triangle-count proxy mesh while preserving vertex colors."""

import argparse
from pathlib import Path

import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target_faces", required=True, type=int)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.target_faces <= 0:
        raise ValueError("--target_faces must be positive")

    print(f"Reading: {args.input}", flush=True)
    mesh = o3d.io.read_triangle_mesh(str(args.input), enable_post_processing=False)
    source_vertices = len(mesh.vertices)
    source_faces = len(mesh.triangles)
    print(
        f"Source: {source_vertices:,} vertices, {source_faces:,} faces, "
        f"colors={mesh.has_vertex_colors()}",
        flush=True,
    )
    if source_faces == 0:
        raise RuntimeError("Input contains no triangle faces")

    print(f"Decimating to {args.target_faces:,} faces...", flush=True)
    proxy = mesh.simplify_quadric_decimation(args.target_faces)
    proxy.remove_degenerate_triangles()
    proxy.remove_duplicated_triangles()
    proxy.remove_unreferenced_vertices()
    proxy.compute_vertex_normals()
    print(
        f"Proxy: {len(proxy.vertices):,} vertices, {len(proxy.triangles):,} faces, "
        f"colors={proxy.has_vertex_colors()}, normals={proxy.has_vertex_normals()}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing: {args.output}", flush=True)
    if not o3d.io.write_triangle_mesh(
        str(args.output),
        proxy,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    ):
        raise RuntimeError(f"Failed to write {args.output}")
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
