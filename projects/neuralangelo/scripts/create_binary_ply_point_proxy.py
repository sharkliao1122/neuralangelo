"""Create a lightweight proxy from a binary little-endian PLY point cloud."""

import argparse
from pathlib import Path

import numpy as np


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


def read_vertex_layout(path):
    properties = []
    vertex_count = None
    in_vertices = False
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Input is not a PLY file")
        if stream.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("Only binary_little_endian PLY files are supported")

        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header is missing end_header")
            text = line.decode("ascii").strip()
            if text.startswith("element "):
                parts = text.split()
                in_vertices = parts[1] == "vertex"
                if in_vertices:
                    vertex_count = int(parts[2])
            elif text.startswith("property ") and in_vertices:
                parts = text.split()
                if parts[1] == "list":
                    raise ValueError("List properties are not supported in the vertex element")
                if parts[1] not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property type: {parts[1]}")
                properties.append((parts[2], PLY_TYPES[parts[1]], parts[1]))
            elif text == "end_header":
                data_offset = stream.tell()
                break

    if vertex_count is None or not properties:
        raise ValueError("PLY vertex element was not found")

    dtype = np.dtype([(name, numpy_type) for name, numpy_type, _ in properties])
    return vertex_count, properties, dtype, data_offset


def write_proxy(input_path, output_path, target_points, bounds, seed, chunk_size):
    vertex_count, properties, dtype, data_offset = read_vertex_layout(input_path)
    required = {"x", "y", "z"}
    if not required.issubset(dtype.names):
        raise ValueError("PLY must contain x, y, and z vertex properties")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    probability = min(1.0, target_points / vertex_count)
    rng = np.random.default_rng(seed)
    selected_chunks = []
    selected_count = 0

    with input_path.open("rb") as stream:
        stream.seek(data_offset)
        remaining = vertex_count
        processed = 0
        while remaining:
            count = min(chunk_size, remaining)
            vertices = np.fromfile(stream, dtype=dtype, count=count)
            if len(vertices) != count:
                raise IOError(f"Unexpected end of file after {processed} vertices")

            keep = rng.random(count) < probability
            if bounds is not None:
                xmin, xmax, ymin, ymax, zmin, zmax = bounds
                keep &= np.isfinite(vertices["x"])
                keep &= np.isfinite(vertices["y"])
                keep &= np.isfinite(vertices["z"])
                keep &= (vertices["x"] >= xmin) & (vertices["x"] <= xmax)
                keep &= (vertices["y"] >= ymin) & (vertices["y"] <= ymax)
                keep &= (vertices["z"] >= zmin) & (vertices["z"] <= zmax)

            chosen = vertices[keep].copy()
            if len(chosen):
                selected_chunks.append(chosen)
                selected_count += len(chosen)

            processed += count
            remaining -= count
            print(
                f"\rProcessed {processed:,}/{vertex_count:,}; selected {selected_count:,}",
                end="",
                flush=True,
            )

    print()
    selected = np.concatenate(selected_chunks) if selected_chunks else np.empty(0, dtype=dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reverse_types = {value: key for key, value in PLY_TYPES.items()}
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment proxy_source {input_path}",
        f"comment random_seed {seed}",
        f"comment source_vertices {vertex_count}",
        f"element vertex {len(selected)}",
    ]
    for name, numpy_type, original_type in properties:
        ply_type = original_type if original_type in PLY_TYPES else reverse_types[numpy_type]
        header_lines.append(f"property {ply_type} {name}")
    header_lines.append("end_header")

    with output_path.open("wb") as stream:
        stream.write(("\n".join(header_lines) + "\n").encode("ascii"))
        selected.tofile(stream)

    return vertex_count, len(selected), dtype.itemsize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target_points", type=int, default=2_000_000)
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--chunk_size", type=int, default=2_000_000)
    args = parser.parse_args()

    source_count, output_count, record_size = write_proxy(
        args.input,
        args.output,
        args.target_points,
        args.bounds,
        args.seed,
        args.chunk_size,
    )
    print(f"Source vertices: {source_count:,}")
    print(f"Proxy vertices: {output_count:,}")
    print(f"Vertex record bytes: {record_size}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
