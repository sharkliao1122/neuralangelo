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
import glob
import os
import shutil
import sys

import trimesh

sys.path.append(os.getcwd())
from projects.neuralangelo.utils.mesh_cleanup import (  # noqa: E402
    default_clean_output_path,
    default_preclean_backup_path,
    export_cleanup_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove large connected black regions from a textured Neuralangelo mesh."
    )
    parser.add_argument("--mesh", default=None, help="Input mesh path. If omitted, auto-detect from --search_dir.")
    parser.add_argument("--search_dir", default=".", help="Directory used when auto-detecting a mesh.")
    parser.add_argument("--mesh_glob", default="*texture.ply",
                        help="Glob used when --mesh is omitted. The newest match under --search_dir is used.")
    parser.add_argument("--output", default=None, help="Output mesh path. Defaults to *_cleaned next to the input mesh.")
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Optional JSON file for a detailed cleanup summary.",
    )
    parser.add_argument(
        "--preclean_backup",
        default=None,
        help="Optional backup path used only when --output would overwrite the input mesh.",
    )
    parser.add_argument(
        "--black_rgb_threshold",
        type=int,
        default=16,
        help="Conservative black threshold. Faces are black only when all average RGB channels are <= this value.",
    )
    parser.add_argument(
        "--min_black_component_faces",
        type=int,
        default=2000,
        help="Minimum connected black-face count required to remove a region.",
    )
    parser.add_argument(
        "--min_black_component_area_ratio",
        type=float,
        default=0.015,
        help="Minimum mesh area ratio required to remove a black region.",
    )
    parser.add_argument(
        "--min_black_component_area",
        type=float,
        default=0.0,
        help="Optional absolute surface-area threshold for removing a black region. Use 0 to disable.",
    )
    parser.add_argument(
        "--skip_component_counts",
        action="store_true",
        help="Skip expensive full-mesh connected component counts in the JSON summary.",
    )
    return parser.parse_args()


def resolve_mesh_path(mesh_path, search_dir, mesh_glob):
    if mesh_path:
        resolved = os.path.abspath(mesh_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Input mesh not found: {resolved}")
        return resolved

    search_pattern = os.path.join(os.path.abspath(search_dir), "**", mesh_glob)
    matches = sorted(glob.glob(search_pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No mesh matched {mesh_glob!r} under {os.path.abspath(search_dir)}")
    newest = max(matches, key=lambda path: (os.path.getmtime(path), path))
    return os.path.abspath(newest)


def preserve_preclean_mesh(input_mesh_path, output_mesh_path, backup_path):
    if os.path.abspath(input_mesh_path) != os.path.abspath(output_mesh_path):
        return None
    resolved_backup_path = os.path.abspath(backup_path or default_preclean_backup_path(input_mesh_path))
    if resolved_backup_path == os.path.abspath(input_mesh_path):
        raise RuntimeError("--preclean_backup must be different from the input mesh path when overwriting.")
    backup_dir = os.path.dirname(resolved_backup_path)
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(input_mesh_path, resolved_backup_path)
    return resolved_backup_path


def main():
    args = parse_args()
    mesh_path = resolve_mesh_path(args.mesh, args.search_dir, args.mesh_glob)
    output_path = os.path.abspath(args.output) if args.output else default_clean_output_path(mesh_path)
    preclean_backup_path = preserve_preclean_mesh(mesh_path, output_path, args.preclean_backup)

    mesh = trimesh.load(mesh_path, process=False)
    _, summary, summary_path = export_cleanup_outputs(
        mesh,
        output_path,
        summary_json=args.summary_json,
        input_mesh_path=mesh_path,
        preclean_backup_path=preclean_backup_path,
        black_rgb_threshold=args.black_rgb_threshold,
        min_black_component_faces=args.min_black_component_faces,
        min_black_component_area_ratio=args.min_black_component_area_ratio,
        min_black_component_area=args.min_black_component_area,
        compute_component_counts=not args.skip_component_counts,
    )

    print(f"Source mesh: {mesh_path}")
    if preclean_backup_path:
        print(f"Saved preclean backup to {preclean_backup_path}")
    print(f"Saved cleaned mesh to {output_path}")
    print(f"Saved summary to {summary_path}")
    print(
        "Final mesh:",
        f"verts={summary['final']['verts']}",
        f"faces={summary['final']['faces']}",
        f"components={summary['final']['components']}",
    )


if __name__ == "__main__":
    main()
