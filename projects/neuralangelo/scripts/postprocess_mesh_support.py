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
import os
import sys

import trimesh
sys.path.append(os.getcwd())
from projects.neuralangelo.utils.mesh_cleanup import export_cleanup_outputs


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
    parser.add_argument(
        "--bottom_z_margin",
        type=float,
        default=0.0,
        help="If > 0, remove kept components whose top sits this far below the anchor bottom.",
    )
    parser.add_argument(
        "--bottom_gap",
        type=float,
        default=0.0,
        help="Maximum XY bbox gap for a low component to be treated as under the anchor.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mesh = trimesh.load(args.mesh, process=False)
    _, summary, face_filtered_path, summary_path = export_cleanup_outputs(
        mesh,
        args.transforms,
        args.output,
        summary_json=args.summary_json,
        write_face_filtered=args.write_face_filtered,
        frame_sample_count=args.frame_sample_count,
        min_valid_views=args.min_valid_views,
        min_face_support=args.min_face_support,
        low_z_threshold=args.low_z_threshold,
        low_z_support=args.low_z_support,
        min_component_faces=args.min_component_faces,
        min_component_support=args.min_component_support,
        near_component_faces=args.near_component_faces,
        near_component_support=args.near_component_support,
        near_distance=args.near_distance,
        bottom_z_margin=args.bottom_z_margin,
        bottom_gap=args.bottom_gap,
    )

    print(f"Saved filtered mesh to {args.output}")
    if face_filtered_path:
        print(f"Saved face-filtered mesh to {face_filtered_path}")
    print(f"Saved summary to {summary_path}")
    print(
        "Final mesh:",
        f"verts={summary['final']['verts']}",
        f"faces={summary['final']['faces']}",
        f"components={summary['final']['components']}",
    )


if __name__ == "__main__":
    main()
