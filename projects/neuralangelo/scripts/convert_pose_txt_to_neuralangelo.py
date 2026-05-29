'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import json
import math
import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image


def read_pose_matrices(pose_file):
    with open(pose_file, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    if len(lines) % 4 != 0:
        raise ValueError(f"{pose_file} does not contain a whole number of 4x4 matrices.")
    poses = []
    for idx in range(0, len(lines), 4):
        rows = [[float(token) for token in lines[idx + row].split()] for row in range(4)]
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Pose block {idx // 4} is not 4x4.")
        poses.append(matrix)
    return poses


def sorted_image_names(image_dir):
    image_names = [name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))]
    if not image_names:
        raise ValueError(f"No images found under {image_dir}.")

    def numeric_key(name):
        stem, _ = os.path.splitext(name)
        try:
            return int(stem)
        except ValueError as exc:
            raise ValueError(
                f"Image '{name}' is not numerically named; numeric sorting is required to align with pose order."
            ) from exc

    return sorted(image_names, key=numeric_key)


def gl_to_cv(matrix):
    return matrix * np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)


def cv_to_gl(matrix):
    return matrix * np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)


def to_cv_c2w(matrix, input_pose_type, input_convention):
    if input_pose_type == "w2c":
        matrix = np.linalg.inv(matrix)
    if input_convention == "gl":
        matrix = gl_to_cv(matrix)
    return matrix


def find_closest_point(p1, d1, p2, d2):
    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)
    coeff = np.vstack((d1_norm, -d2_norm)).T
    rhs = p2 - p1
    t1, t2 = np.linalg.lstsq(coeff, rhs, rcond=None)[0]
    closest_p1 = p1 + d1_norm * t1
    closest_p2 = p2 + d2_norm * t2
    return 0.5 * (closest_p1 + closest_p2)


def estimate_scene_bounds(c2w_cv_list):
    poses = np.stack(c2w_cv_list)
    camera_positions = poses[:, :3, 3]
    forward_vectors = poses[:, :3, 2]

    center = np.zeros(3, dtype=np.float64)
    for src in range(len(poses)):
        for dst in range(len(poses)):
            center += find_closest_point(
                camera_positions[src],
                forward_vectors[src],
                camera_positions[dst],
                forward_vectors[dst],
            )
    center /= len(poses) ** 2

    radius = np.linalg.norm(camera_positions - center[None], axis=1).mean()
    bounding_box = [
        [float(center[0] - radius), float(center[0] + radius)],
        [float(center[1] - radius), float(center[1] + radius)],
        [float(center[2] - radius), float(center[2] + radius)],
    ]
    return center, float(radius), bounding_box


def export_transforms(args):
    image_dir = os.path.join(args.data_dir, args.image_dir)
    pose_file = os.path.join(args.data_dir, args.pose_file)
    focal_file = os.path.join(args.data_dir, args.focal_file)

    image_names = sorted_image_names(image_dir)
    c2w_input = read_pose_matrices(pose_file)
    if len(image_names) != len(c2w_input):
        raise ValueError(
            f"Image count ({len(image_names)}) does not match pose count ({len(c2w_input)})."
        )

    with open(focal_file, "r", encoding="utf-8") as file:
        focal = float(file.read().strip())

    sample_image = Image.open(os.path.join(image_dir, image_names[0]))
    width, height = sample_image.size
    sample_image.close()

    cx = args.cx if args.cx is not None else width / 2.0
    cy = args.cy if args.cy is not None else height / 2.0

    c2w_cv_list = [
        to_cv_c2w(matrix, args.input_pose_type, args.input_convention)
        for matrix in c2w_input
    ]
    center, radius, bounding_box = estimate_scene_bounds(c2w_cv_list)
    radius *= args.radius_scale
    bounding_box = [
        [float(center[axis] - radius), float(center[axis] + radius)]
        for axis in range(3)
    ]

    angle_x = math.atan(width / (focal * 2.0)) * 2.0
    angle_y = math.atan(height / (focal * 2.0)) * 2.0
    transforms = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": focal,
        "fl_y": focal,
        "sk_x": 0.0,
        "sk_y": 0.0,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "is_fisheye": False,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "aabb_scale": float(2 ** np.rint(np.log2(radius))),
        "aabb_range": bounding_box,
        "sphere_center": center.tolist(),
        "sphere_radius": radius,
        "frames": [],
    }

    for image_name, c2w_cv in zip(image_names, c2w_cv_list):
        transforms["frames"].append({
            "file_path": f"{args.image_dir}/{image_name}".replace("\\", "/"),
            "transform_matrix": cv_to_gl(c2w_cv).tolist(),
        })

    output_path = os.path.join(args.data_dir, args.output_file)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(transforms, file, indent=2)
    print(f"Wrote {output_path}")
    print(f"image_count={len(image_names)}")
    print(f"sphere_center={center.tolist()}")
    print(f"sphere_radius={radius}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Dataset root directory.")
    parser.add_argument("--pose_file", default="trainval_poses.txt", help="Relative path to the 4x4 pose text file.")
    parser.add_argument("--focal_file", default="focal.txt", help="Relative path to the focal-length text file.")
    parser.add_argument("--image_dir", default="images", help="Relative path to the image directory.")
    parser.add_argument("--output_file", default="transforms.json", help="Relative path for the output transforms JSON.")
    parser.add_argument("--input_pose_type", choices=["c2w", "w2c"], default="c2w",
                        help="Whether the pose matrices are camera-to-world or world-to-camera.")
    parser.add_argument("--input_convention", choices=["gl", "cv"], default="gl",
                        help="Whether the input matrices use OpenGL or OpenCV camera convention.")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x in pixels. Defaults to image center.")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y in pixels. Defaults to image center.")
    parser.add_argument("--radius_scale", type=float, default=1.0,
                        help="Multiplier applied to the estimated scene radius.")
    args = parser.parse_args()
    export_transforms(args)
