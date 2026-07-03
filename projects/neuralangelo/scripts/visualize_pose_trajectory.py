import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


def natural_sort_key(text: str):
    parts = re.split(r"(\d+)", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def load_transforms(transforms_path: Path):
    return json.loads(transforms_path.read_text(encoding="utf-8"))


def load_frames(transforms_data, sort_mode: str):
    frames = transforms_data.get("frames", [])
    if not frames:
        raise RuntimeError("No frames found in transforms data")
    if sort_mode == "natural":
        frames = sorted(frames, key=lambda frame: natural_sort_key(frame["file_path"]))
    elif sort_mode != "json":
        raise ValueError(f"Unsupported sort_mode: {sort_mode}")
    return frames


def subsample_frames(frames, frame_step: int, max_frames: int | None):
    num_frames = len(frames)
    indices = list(range(0, num_frames, max(1, frame_step)))
    if indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        sampled = np.linspace(0, len(indices) - 1, num=max_frames, dtype=int)
        indices = [indices[idx] for idx in sampled]
        if indices[-1] != num_frames - 1:
            indices[-1] = num_frames - 1
    return [frames[idx] for idx in indices], indices


def get_camera_centers(frames):
    centers = []
    for frame in frames:
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise RuntimeError(f"Expected 4x4 transform_matrix, got {matrix.shape}")
        centers.append(matrix[:3, 3])
    return np.asarray(centers, dtype=np.float64)


def get_camera_forward_vectors(frames):
    forward_vectors = []
    for frame in frames:
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise RuntimeError(f"Expected 4x4 transform_matrix, got {matrix.shape}")
        # Stored transforms use GL convention; the camera looks along -Z in local space.
        forward = -matrix[:3, 2]
        norm = np.linalg.norm(forward)
        if norm < 1e-12:
            raise RuntimeError("Encountered near-zero camera forward vector")
        forward_vectors.append(forward / norm)
    return np.asarray(forward_vectors, dtype=np.float64)


def estimate_object_center(transforms_data, centers, forward_vectors, center_source: str):
    sphere_center = transforms_data.get("sphere_center")
    if center_source in {"auto", "sphere"} and sphere_center is not None:
        center = np.asarray(sphere_center, dtype=np.float64)
        if center.shape != (3,):
            raise RuntimeError(f"Expected sphere_center shape (3,), got {center.shape}")
        return center, "sphere_center"
    if center_source == "sphere":
        raise RuntimeError("Requested --object_center_source=sphere but transforms.json has no sphere_center")

    system_matrix = np.zeros((3, 3), dtype=np.float64)
    system_rhs = np.zeros(3, dtype=np.float64)
    for camera_center, forward in zip(centers, forward_vectors):
        projection = np.eye(3, dtype=np.float64) - np.outer(forward, forward)
        system_matrix += projection
        system_rhs += projection @ camera_center
    center, *_ = np.linalg.lstsq(system_matrix, system_rhs, rcond=None)
    return center, "look_at_lstsq"


def recenter_scene(centers, object_center, center_object_at_origin: bool):
    if not center_object_at_origin:
        return centers, object_center
    return centers - object_center[None], np.zeros(3, dtype=np.float64)


def set_equal_xyz(ax, xyz, pad_ratio: float = 0.08):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    mids = 0.5 * (mins + maxs)
    radius = 0.5 * max(np.max(maxs - mins), 1e-6)
    radius *= 1.0 + pad_ratio
    ax.set_xlim(mids[0] - radius, mids[0] + radius)
    ax.set_ylim(mids[1] - radius, mids[1] + radius)
    ax.set_zlim(mids[2] - radius, mids[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def setup_3d_axes(ax3d, scene_points, title: str):
    ax3d.set_title(title)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.grid(True, linewidth=0.4, alpha=0.35)
    set_equal_xyz(ax3d, scene_points)


def get_orientation_length(centers, orientation_scale: float):
    spans = centers.max(axis=0) - centers.min(axis=0)
    base_span = max(float(np.max(spans)), 1e-6)
    return base_span * max(orientation_scale, 0.0)


def draw_orientation_quiver(ax3d, center, forward, length, color="#D62728", alpha=0.95, linewidth=1.4):
    return ax3d.quiver(
        center[0],
        center[1],
        center[2],
        forward[0],
        forward[1],
        forward[2],
        length=length,
        normalize=True,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
    )


def draw_object_center_marker(ax3d, object_center, label: str = "Object"):
    scatter = ax3d.scatter(
        [object_center[0]],
        [object_center[1]],
        [object_center[2]],
        color="#111111",
        edgecolors="#FFD166",
        linewidths=1.5,
        s=120,
        marker="o",
        zorder=5,
    )
    text = ax3d.text(
        object_center[0],
        object_center[1],
        object_center[2],
        f" {label}",
        color="#111111",
    )
    return scatter, text


def draw_focus_line(ax3d, camera_center, object_center, color="#2A9D8F", alpha=0.35, linewidth=1.0):
    line, = ax3d.plot(
        [camera_center[0], object_center[0]],
        [camera_center[1], object_center[1]],
        [camera_center[2], object_center[2]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle="--",
    )
    return line


def save_animation(anim, output_path: Path, fps: int, dpi: int, video_codec: str, video_bitrate: int):
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=max(40, dpi))
        return
    if suffix == ".mp4":
        writer = animation.FFMpegWriter(
            fps=fps,
            codec=video_codec,
            bitrate=video_bitrate,
            extra_args=["-pix_fmt", "yuv420p"],
        )
        anim.save(output_path, writer=writer, dpi=max(40, dpi))
        return
    raise ValueError("Animated output currently supports only .gif or .mp4")


def draw_trajectory_figure(
    centers,
    forward_vectors,
    object_center,
    object_center_source,
    frames,
    output_path: Path,
    title: str,
    show_orientation: bool,
    orientation_scale: float,
    orientation_stride: int,
    show_object_center: bool,
    show_focus_line: bool,
    focus_stride: int,
):
    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    ax3d = fig.add_subplot(1, 1, 1, projection="3d")

    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    colors = np.linspace(0.0, 1.0, len(centers))
    scene_points = centers if object_center is None else np.vstack([centers, object_center[None]])

    ax3d.plot(x, y, z, color="#8A8F98", linewidth=1.2, alpha=0.85)
    pts = ax3d.scatter(x, y, z, c=colors, cmap="viridis", s=18)
    ax3d.scatter([x[0]], [y[0]], [z[0]], color="#1B9E77", s=70, marker="o")
    ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color="#D95F02", s=80, marker="X")
    ax3d.text(x[0], y[0], z[0], " Start", color="#1B9E77")
    ax3d.text(x[-1], y[-1], z[-1], " End", color="#D95F02")
    if show_object_center and object_center is not None:
        draw_object_center_marker(ax3d, object_center)
    if show_focus_line and object_center is not None:
        for idx in range(0, len(centers), max(1, focus_stride)):
            draw_focus_line(
                ax3d,
                centers[idx],
                object_center,
                alpha=0.22,
                linewidth=0.9,
            )
    if show_orientation:
        orientation_length = get_orientation_length(scene_points, orientation_scale=orientation_scale)
        stride = max(1, orientation_stride)
        for idx in range(0, len(centers), stride):
            draw_orientation_quiver(
                ax3d,
                centers[idx],
                forward_vectors[idx],
                length=orientation_length,
                color="#C44E52",
                alpha=0.55,
                linewidth=1.0,
            )
    setup_3d_axes(ax3d, scene_points, title=title)

    cbar = fig.colorbar(pts, ax=ax3d, shrink=0.72, pad=0.04)
    cbar.set_label("Frame Order")

    file_names = [Path(frame["file_path"]).name for frame in frames]
    fig.suptitle(
        f"{title}\nFrames: {len(frames)} | Start: {file_names[0]} | End: {file_names[-1]} | Center: {object_center_source}",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_trajectory_animation(
    centers,
    forward_vectors,
    object_center,
    object_center_source,
    frames,
    output_path: Path,
    title: str,
    fps: int,
    tail_length: int,
    show_orientation: bool,
    orientation_scale: float,
    show_object_center: bool,
    show_focus_line: bool,
    animation_dpi: int,
    video_codec: str,
    video_bitrate: int,
):
    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    ax3d = fig.add_subplot(1, 1, 1, projection="3d")

    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    colors = np.linspace(0.0, 1.0, len(centers))
    file_names = [Path(frame["file_path"]).name for frame in frames]
    scene_points = centers if object_center is None else np.vstack([centers, object_center[None]])
    orientation_length = get_orientation_length(scene_points, orientation_scale=orientation_scale)

    setup_3d_axes(ax3d, scene_points, title=title)

    start_marker = ax3d.scatter([x[0]], [y[0]], [z[0]], color="#1B9E77", s=70, marker="o")
    end_marker = ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color="#D95F02", s=80, marker="X")
    ax3d.text(x[0], y[0], z[0], " Start", color="#1B9E77")
    ax3d.text(x[-1], y[-1], z[-1], " End", color="#D95F02")
    object_marker = object_label = None
    if show_object_center and object_center is not None:
        object_marker, object_label = draw_object_center_marker(ax3d, object_center)

    tail_line, = ax3d.plot([], [], [], color="#8A8F98", linewidth=1.6, alpha=0.95)
    current_point = ax3d.scatter([], [], [], c=[], cmap="viridis", vmin=0.0, vmax=1.0, s=36)
    title_text = fig.suptitle("", fontsize=13)
    cbar = fig.colorbar(current_point, ax=ax3d, shrink=0.72, pad=0.04)
    cbar.set_label("Frame Order")
    orientation_artist = None
    focus_line_artist = None

    def update(frame_idx: int):
        nonlocal orientation_artist, focus_line_artist
        start_idx = max(0, frame_idx - tail_length + 1) if tail_length > 0 else 0
        tail_line.set_data(x[start_idx : frame_idx + 1], y[start_idx : frame_idx + 1])
        tail_line.set_3d_properties(z[start_idx : frame_idx + 1])
        current_point._offsets3d = ([x[frame_idx]], [y[frame_idx]], [z[frame_idx]])
        current_point.set_array(np.asarray([colors[frame_idx]]))
        if orientation_artist is not None:
            orientation_artist.remove()
            orientation_artist = None
        if show_orientation:
            orientation_artist = draw_orientation_quiver(
                ax3d,
                centers[frame_idx],
                forward_vectors[frame_idx],
                length=orientation_length,
                color="#C44E52",
                alpha=0.98,
                linewidth=1.8,
            )
        if focus_line_artist is not None:
            focus_line_artist.remove()
            focus_line_artist = None
        if show_focus_line and object_center is not None:
            focus_line_artist = draw_focus_line(
                ax3d,
                centers[frame_idx],
                object_center,
                alpha=0.45,
                linewidth=1.3,
            )
        title_text.set_text(
            f"{title}\nFrame {frame_idx + 1}/{len(frames)} | File: {file_names[frame_idx]} | Center: {object_center_source}"
        )
        artists = [tail_line, current_point, start_marker, end_marker, title_text]
        if object_marker is not None:
            artists.append(object_marker)
        if object_label is not None:
            artists.append(object_label)
        if orientation_artist is not None:
            artists.append(orientation_artist)
        if focus_line_artist is not None:
            artists.append(focus_line_artist)
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=max(1, int(round(1000 / max(1, fps)))),
        blit=False,
        repeat=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_animation(
        anim,
        output_path,
        fps=fps,
        dpi=max(40, animation_dpi),
        video_codec=video_codec,
        video_bitrate=video_bitrate,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize camera-center trajectory from Neuralangelo transforms.json")
    parser.add_argument("--transforms_json", type=Path, required=True, help="Path to transforms.json")
    parser.add_argument("--output", type=Path, required=True, help="Output path (.png for static image, .gif/.mp4 for animation)")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title")
    parser.add_argument(
        "--sort_mode",
        choices=["natural", "json"],
        default="natural",
        help="How to order frames before connecting the trajectory",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Export an animation instead of a static image",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Animation frames per second when --animate is enabled",
    )
    parser.add_argument(
        "--tail_length",
        type=int,
        default=40,
        help="How many recent camera centers remain visible in the animated trail; 0 means full history",
    )
    parser.add_argument(
        "--show_orientation",
        action="store_true",
        help="Draw camera viewing direction arrows",
    )
    parser.add_argument(
        "--orientation_scale",
        type=float,
        default=0.12,
        help="Arrow length as a fraction of the trajectory bounding span",
    )
    parser.add_argument(
        "--orientation_stride",
        type=int,
        default=10,
        help="Static image only: draw one orientation arrow every N frames",
    )
    parser.add_argument(
        "--object_center_source",
        choices=["auto", "sphere", "lookat"],
        default="auto",
        help="How to estimate the object center for orbit visualization",
    )
    parser.add_argument(
        "--show_object_center",
        action="store_true",
        help="Draw the estimated object center",
    )
    parser.add_argument(
        "--show_focus_line",
        action="store_true",
        help="Draw line(s) from camera center(s) to the estimated object center",
    )
    parser.add_argument(
        "--focus_stride",
        type=int,
        default=24,
        help="Static image only: draw one camera-to-object line every N frames",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Use every N-th frame before plotting/animating; larger values are much faster",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on plotted/animated frames after subsampling",
    )
    parser.add_argument(
        "--animation_dpi",
        type=int,
        default=140,
        help="Animation export dpi; lower values are faster and smaller",
    )
    parser.add_argument(
        "--video_codec",
        type=str,
        default="libx264",
        help="Video codec for MP4 export",
    )
    parser.add_argument(
        "--video_bitrate",
        type=int,
        default=2400,
        help="Video bitrate in kbps for MP4 export",
    )
    parser.add_argument(
        "--center_object_at_origin",
        action="store_true",
        help="Translate the plotted scene so the estimated object center becomes (0,0,0)",
    )
    args = parser.parse_args()

    transforms_data = load_transforms(args.transforms_json)
    frames_all = load_frames(transforms_data, sort_mode=args.sort_mode)
    frames, sampled_indices = subsample_frames(
        frames_all,
        frame_step=max(1, args.frame_step),
        max_frames=args.max_frames,
    )
    centers = get_camera_centers(frames)
    forward_vectors = get_camera_forward_vectors(frames)
    object_center, object_center_source = estimate_object_center(
        transforms_data,
        centers,
        forward_vectors,
        center_source=args.object_center_source,
    )
    centers_plot, object_center_plot = recenter_scene(
        centers,
        object_center,
        center_object_at_origin=args.center_object_at_origin,
    )
    title = args.title or args.transforms_json.parent.name
    output_suffix = args.output.suffix.lower()
    animate = args.animate or output_suffix in {".gif", ".mp4"}
    if animate:
        if output_suffix not in {".gif", ".mp4"}:
            raise ValueError("Animated output currently requires a .gif or .mp4 output path")
        draw_trajectory_animation(
            centers_plot,
            forward_vectors,
            object_center_plot,
            object_center_source,
            frames,
            args.output,
            title=title,
            fps=max(1, args.fps),
            tail_length=max(0, args.tail_length),
            show_orientation=args.show_orientation,
            orientation_scale=max(0.0, args.orientation_scale),
            show_object_center=args.show_object_center,
            show_focus_line=args.show_focus_line,
            animation_dpi=max(40, args.animation_dpi),
            video_codec=args.video_codec,
            video_bitrate=max(1, args.video_bitrate),
        )
    else:
        if output_suffix != ".png":
            raise ValueError("Static output currently requires a .png output path")
        draw_trajectory_figure(
            centers_plot,
            forward_vectors,
            object_center_plot,
            object_center_source,
            frames,
            args.output,
            title=title,
            show_orientation=args.show_orientation,
            orientation_scale=max(0.0, args.orientation_scale),
            orientation_stride=max(1, args.orientation_stride),
            show_object_center=args.show_object_center,
            show_focus_line=args.show_focus_line,
            focus_stride=max(1, args.focus_stride),
        )

    print(
        json.dumps(
            {
                "transforms_json": str(args.transforms_json),
                "output": str(args.output),
                "num_frames": len(frames),
                "sort_mode": args.sort_mode,
                "num_frames_original": len(frames_all),
                "animate": animate,
                "fps": max(1, args.fps),
                "tail_length": max(0, args.tail_length),
                "show_orientation": args.show_orientation,
                "orientation_scale": max(0.0, args.orientation_scale),
                "orientation_stride": max(1, args.orientation_stride),
                "object_center_source_requested": args.object_center_source,
                "object_center_source_used": object_center_source,
                "object_center_world": object_center.tolist(),
                "object_center_plot": object_center_plot.tolist(),
                "center_object_at_origin": args.center_object_at_origin,
                "show_object_center": args.show_object_center,
                "show_focus_line": args.show_focus_line,
                "focus_stride": max(1, args.focus_stride),
                "frame_step": max(1, args.frame_step),
                "max_frames": args.max_frames,
                "animation_dpi": max(40, args.animation_dpi),
                "video_codec": args.video_codec,
                "video_bitrate": max(1, args.video_bitrate),
                "sampled_frame_count": len(frames),
                "sampled_frame_indices_preview": sampled_indices[:10],
                "first_frame": frames[0]["file_path"],
                "last_frame": frames[-1]["file_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
