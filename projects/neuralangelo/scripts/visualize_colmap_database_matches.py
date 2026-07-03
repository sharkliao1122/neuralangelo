import argparse
import json
import math
import sqlite3
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


MAX_IMAGE_ID = 2**31 - 1


def image_ids_to_pair_id(image_id1: int, image_id2: int):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return int(image_id1 * MAX_IMAGE_ID + image_id2)


def pair_id_to_image_ids(pair_id: int):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def load_image_map(con):
    rows = con.execute("SELECT image_id, name FROM images ORDER BY image_id").fetchall()
    return {int(image_id): name for image_id, name in rows}


def load_name_to_id_map(image_map):
    return {name: image_id for image_id, name in image_map.items()}


def resolve_image_name(image_name: str, name_to_id_map):
    if image_name in name_to_id_map:
        return name_to_id_map[image_name], image_name

    basename_matches = [name for name in name_to_id_map if Path(name).name == Path(image_name).name]
    if len(basename_matches) == 1:
        resolved_name = basename_matches[0]
        return name_to_id_map[resolved_name], resolved_name
    if len(basename_matches) > 1:
        raise RuntimeError(
            f"Image name '{image_name}' is ambiguous. Matches: {', '.join(sorted(basename_matches)[:10])}"
        )
    raise RuntimeError(f"Image name '{image_name}' was not found in COLMAP images table.")


def load_keypoints(con, image_id: int):
    row = con.execute(
        "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"No keypoints found for image_id={image_id}")
    rows, cols, data = row
    return blob_to_array(data, np.float32, (rows, cols))


def load_pair_rows(con, table_name: str):
    if table_name not in {"matches", "two_view_geometries"}:
        raise ValueError(f"Unsupported table: {table_name}")
    query = (
        f"SELECT pair_id, rows FROM {table_name} "
        f"WHERE rows > 0 AND data IS NOT NULL "
        f"ORDER BY rows DESC, pair_id ASC"
    )
    return con.execute(query).fetchall()


def load_pair_row_count_map(con, table_name: str):
    if table_name not in {"matches", "two_view_geometries"}:
        raise ValueError(f"Unsupported table: {table_name}")
    query = f"SELECT pair_id, rows FROM {table_name} WHERE rows > 0 AND data IS NOT NULL"
    return {int(pair_id): int(rows) for pair_id, rows in con.execute(query).fetchall()}


def load_matches_for_pair(con, table_name: str, pair_id: int):
    if table_name not in {"matches", "two_view_geometries"}:
        raise ValueError(f"Unsupported table: {table_name}")
    row = con.execute(
        f"SELECT rows, cols, data FROM {table_name} WHERE pair_id = ?",
        (pair_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"pair_id={pair_id} not found in {table_name}")
    rows, cols, data = row
    if rows <= 0 or data is None:
        raise RuntimeError(
            f"pair_id={pair_id} in {table_name} has no stored match coordinates "
            f"(rows={rows}, data_is_none={data is None})."
        )
    return blob_to_array(data, np.uint32, (rows, cols))


def parse_limit_spec(limit_spec: str, total_pairs: int, source_name: str):
    spec = limit_spec.strip()
    if spec == "":
        return total_pairs, {
            "input": spec,
            "mode": "all",
            "description": f"Exporting all {total_pairs} pairs.",
        }

    if spec.endswith("%"):
        percent_text = spec[:-1].strip()
        try:
            ratio = float(percent_text) / 100.0
        except ValueError as exc:
            raise RuntimeError(f"{source_name}='{limit_spec}' is not a valid percentage.") from exc
        if not (0.0 < ratio <= 1.0):
            raise RuntimeError(f"{source_name}='{limit_spec}' must be between 0% and 100%.")
        count = math.floor(total_pairs * ratio)
        if count < 1:
            raise RuntimeError(
                f"{source_name}='{limit_spec}' resolves to 0 pairs after flooring. "
                f"Please choose a larger ratio."
            )
        return count, {
            "input": limit_spec,
            "mode": "percentage",
            "ratio": ratio,
            "description": f"Using {limit_spec} of {total_pairs} pairs -> {count} pairs after flooring.",
        }

    if "/" in spec:
        parts = spec.split("/", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise RuntimeError(f"{source_name}='{limit_spec}' is not a valid fraction.")
        try:
            numerator = float(parts[0].strip())
            denominator = float(parts[1].strip())
        except ValueError as exc:
            raise RuntimeError(f"{source_name}='{limit_spec}' is not a valid fraction.") from exc
        if denominator <= 0:
            raise RuntimeError(f"{source_name}='{limit_spec}' must have a positive denominator.")
        ratio = numerator / denominator
        if not (0.0 < ratio <= 1.0):
            raise RuntimeError(f"{source_name}='{limit_spec}' must resolve to a ratio between 0 and 1.")
        count = math.floor(total_pairs * ratio)
        if count < 1:
            raise RuntimeError(
                f"{source_name}='{limit_spec}' resolves to 0 pairs after flooring. "
                f"Please choose a larger ratio."
            )
        return count, {
            "input": limit_spec,
            "mode": "fraction",
            "ratio": ratio,
            "description": f"Using ratio {limit_spec} of {total_pairs} pairs -> {count} pairs after flooring.",
        }

    try:
        count = int(spec)
    except ValueError:
        try:
            numeric_value = float(spec)
        except ValueError as exc:
            raise RuntimeError(
                f"{source_name}='{limit_spec}' is invalid. Use an integer count, "
                "a ratio like 0.1, a percentage like 10%, or a fraction like 1/3."
            ) from exc

        if 0.0 < numeric_value < 1.0:
            count = math.floor(total_pairs * numeric_value)
            if count < 1:
                raise RuntimeError(
                    f"{source_name}='{limit_spec}' resolves to 0 pairs after flooring. "
                    f"Please choose a larger ratio."
                )
            return count, {
                "input": limit_spec,
                "mode": "ratio",
                "ratio": numeric_value,
                "description": f"Using ratio {limit_spec} of {total_pairs} pairs -> {count} pairs after flooring.",
            }

        if numeric_value >= 1.0 and float(numeric_value).is_integer():
            count = int(numeric_value)
        else:
            raise RuntimeError(
                f"{source_name}='{limit_spec}' is invalid. "
                "Numbers smaller than 1 are treated as ratios; counts must be integers."
            )

    if count < 1:
        raise RuntimeError(f"{source_name} must be >= 1.")
    if count > total_pairs:
        raise RuntimeError(
            f"Requested {source_name}={count}, but only {total_pairs} non-empty pairs exist."
        )
    return count, {
        "input": limit_spec,
        "mode": "count",
        "description": f"Exporting {count} pairs.",
    }


def resolve_limit_pairs(total_pairs: int, table_name: str, limit_pairs: str | None, no_prompt: bool):
    if total_pairs <= 0:
        return 0, {"mode": "empty", "description": "No pairs available."}

    print(f"Found {total_pairs} non-empty pairs in {table_name}.")

    if limit_pairs is not None:
        count, selection_info = parse_limit_spec(limit_pairs, total_pairs, "--limit_pairs")
        print(selection_info["description"])
        return count, selection_info

    if no_prompt:
        return total_pairs, {
            "input": "",
            "mode": "all",
            "description": f"Exporting all {total_pairs} pairs because --no_prompt was enabled.",
        }

    while True:
        try:
            raw = input(
                f"How many pairs do you want to export? "
                f"[count 1-{total_pairs}, ratio like 0.1 or 10% or 1/3, Enter=all]: "
            ).strip()
        except EOFError as exc:
            raise RuntimeError(
                "Interactive input is unavailable in the current shell. "
                "Please re-run with --limit_pairs=<N|ratio> to choose a count, "
                "or add --no_prompt to export all pairs."
            ) from exc
        except KeyboardInterrupt as exc:
            raise RuntimeError("Cancelled while waiting for pair count input.") from exc
        if raw == "":
            return total_pairs, {
                "input": "",
                "mode": "all",
                "description": f"Exporting all {total_pairs} pairs.",
            }
        try:
            count, selection_info = parse_limit_spec(raw, total_pairs, "input")
        except RuntimeError as exc:
            print(exc)
            continue
        print(selection_info["description"])
        return count, selection_info


def load_grayscale(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


class LRUCache:
    def __init__(self, max_items: int):
        self.max_items = max(0, int(max_items))
        self._items = OrderedDict()

    def get(self, key):
        if self.max_items <= 0:
            return None
        if key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key, value):
        if self.max_items <= 0:
            return
        if key in self._items:
            self._items.pop(key)
        self._items[key] = value
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


class RenderAssetCache:
    def __init__(self, image_cache_size: int, keypoint_cache_size: int):
        self.image_cache = LRUCache(image_cache_size)
        self.keypoint_cache = LRUCache(keypoint_cache_size)

    def get_keypoints(self, con, image_id: int):
        cached = self.keypoint_cache.get(image_id)
        if cached is not None:
            return cached
        keypoints = load_keypoints(con, image_id)
        self.keypoint_cache.put(image_id, keypoints)
        return keypoints

    def get_resized_grayscale(self, path: Path, max_side: int | None):
        cache_key = (str(path), max_side)
        cached = self.image_cache.get(cache_key)
        if cached is not None:
            return cached
        resized = resize_with_scale(load_grayscale(path), max_side=max_side)
        self.image_cache.put(cache_key, resized)
        return resized


def resize_with_scale(image, max_side: int | None):
    if not max_side:
        return image, 1.0
    height, width = image.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale == 1.0:
        return image, 1.0
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def make_match_color(index: int, total: int):
    if total <= 1:
        hue = 60
    else:
        hue = int(round((179 * index) / max(total - 1, 1)))
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_title_band(canvas, text: str, x: int, y: int, width: int, title_height: int):
    cv2.rectangle(canvas, (x, y), (x + width, y + title_height), (255, 255, 255), thickness=-1)
    text_origin = (x + 8, y + title_height - 8)
    cv2.putText(canvas, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)


def create_canvas(image_a, image_b, layout: str, image_gap: int, title_a: str | None = None, title_b: str | None = None):
    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]
    title_height = 28
    band_a = title_height if title_a else 0
    band_b = title_height if title_b else 0

    if layout == "vertical":
        canvas_height = band_a + height_a + image_gap + band_b + height_b
        canvas_width = max(width_a, width_b)
        offset_a = ((canvas_width - width_a) // 2, band_a)
        offset_b = ((canvas_width - width_b) // 2, band_a + height_a + image_gap + band_b)
        title_pos_a = (0, 0, canvas_width)
        title_pos_b = (0, band_a + height_a + image_gap, canvas_width)
    elif layout == "horizontal":
        canvas_height = max(band_a + height_a, band_b + height_b)
        canvas_width = width_a + image_gap + width_b
        offset_a = (0, band_a)
        offset_b = (width_a + image_gap, band_b)
        title_pos_a = (0, 0, width_a)
        title_pos_b = (width_a + image_gap, 0, width_b)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    if title_a:
        draw_title_band(canvas, title_a, title_pos_a[0], title_pos_a[1], title_pos_a[2], title_height)
    if title_b:
        draw_title_band(canvas, title_b, title_pos_b[0], title_pos_b[1], title_pos_b[2], title_height)
    canvas[offset_a[1]:offset_a[1] + height_a, offset_a[0]:offset_a[0] + width_a] = cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR)
    canvas[offset_b[1]:offset_b[1] + height_b, offset_b[0]:offset_b[0] + width_b] = cv2.cvtColor(image_b, cv2.COLOR_GRAY2BGR)
    return canvas, offset_a, offset_b


def draw_match_overlay(canvas, points_a, points_b, offset_a, offset_b, draw_labels: bool, line_thickness: int, point_radius: int):
    total = len(points_a)
    for idx, (point_a, point_b) in enumerate(zip(points_a[:, :2], points_b[:, :2])):
        line_color = make_match_color(idx, total)
        point_color = (255, 0, 0)
        pt_a = (int(round(point_a[0])) + offset_a[0], int(round(point_a[1])) + offset_a[1])
        pt_b = (int(round(point_b[0])) + offset_b[0], int(round(point_b[1])) + offset_b[1])
        cv2.line(canvas, pt_a, pt_b, line_color, line_thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt_a, point_radius, point_color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt_b, point_radius, point_color, 1, lineType=cv2.LINE_AA)
        if draw_labels:
            label = str(idx + 1)
            for point in (pt_a, pt_b):
                text_origin = (point[0] + 6, point[1] - 6)
                cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, line_color, 1, cv2.LINE_AA)
    return canvas


def select_match_subset(matches, points_a, points_b, max_draw_matches: int, strategy: str):
    if len(matches) <= max_draw_matches:
        return matches
    if strategy == "first":
        return matches[:max_draw_matches]
    if strategy != "spread":
        raise ValueError(f"Unsupported selection strategy: {strategy}")

    def _normalize(points):
        points = points.astype(np.float32)
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        scale = np.maximum(max_xy - min_xy, 1.0)
        return (points - min_xy) / scale

    feat_a = _normalize(points_a[:, :2])
    feat_b = _normalize(points_b[:, :2])
    features = np.concatenate([feat_a, feat_b], axis=1)

    selected = [0]
    min_dists = np.linalg.norm(features - features[0], axis=1)
    min_dists[0] = -1.0

    while len(selected) < max_draw_matches:
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] < 0:
            break
        selected.append(next_idx)
        candidate_dists = np.linalg.norm(features - features[next_idx], axis=1)
        min_dists = np.minimum(min_dists, candidate_dists)
        min_dists[selected] = -1.0

    return matches[np.array(selected, dtype=np.int64)]


def sanitize_name_for_filename(name: str):
    stem = Path(name).stem
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return sanitized or "image"


def build_output_filename(index: int, pair_summary: dict):
    name1 = sanitize_name_for_filename(pair_summary["image_name1"])
    name2 = sanitize_name_for_filename(pair_summary["image_name2"])
    return f"{index:06d}__pair_{pair_summary['pair_id']}__{name1}__{name2}.png"


def build_image_title(slot_label: str, image_id: int, image_name: str):
    return f"{slot_label}: id={image_id}  file={Path(image_name).name}"


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def draw_pair(
    asset_cache: RenderAssetCache,
    image_root: Path,
    image_id_a: int,
    image_id_b: int,
    image_name_a: str,
    image_name_b: str,
    keypoints_a,
    keypoints_b,
    matches,
    output_path: Path,
    max_draw_matches: int,
    max_side: int | None,
    layout: str,
    image_gap: int,
    selection_strategy: str,
    draw_labels: bool,
    line_thickness: int,
    point_radius: int,
    show_pair_titles: bool,
):
    image_a_full = image_root / image_name_a
    image_b_full = image_root / image_name_b
    image_a, scale_a = asset_cache.get_resized_grayscale(image_a_full, max_side=max_side)
    image_b, scale_b = asset_cache.get_resized_grayscale(image_b_full, max_side=max_side)

    candidate_points_a = keypoints_a[matches[:, 0]]
    candidate_points_b = keypoints_b[matches[:, 1]]
    draw_matches = select_match_subset(matches, candidate_points_a, candidate_points_b, max_draw_matches, selection_strategy)
    points_a = keypoints_a[draw_matches[:, 0]].copy()
    points_b = keypoints_b[draw_matches[:, 1]].copy()
    points_a[:, 0] *= scale_a
    points_a[:, 1] *= scale_a
    points_b[:, 0] *= scale_b
    points_b[:, 1] *= scale_b

    title_a = build_image_title("Photo A", image_id_a, image_name_a) if show_pair_titles else None
    title_b = build_image_title("Photo B", image_id_b, image_name_b) if show_pair_titles else None
    canvas, offset_a, offset_b = create_canvas(
        image_a,
        image_b,
        layout=layout,
        image_gap=image_gap,
        title_a=title_a,
        title_b=title_b,
    )
    canvas = draw_match_overlay(
        canvas,
        points_a,
        points_b,
        offset_a,
        offset_b,
        draw_labels=draw_labels,
        line_thickness=line_thickness,
        point_radius=point_radius,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return image_a_full, image_b_full, len(draw_matches)


def summarize_pair(image_map, pair_id: int, matches_row_count_map: dict, two_view_row_count_map: dict):
    image_id1, image_id2 = pair_id_to_image_ids(pair_id)
    name1 = image_map[image_id1]
    name2 = image_map[image_id2]
    return {
        "pair_id": int(pair_id),
        "image_id1": image_id1,
        "image_id2": image_id2,
        "image_name1": name1,
        "image_name2": name2,
        "matches_rows": int(matches_row_count_map.get(int(pair_id), 0)),
        "two_view_geometries_rows": int(two_view_row_count_map.get(int(pair_id), 0)),
    }


def render_pair_visualization(
    con,
    image_map,
    pair_summary: dict,
    asset_cache: RenderAssetCache,
    pair_id: int,
    database_path: Path,
    image_root: Path,
    output_path: Path,
    table_name: str,
    max_draw_matches: int,
    max_side: int | None,
    layout: str,
    image_gap: int,
    selection_strategy: str,
    draw_labels: bool,
    line_thickness: int,
    point_radius: int,
    show_pair_titles: bool,
    selection_mode: str,
    top_rows: list,
    write_metadata: bool,
    metadata_output: Path | None = None,
    metadata_overrides: dict | None = None,
):
    summary = pair_summary
    image_id1 = summary["image_id1"]
    image_id2 = summary["image_id2"]
    image_name1 = summary["image_name1"]
    image_name2 = summary["image_name2"]
    keypoints1 = asset_cache.get_keypoints(con, image_id1)
    keypoints2 = asset_cache.get_keypoints(con, image_id2)
    matches = load_matches_for_pair(con, table_name, pair_id)
    image_a_full, image_b_full, drawn_count = draw_pair(
        asset_cache=asset_cache,
        image_root=image_root,
        image_id_a=image_id1,
        image_id_b=image_id2,
        image_name_a=image_name1,
        image_name_b=image_name2,
        keypoints_a=keypoints1,
        keypoints_b=keypoints2,
        matches=matches,
        output_path=output_path,
        max_draw_matches=max_draw_matches,
        max_side=max_side,
        layout=layout,
        image_gap=image_gap,
        selection_strategy=selection_strategy,
        draw_labels=draw_labels,
        line_thickness=line_thickness,
        point_radius=point_radius,
        show_pair_titles=show_pair_titles,
    )

    metadata = {
        "database_path": str(database_path),
        "image_root": str(image_root),
        "visualized_table": table_name,
        "selection_mode": selection_mode,
        "selected_pair": summary,
        "num_drawn_matches": drawn_count,
        "output_image": str(output_path),
        "source_image1": str(image_a_full),
        "source_image2": str(image_b_full),
        "layout": layout,
        "image_gap": image_gap,
        "selection_strategy": selection_strategy,
        "label_matches": draw_labels,
        "line_thickness": line_thickness,
        "point_radius": point_radius,
        "show_pair_titles": show_pair_titles,
        "top_pairs_by_table_rows": top_rows,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)

    if write_metadata:
        if metadata_output is None:
            metadata_output = output_path.with_suffix(output_path.suffix + ".json")
        write_json(metadata_output, metadata)
    else:
        metadata_output = None
    return metadata, metadata_output


def main():
    parser = argparse.ArgumentParser(description="Inspect and visualize actual COLMAP database matches.")
    parser.add_argument("--database_path", type=Path, required=True, help="Path to COLMAP database.db")
    parser.add_argument("--image_root", type=Path, required=True, help="Folder containing the source images used by COLMAP")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path for single-pair mode")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory for batch all-pairs mode")
    parser.add_argument("--summary_output", type=Path, default=None, help="Optional summary JSON path for batch mode")
    parser.add_argument("--metadata_output", type=Path, default=None, help="Optional output JSON path for single-pair mode")
    parser.add_argument(
        "--table",
        choices=["matches", "two_view_geometries"],
        default="two_view_geometries",
        help="Which COLMAP table to visualize",
    )
    parser.add_argument("--all_pairs", action="store_true", help="Render every pair stored in the selected COLMAP table")
    parser.add_argument("--pair_id", type=int, default=None, help="Specific COLMAP pair_id to visualize")
    parser.add_argument("--image_name1", type=str, default=None, help="First image name from COLMAP images table")
    parser.add_argument("--image_name2", type=str, default=None, help="Second image name from COLMAP images table")
    parser.add_argument("--rank", type=int, default=1, help="1-based rank when auto-selecting by descending row count")
    parser.add_argument(
        "--limit_pairs",
        type=str,
        default=None,
        help="Optional batch limit. Supports counts like 20, ratios like 0.1, percentages like 10%, or fractions like 1/3",
    )
    parser.add_argument("--count_only", action="store_true", help="Only report how many non-empty pairs are available, then exit")
    parser.add_argument("--no_prompt", action="store_true", help="Do not prompt for pair count in batch mode; export all when --limit_pairs is omitted")
    parser.add_argument("--show_pbar", action="store_true", help="Show a progress bar while rendering batch outputs")
    parser.add_argument("--max_draw_matches", type=int, default=100, help="Maximum number of match lines to draw")
    parser.add_argument("--max_side", type=int, default=1280, help="Resize image long side for output visualization")
    parser.add_argument("--layout", choices=["vertical", "horizontal"], default="vertical", help="How to arrange the two images")
    parser.add_argument("--image_gap", type=int, default=40, help="Gap in pixels between the two arranged images")
    parser.add_argument("--selection_strategy", choices=["spread", "first"], default="spread", help="How to pick the displayed subset when matches are dense")
    parser.add_argument("--label_matches", action="store_true", help="Draw index labels next to endpoints")
    parser.add_argument("--show_pair_titles", action="store_true", help="Show image ids and filenames in the figure")
    parser.add_argument("--line_thickness", type=int, default=1, help="Line thickness for match connections")
    parser.add_argument("--point_radius", type=int, default=3, help="Radius of endpoint markers")
    parser.add_argument(
        "--skip_pair_metadata",
        action="store_true",
        help="Skip per-pair sidecar JSON files in batch mode. The batch summary JSON is still written.",
    )
    parser.add_argument(
        "--image_cache_size",
        type=int,
        default=64,
        help="Maximum number of resized grayscale images to cache in memory. Use 0 to disable image caching.",
    )
    parser.add_argument(
        "--keypoint_cache_size",
        type=int,
        default=256,
        help="Maximum number of keypoint arrays to cache in memory. Use 0 to disable keypoint caching.",
    )
    args = parser.parse_args()

    con = sqlite3.connect(args.database_path)
    image_map = load_image_map(con)
    name_to_id_map = load_name_to_id_map(image_map)
    pair_rows = load_pair_rows(con, args.table)
    num_pairs_available = len(pair_rows)
    matches_row_count_map = load_pair_row_count_map(con, "matches")
    two_view_row_count_map = load_pair_row_count_map(con, "two_view_geometries")
    top_rows = [
        summarize_pair(
            image_map,
            int(row[0]),
            matches_row_count_map=matches_row_count_map,
            two_view_row_count_map=two_view_row_count_map,
        )
        for row in pair_rows[:10]
    ]
    asset_cache = RenderAssetCache(
        image_cache_size=args.image_cache_size,
        keypoint_cache_size=args.keypoint_cache_size,
    )

    if args.count_only:
        con.close()
        print(json.dumps({
            "mode": "count_only",
            "database_path": str(args.database_path),
            "visualized_table": args.table,
            "num_pairs_available": num_pairs_available,
        }, indent=2))
        return

    if args.all_pairs:
        if args.output_dir is None:
            raise RuntimeError("Please provide --output_dir when using --all_pairs.")
        if args.output is not None or args.metadata_output is not None:
            raise RuntimeError("Do not combine --all_pairs with --output or --metadata_output.")
        if args.pair_id is not None or args.image_name1 is not None or args.image_name2 is not None:
            raise RuntimeError("Do not combine --all_pairs with --pair_id or --image_name1/--image_name2.")
        if not pair_rows:
            raise RuntimeError(f"No non-empty rows found in {args.table}")

        selected_limit, selection_info = resolve_limit_pairs(
            total_pairs=num_pairs_available,
            table_name=args.table,
            limit_pairs=args.limit_pairs,
            no_prompt=args.no_prompt,
        )
        selected_rows = pair_rows[:selected_limit]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        batch_items = []
        row_iterator = selected_rows
        if args.show_pbar:
            row_iterator = tqdm(selected_rows, total=len(selected_rows), desc=f"Rendering {args.table}", leave=True)

        for batch_rank, row in enumerate(row_iterator, start=1):
            pair_id = int(row[0])
            pair_summary = summarize_pair(
                image_map,
                pair_id,
                matches_row_count_map=matches_row_count_map,
                two_view_row_count_map=two_view_row_count_map,
            )
            output_path = args.output_dir / build_output_filename(batch_rank, pair_summary)
            metadata, metadata_path = render_pair_visualization(
                con=con,
                image_map=image_map,
                pair_summary=pair_summary,
                asset_cache=asset_cache,
                pair_id=pair_id,
                database_path=args.database_path,
                image_root=args.image_root,
                output_path=output_path,
                table_name=args.table,
                max_draw_matches=args.max_draw_matches,
                max_side=args.max_side,
                layout=args.layout,
                image_gap=args.image_gap,
                selection_strategy=args.selection_strategy,
                draw_labels=args.label_matches,
                line_thickness=args.line_thickness,
                point_radius=args.point_radius,
                show_pair_titles=True,
                selection_mode="all_pairs",
                top_rows=top_rows,
                write_metadata=not args.skip_pair_metadata,
                metadata_overrides={"batch_rank": batch_rank},
            )
            batch_items.append({
                "batch_rank": batch_rank,
                "pair_id": metadata["selected_pair"]["pair_id"],
                "image_id1": metadata["selected_pair"]["image_id1"],
                "image_id2": metadata["selected_pair"]["image_id2"],
                "image_name1": metadata["selected_pair"]["image_name1"],
                "image_name2": metadata["selected_pair"]["image_name2"],
                "matches_rows": metadata["selected_pair"]["matches_rows"],
                "two_view_geometries_rows": metadata["selected_pair"]["two_view_geometries_rows"],
                "output_image": metadata["output_image"],
                "output_metadata": str(metadata_path) if metadata_path is not None else None,
            })

        summary_output = args.summary_output or (args.output_dir / f"{args.table}_all_pairs_summary.json")
        batch_summary = {
            "database_path": str(args.database_path),
            "image_root": str(args.image_root),
            "visualized_table": args.table,
            "num_pairs_available": num_pairs_available,
            "selection_input": selection_info.get("input"),
            "selection_mode": selection_info.get("mode"),
            "selection_ratio": selection_info.get("ratio"),
            "selection_description": selection_info.get("description"),
            "num_pairs_selected": len(selected_rows),
            "num_pairs_processed": len(batch_items),
            "layout": args.layout,
            "image_gap": args.image_gap,
            "selection_strategy": args.selection_strategy,
            "label_matches": args.label_matches,
            "show_pair_titles": True,
            "show_pbar": args.show_pbar,
            "line_thickness": args.line_thickness,
            "point_radius": args.point_radius,
            "skip_pair_metadata": args.skip_pair_metadata,
            "image_cache_size": args.image_cache_size,
            "keypoint_cache_size": args.keypoint_cache_size,
            "output_dir": str(args.output_dir),
            "pairs": batch_items,
        }
        write_json(summary_output, batch_summary)
        con.close()
        print(json.dumps({
            "mode": "all_pairs",
            "summary_output": str(summary_output),
            "num_pairs_available": num_pairs_available,
            "selection_input": selection_info.get("input"),
            "selection_mode": selection_info.get("mode"),
            "selection_ratio": selection_info.get("ratio"),
            "num_pairs_selected": len(selected_rows),
            "num_pairs_processed": len(batch_items),
            "show_pbar": args.show_pbar,
            "skip_pair_metadata": args.skip_pair_metadata,
            "first_output": batch_items[0]["output_image"] if batch_items else None,
        }, indent=2))
        return

    if (args.image_name1 is None) != (args.image_name2 is None):
        raise RuntimeError("Please provide both --image_name1 and --image_name2 together.")
    if args.output is None:
        raise RuntimeError("Please provide --output in single-pair mode.")

    selection_mode = "pair_id"
    metadata_overrides = {}
    if args.image_name1 is not None:
        image_id1_req, image_name1_req = resolve_image_name(args.image_name1, name_to_id_map)
        image_id2_req, image_name2_req = resolve_image_name(args.image_name2, name_to_id_map)
        pair_id = image_ids_to_pair_id(image_id1_req, image_id2_req)
        selection_mode = "image_names"
        metadata_overrides.update({
            "requested_image_name1": args.image_name1,
            "requested_image_name2": args.image_name2,
            "resolved_image_name1": image_name1_req,
            "resolved_image_name2": image_name2_req,
        })
    elif args.pair_id is None:
        if not pair_rows:
            raise RuntimeError(f"No rows found in {args.table}")
        rank_index = max(0, args.rank - 1)
        if rank_index >= len(pair_rows):
            raise RuntimeError(f"Requested rank {args.rank} but only {len(pair_rows)} rows exist in {args.table}")
        pair_id = int(pair_rows[rank_index][0])
        selection_mode = "rank"
    else:
        pair_id = args.pair_id

    pair_summary = summarize_pair(
        image_map,
        pair_id,
        matches_row_count_map=matches_row_count_map,
        two_view_row_count_map=two_view_row_count_map,
    )
    metadata, _ = render_pair_visualization(
        con=con,
        image_map=image_map,
        pair_summary=pair_summary,
        asset_cache=asset_cache,
        pair_id=pair_id,
        database_path=args.database_path,
        image_root=args.image_root,
        output_path=args.output,
        table_name=args.table,
        max_draw_matches=args.max_draw_matches,
        max_side=args.max_side,
        layout=args.layout,
        image_gap=args.image_gap,
        selection_strategy=args.selection_strategy,
        draw_labels=args.label_matches,
        line_thickness=args.line_thickness,
        point_radius=args.point_radius,
        show_pair_titles=args.show_pair_titles,
        selection_mode=selection_mode,
        top_rows=top_rows,
        write_metadata=True,
        metadata_output=args.metadata_output,
        metadata_overrides=metadata_overrides,
    )
    con.close()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
