"""Build a lightweight interactive HTML fragment for depth-map preview."""

import argparse
import json
from pathlib import Path


def load_summary(path):
    if not path:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_fragment(preview, summary):
    depth = summary.get("depth", {})
    plane = summary.get("plane", {})
    data_json = json.dumps(preview, separators=(",", ":"))
    summary_json = json.dumps(
        {
            "max_depth": depth.get("max_depth"),
            "bottom_mean_depth": depth.get("bottom_mean_depth"),
            "bottom_percentile_depth": depth.get("bottom_percentile_depth"),
            "inlier_ratio": plane.get("inlier_ratio"),
        },
        separators=(",", ":"),
    )
    return f"""<div id="pothole-depth-map-vis" class="pothole-depth-map-vis">
  <div class="viz-grid">
    <div class="card viz-stat">
      <span class="text-muted">Max depth</span>
      <span class="viz-stat-value" data-role="max-depth"></span>
      <span class="text-small text-muted">model units</span>
    </div>
    <div class="card viz-stat">
      <span class="text-muted">Deepest 1% mean</span>
      <span class="viz-stat-value" data-role="mean-depth"></span>
      <span class="text-small text-muted">model units</span>
    </div>
    <div class="card viz-stat">
      <span class="text-muted">Preview points</span>
      <span class="viz-stat-value" data-role="point-count"></span>
      <span class="text-small text-muted">sampled from mesh</span>
    </div>
  </div>

  <div class="viz-controls" aria-label="View controls">
    <button type="button" class="btn" data-view="oblique">Oblique</button>
    <button type="button" class="btn" data-view="top">Top</button>
    <button type="button" class="btn" data-view="side">Side</button>
    <button type="button" class="btn btn-ghost" data-view="reset">Reset</button>
    <label class="form-label">Point size
      <input class="form-range" type="range" min="1" max="5" step="1" value="2" data-role="point-size">
    </label>
  </div>

  <div class="depth-map-stage">
    <canvas data-role="canvas" role="img" aria-label="3D point cloud colored by pothole depth"></canvas>
    <div class="text-small depth-readout" data-role="readout">Depth map</div>
  </div>

  <div class="depth-legend" aria-hidden="true">
    <span>0</span>
    <div class="depth-legend-bar"></div>
    <span data-role="color-max"></span>
  </div>
</div>

<style>
#pothole-depth-map-vis {{
  display: grid;
  gap: 0.75rem;
  color: var(--foreground);
}}
#pothole-depth-map-vis .depth-map-stage {{
  position: relative;
  min-height: 320px;
}}
#pothole-depth-map-vis canvas {{
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
  min-height: 320px;
  touch-action: none;
}}
#pothole-depth-map-vis .depth-readout {{
  position: absolute;
  left: 0.5rem;
  bottom: 0.5rem;
  color: var(--foreground);
  background: color-mix(in srgb, var(--background) 78%, transparent);
  border: 1px solid var(--border);
  padding: 0.25rem 0.5rem;
}}
#pothole-depth-map-vis .depth-legend {{
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 0.5rem;
  color: var(--muted-foreground);
}}
#pothole-depth-map-vis .depth-legend-bar {{
  height: 0.65rem;
  background: linear-gradient(90deg, rgb(48,84,150), rgb(44,178,214), rgb(67,190,112), rgb(245,206,66), rgb(218,62,55));
}}
@media (max-width: 520px) {{
  #pothole-depth-map-vis .depth-map-stage,
  #pothole-depth-map-vis canvas {{
    min-height: 260px;
  }}
}}
</style>

<script>
(() => {{
  const root = document.getElementById("pothole-depth-map-vis");
  const preview = {data_json};
  const summary = {summary_json};
  const canvas = root.querySelector('[data-role="canvas"]');
  const ctx = canvas.getContext("2d");
  const readout = root.querySelector('[data-role="readout"]');
  const sizeInput = root.querySelector('[data-role="point-size"]');
  const points = preview.points;
  const colors = points.map((p) => `rgb(${{p[4]}},${{p[5]}},${{p[6]}})`);
  const bounds = preview.bounds;
  const center = [
    (bounds.min[0] + bounds.max[0]) / 2,
    (bounds.min[1] + bounds.max[1]) / 2,
    (bounds.min[2] + bounds.max[2]) / 2
  ];
  const radius = Math.max(1e-6, Math.hypot(bounds.size[0], bounds.size[1], bounds.size[2]) / 2);
  const projected = new Array(points.length);
  let yaw = -0.72;
  let pitch = -0.58;
  let zoom = 1.0;
  let pointSize = Number(sizeInput.value);
  let pointerDown = false;
  let lastX = 0;
  let lastY = 0;

  const fmt = (value) => Number.isFinite(value) ? value.toFixed(4) : "n/a";
  root.querySelector('[data-role="max-depth"]').textContent = fmt(summary.max_depth);
  root.querySelector('[data-role="mean-depth"]').textContent = fmt(summary.bottom_mean_depth);
  root.querySelector('[data-role="point-count"]').textContent = preview.point_count.toLocaleString();
  root.querySelector('[data-role="color-max"]').textContent = fmt(preview.color_max_depth);

  function resizeCanvas() {{
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(320, Math.floor(rect.width));
    const height = Math.max(260, Math.floor(rect.height));
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    render();
  }}

  function setPressed(name) {{
    root.querySelectorAll("[data-view]").forEach((button) => {{
      button.setAttribute("aria-pressed", String(button.dataset.view === name));
    }});
  }}

  function render() {{
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);
    const cy = Math.cos(yaw);
    const sy = Math.sin(yaw);
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);
    const scale = Math.min(width, height) * 0.82 * zoom / (radius * 2);
    const cx = width / 2;
    const midY = height / 2;
    const order = [];

    for (let i = 0; i < points.length; i += 1) {{
      const p = points[i];
      const x = p[0] - center[0];
      const y = p[1] - center[1];
      const z = p[2] - center[2];
      const rx = x * cy + z * sy;
      const rz = -x * sy + z * cy;
      const ry = y * cp - rz * sp;
      const depthZ = y * sp + rz * cp;
      const sx = cx + rx * scale;
      const sy2 = midY - ry * scale;
      projected[i] = [sx, sy2, depthZ];
      order.push(i);
    }}

    order.sort((a, b) => projected[a][2] - projected[b][2]);
    ctx.globalAlpha = 0.86;
    for (const i of order) {{
      const [sx, sy2] = projected[i];
      if (sx < -8 || sx > width + 8 || sy2 < -8 || sy2 > height + 8) continue;
      ctx.fillStyle = colors[i];
      ctx.beginPath();
      ctx.arc(sx, sy2, pointSize, 0, Math.PI * 2);
      ctx.fill();
    }}
    ctx.globalAlpha = 1;
    drawScale(width, height);
  }}

  function drawScale(width, height) {{
    const maxBar = Math.max(0.001, preview.color_max_depth);
    const barWidth = Math.min(180, width * 0.32);
    const x = width - barWidth - 12;
    const y = height - 20;
    ctx.strokeStyle = getComputedStyle(root).getPropertyValue("--muted-foreground").trim() || "gray";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + barWidth, y);
    ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(`${{fmt(maxBar)}} units`, x, y - 6);
  }}

  function nearestPoint(x, y) {{
    let best = -1;
    let bestDistance = 64;
    for (let i = 0; i < projected.length; i += 1) {{
      const p = projected[i];
      const dx = p[0] - x;
      const dy = p[1] - y;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {{
        bestDistance = distance;
        best = i;
      }}
    }}
    return best;
  }}

  canvas.addEventListener("pointerdown", (event) => {{
    pointerDown = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  }});
  canvas.addEventListener("pointerup", (event) => {{
    pointerDown = false;
    canvas.releasePointerCapture(event.pointerId);
  }});
  canvas.addEventListener("pointerleave", () => {{
    pointerDown = false;
  }});
  canvas.addEventListener("pointermove", (event) => {{
    const rect = canvas.getBoundingClientRect();
    if (pointerDown) {{
      yaw += (event.clientX - lastX) * 0.008;
      pitch = Math.max(-1.45, Math.min(1.45, pitch + (event.clientY - lastY) * 0.008));
      lastX = event.clientX;
      lastY = event.clientY;
      setPressed("");
      render();
    }} else {{
      const selected = nearestPoint(event.clientX - rect.left, event.clientY - rect.top);
      if (selected >= 0) {{
        const p = points[selected];
        readout.textContent = `depth ${{fmt(p[3])}} at (${{fmt(p[0])}}, ${{fmt(p[1])}}, ${{fmt(p[2])}})`;
      }}
    }}
  }});
  canvas.addEventListener("wheel", (event) => {{
    event.preventDefault();
    zoom = Math.max(0.55, Math.min(2.8, zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
    render();
  }}, {{ passive: false }});

  root.querySelectorAll("[data-view]").forEach((button) => {{
    button.addEventListener("click", () => {{
      const view = button.dataset.view;
      if (view === "top") {{
        yaw = 0;
        pitch = -1.35;
        zoom = 1.0;
      }} else if (view === "side") {{
        yaw = -1.55;
        pitch = -0.05;
        zoom = 1.0;
      }} else {{
        yaw = -0.72;
        pitch = -0.58;
        zoom = 1.0;
      }}
      setPressed(view === "reset" ? "oblique" : view);
      render();
    }});
  }});
  sizeInput.addEventListener("input", () => {{
    pointSize = Number(sizeInput.value);
    render();
  }});

  if ("ResizeObserver" in window) {{
    new ResizeObserver(resizeCanvas).observe(canvas);
  }} else {{
    window.addEventListener("resize", resizeCanvas);
  }}
  setPressed("oblique");
  resizeCanvas();
}})();
</script>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-json", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    preview = json.loads(args.preview_json.read_text(encoding="utf-8"))
    summary = load_summary(args.summary_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_fragment(preview, summary), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
