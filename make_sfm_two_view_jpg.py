from PIL import Image, ImageDraw, ImageFont
import math
from pathlib import Path


OUT = Path(r"C:\neuralangelo\sfm_two_view_initialization_diagram.jpg")
W, H = 1920, 1080
S = 2


def font(size, bold=False):
    path = r"C:\Windows\Fonts\msjhbd.ttc" if bold else r"C:\Windows\Fonts\msjh.ttc"
    return ImageFont.truetype(path, size * S)


def xy(v):
    return tuple(int(round(x * S)) for x in v)


def box(v):
    return tuple(int(round(x * S)) for x in v)


img = Image.new("RGB", (W * S, H * S), "#f6f8fb")
draw = ImageDraw.Draw(img)

COL = {
    "ink": "#102033",
    "muted": "#53657a",
    "soft": "#e7edf5",
    "line": "#c8d3df",
    "blue": "#2563eb",
    "green": "#16a34a",
    "purple": "#7c3aed",
    "red": "#ef4444",
    "red_dark": "#991b1b",
    "amber": "#f59e0b",
}

title_f = font(48, True)
sub_f = font(25)
label_f = font(29, True)
body_f = font(24)
small_f = font(21)
tiny_f = font(18)
mono_f = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 23 * S)


def text(pos, s, f, fill=COL["ink"], anchor=None):
    draw.text(xy(pos), s, font=f, fill=fill, anchor=anchor)


def rounded(rect, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box(rect), radius=radius * S, fill=fill, outline=outline, width=width * S)


def line(points, fill, width=3, joint="curve"):
    draw.line([xy(p) for p in points], fill=fill, width=width * S, joint=joint)


def dashed_line(p1, p2, fill, width=4, dash=18, gap=12):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    t = 0
    while t < length:
        a = t
        b = min(t + dash, length)
        line([(x1 + ux * a, y1 + uy * a), (x1 + ux * b, y1 + uy * b)], fill, width)
        t += dash + gap


def arrow(p1, p2, fill, width=4, head=18):
    line([p1, p2], fill, width)
    x1, y1 = p1
    x2, y2 = p2
    angle = math.atan2(y2 - y1, x2 - x1)
    pts = []
    for a in (angle + math.pi * 0.82, angle - math.pi * 0.82):
        pts.append((x2 + math.cos(a) * head, y2 + math.sin(a) * head))
    draw.polygon([xy(p2), xy(pts[0]), xy(pts[1])], fill=fill)


def camera(cx, cy, angle_deg, label, note):
    a = math.radians(angle_deg)
    base = [(-34, -25), (42, 0), (-34, 25)]
    pts = []
    for x, y in base:
        xr = x * math.cos(a) - y * math.sin(a)
        yr = x * math.sin(a) + y * math.cos(a)
        pts.append((cx + xr, cy + yr))
    draw.polygon([xy(p) for p in pts], fill=COL["ink"])
    arrow((cx + math.cos(a) * 40, cy + math.sin(a) * 40), (cx + math.cos(a) * 118, cy + math.sin(a) * 118), "#334155", 4, 14)
    text((cx - 78, cy + 68), label, label_f)
    text((cx - 78, cy + 101), note, tiny_f, COL["muted"])


def image_plane(rect, fill, stroke, label, kp, kp_label, accent):
    x0, y0, x1, y1 = rect
    rounded(rect, 10, fill, stroke, 4)
    text(((x0 + x1) / 2, y0 - 35), label, label_f, anchor="mm")
    line([(x0 + 35, y0 + 55), (x1 - 30, y1 - 76)], accent, 5)
    line([(x0 + 52, y1 - 65), (x1 - 50, y0 + 72)], stroke, 4)
    line([(x0 + 80, y0 + 145), (x1 - 54, y0 + 174)], "#94a3b8", 3)
    draw.ellipse(box((kp[0] - 13, kp[1] - 13, kp[0] + 13, kp[1] + 13)), fill=stroke)
    text((kp[0] + 24, kp[1] - 27), kp_label, tiny_f, COL["ink"])


def card(rect, idx, title, body, color):
    rounded(rect, 14, "#ffffff", COL["line"], 2)
    x0, y0, x1, y1 = rect
    draw.ellipse(box((x0 + 22, y0 + 24, x0 + 62, y0 + 64)), fill=color)
    text((x0 + 42, y0 + 43), str(idx), font(20, True), "#ffffff", anchor="mm")
    text((x0 + 78, y0 + 28), title, label_f, COL["ink"])
    text((x0 + 78, y0 + 68), body, small_f, COL["muted"])


# Header
text((W / 2, 58), "SfM 初始化兩張影像與三角化", title_f, COL["ink"], anchor="mm")
text((W / 2, 105), "從 inlier matches 估計相對相機姿態，再由兩條視線交會推回第一批 3D 點", sub_f, COL["muted"], anchor="mm")

# Step cards
card((70, 145, 610, 250), 1, "選一對品質好的影像", "兩張圖需要有足夠且穩定的 inlier matches。", COL["blue"])
card((690, 145, 1230, 250), 2, "估計相對姿態 R, t", "由 2D 對應點推估相機 2 的旋轉與平移。", COL["purple"])
card((1310, 145, 1850, 250), 3, "三角化 3D 點", "同一個 3D 點在兩張圖上的投影，反推回空間位置。", COL["green"])
arrow((620, 198), (675, 198), COL["muted"], 3, 12)
arrow((1240, 198), (1295, 198), COL["muted"], 3, 12)

# Main panel
rounded((70, 295, 1850, 945), 18, "#ffffff", "#ccd7e4", 2)
rounded((115, 330, 470, 760), 12, "#e0f2fe", "#0284c7", 4)
image_plane((145, 370, 440, 725), "#e0f2fe", "#0284c7", "Image 1", (335, 505), "keypoint A", "#38bdf8")

rounded((1450, 330, 1805, 760), 12, "#dcfce7", "#16a34a", 4)
image_plane((1480, 370, 1775, 725), "#dcfce7", "#16a34a", "Image 2", (1570, 535), "keypoint B", "#86efac")

# Relative pose arc
text((960, 355), "相對姿態估計", label_f, COL["ink"], anchor="mm")
text((960, 392), "從 inlier matches 求出 R 與 t", body_f, COL["muted"], anchor="mm")
for offset, width in [(0, 5), (10, 2)]:
    pts = []
    for t in range(0, 101):
        u = t / 100
        x = 505 + u * 890
        y = 560 - 130 * math.sin(math.pi * u) + offset
        pts.append((x, y))
    line(pts, COL["purple"], width)
arrow((1375, 555), (1408, 572), COL["purple"], 5, 18)
text((960, 515), "Camera 2 相對 Camera 1", small_f, COL["purple"], anchor="mm")
text((960, 548), "移動多少？旋轉多少？", small_f, COL["purple"], anchor="mm")

# Cameras and rays
cam1 = (460, 815)
cam2 = (1460, 815)
p3d = (960, 640)
camera(*cam1, -4, "Camera 1", "初始化參考座標")
camera(*cam2, -170, "Camera 2", "由 R, t 放到相對位置")

dashed_line(cam1, p3d, COL["blue"], 5, 22, 14)
dashed_line(cam2, p3d, COL["green"], 5, 22, 14)
text((620, 755), "從 Camera 1 拉出視線", small_f, COL["blue"], anchor="mm")
text((1305, 755), "從 Camera 2 拉出視線", small_f, COL["green"], anchor="mm")

# Projection hint lines
dashed_line((335, 505), (cam1[0], cam1[1] - 45), "#60a5fa", 3, 14, 12)
dashed_line((1570, 535), (cam2[0], cam2[1] - 45), "#4ade80", 3, 14, 12)

# 3D point
draw.ellipse(box((p3d[0] - 22, p3d[1] - 22, p3d[0] + 22, p3d[1] + 22)), fill=COL["red"], outline=COL["red_dark"], width=5 * S)
text((960, 586), "估計出的 3D point", label_f, COL["ink"], anchor="mm")
text((960, 682), "兩條視線最接近的交會位置", small_f, COL["muted"], anchor="mm")

# Bottom explanation
rounded((170, 970, 1750, 1042), 12, "#eef4fb", "#cbd5e1", 2)
text((960, 995), "三角化 triangulation：同一個真實場景點在兩張影像中各有一個 2D 投影，", body_f, COL["ink"], anchor="mm")
text((960, 1026), "從兩個相機中心反投影出兩條射線，射線交會處就是該 3D 點的估計位置。", body_f, COL["ink"], anchor="mm")

# Save with antialiasing
img = img.resize((W, H), Image.Resampling.LANCZOS)
img.save(OUT, "JPEG", quality=95, subsampling=0, optimize=True)
print(OUT)
