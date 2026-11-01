"""
Verdant 3D - local pipeline (no external AI API).

Pieces:
  plan_from_prompt(prompt)      -> recipe match + ordered build steps (local recipe library)
  photo_to_blocks(image, ...)   -> block model built from a photo (colour mosaic, row by row)
  build_steps(parts_groups)     -> cumulative snapshots, one per step
  render_step(parts, ...)       -> PNG of the model using a fixed isometric camera (Pillow only)

A model is a list of parts: {"id", "pos": [x,y,z], "size": [w,h,d], "color": "#rrggbb"}.
Units are studs. y is up. Each part is an axis-aligned box.
"""
from __future__ import annotations

import io
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

# ------------------------------------------------------------------ colours

NAMED_COLORS = {
    "red": "#c0392b", "blue": "#2e6fd1", "green": "#2e9e4f", "yellow": "#f5c518",
    "orange": "#e67e22", "purple": "#8e44ad", "pink": "#e88fb0", "white": "#eeeeee",
    "black": "#222222", "grey": "#8a8f98", "gray": "#8a8f98", "brown": "#7b4a2a",
    "beige": "#d9c7a3", "cream": "#efe6d2", "gold": "#d4a72c", "teal": "#1b8c86",
}


def _hex_to_rgb(c: str) -> Tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(v)))) for v in rgb)


def shade(color: str, f: float) -> str:
    r, g, b = _hex_to_rgb(color)
    return _rgb_to_hex((r * f, g * f, b * f))


def colour_from_prompt(prompt: str, default: str) -> str:
    low = prompt.lower()
    for word, hexv in NAMED_COLORS.items():
        if re.search(r"\b" + word + r"\b", low):
            return hexv
    return default


# ------------------------------------------------------------------ recipe library

def _part(pid: str, pos, size, color) -> Dict[str, Any]:
    return {"id": pid, "pos": [float(v) for v in pos], "size": [float(v) for v in size], "color": color}


def _recipe_house(accent: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    wall, roof, door, win = "#d9c7a3", accent if accent != "#d9c7a3" else "#8b2f2f", "#6b4226", "#9fd8ff"
    return [
        ("Foundation", [_part("found", [0, 0.5, 0], [12, 1, 10], "#8a8f98")]),
        ("Side and back walls", [
            _part("wall_back", [0, 3.5, -4.5], [12, 5, 1], wall),
            _part("wall_left", [-5.5, 3.5, 0], [1, 5, 8], wall),
            _part("wall_right", [5.5, 3.5, 0], [1, 5, 8], wall)]),
        ("Front wall with door and windows", [
            _part("front_l", [-3.5, 3.5, 4.5], [5, 5, 1], wall),
            _part("front_r", [3.5, 3.5, 4.5], [5, 5, 1], wall),
            _part("front_top", [0, 5, 4.5], [2, 2, 1], wall),
            _part("door", [0, 2.5, 4.5], [2, 3, 1], door),
            _part("win_l", [-3.5, 3.5, 5.1], [2, 2, 0.2], win),
            _part("win_r", [3.5, 3.5, 5.1], [2, 2, 0.2], win)]),
        ("Roof", [
            _part("roof_l", [-3, 6.5, 0], [6, 1, 10], roof),
            _part("roof_r", [3, 6.5, 0], [6, 1, 10], shade(roof, 0.85)),
            _part("roof_ridge", [0, 7.5, 0], [1, 1, 10], shade(roof, 0.7))]),
    ]


def _recipe_tower(accent: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    body = accent if accent != "#d9c7a3" else "#b9bec7"
    steps = [("Base", [_part("base", [0, 0.5, 0], [6, 1, 6], "#5b616b")])]
    for i in range(1, 6):
        y = 1 + (i - 0.5) * 2          # floor i spans y = 1+2(i-1) .. 3+2(i-1)
        steps.append((f"Floor {i}", [
            _part(f"floor{i}", [0, y, 0], [5, 2, 5], body),
            _part(f"win{i}a", [0, y, 2.6], [2, 1, 0.2], "#9fd8ff"),
        ]))
    steps.append(("Top cap", [_part("cap", [0, 11.5, 0], [6, 1, 6], "#5b616b")]))
    return steps


def _recipe_stairs(accent: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    col = accent if accent != "#d9c7a3" else "#c9a27a"
    # Each step is 1 higher and 1 further on; neighbours overlap by 1 stud so they are attached.
    return [(f"Step {i + 1}", [_part(f"stair{i}", [0, 0.5 + i, 2 + i], [4, 1, 2], col)]) for i in range(6)]


def _recipe_car(accent: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    body = accent if accent != "#d9c7a3" else "#2e6fd1"
    wheels = "#222222"
    return [
        ("Chassis", [_part("chassis", [0, 1.5, 0], [8, 1.5, 4], body)]),
        ("Cabin base", [_part("cabin", [-0.5, 2.75, 0], [4.5, 1, 3.4], shade(body, 0.85))]),
        ("Windows", [_part("glass", [-0.5, 3.75, 0], [4.5, 1, 3.4], "#9fd8ff")]),
        ("Wheels", [
            _part("wheel_fl", [-2.6, 0.7, 2.4], [1.4, 1.4, 0.8], wheels),
            _part("wheel_fr", [2.6, 0.7, 2.4], [1.4, 1.4, 0.8], wheels),
            _part("wheel_bl", [-2.6, 0.7, -2.4], [1.4, 1.4, 0.8], wheels),
            _part("wheel_br", [2.6, 0.7, -2.4], [1.4, 1.4, 0.8], wheels)]),
    ]


RECIPES = {
    "house": {"keys": ["house", "home", "cottage", "cabin", "hut"], "build": _recipe_house, "default": "#d9c7a3"},
    "tower": {"keys": ["tower", "castle", "skyscraper", "building", "turret"], "build": _recipe_tower, "default": "#b9bec7"},
    "stairs": {"keys": ["stairs", "stair", "staircase", "steps", "ladder"], "build": _recipe_stairs, "default": "#c9a27a"},
    "car": {"keys": ["car", "vehicle", "truck", "taxi", "racecar", "auto"], "build": _recipe_car, "default": "#2e6fd1"},
}


def plan_from_prompt(prompt: str) -> Dict[str, Any]:
    """Match a prompt to a recipe. Taught recipes are checked first, then the built-in library."""
    import teach
    hit = teach.match(prompt)
    if hit:
        rec, _score = hit
        teach.mark_used(rec["id"])
        return {"matched": True, "recipe": f"taught:{rec['id']}",
                "steps": [{"name": s["name"], "parts": s["parts"]} for s in rec["steps"]],
                "notes": [f"Used a taught recipe (from \"{rec['prompt']}\", rating {rec['rating']:+d})."]}

    words = set(re.findall(r"[a-z]+", prompt.lower()))
    best_name, best_score = None, 0
    for name, spec in RECIPES.items():
        score = sum(1 for k in spec["keys"] if k in words)
        if score > best_score:
            best_name, best_score = name, score

    if best_name is None:
        return {"matched": False, "recipe": None, "steps": [],
                "notes": ["No recipe matches this prompt. Add a photo, or add a recipe to RECIPES."]}

    spec = RECIPES[best_name]
    accent = colour_from_prompt(prompt, spec["default"])
    steps = spec["build"](accent)
    return {
        "matched": True,
        "recipe": best_name,
        "steps": [{"name": n, "parts": p} for n, p in steps],
        "notes": [f"Matched recipe '{best_name}' (colour: {accent})."],
    }


# ------------------------------------------------------------------ photo -> block mosaic

def photo_to_blocks(image: Image.Image, grid_w: int = 32, depth: float = 1.0,
                    quant_bits: int = 4, max_steps: int = 24) -> Dict[str, Any]:
    """
    Turn a photo into a flat wall of coloured blocks: one block per pixel, merged into runs of the
    same colour along each row. Colours are quantised so neighbouring pixels merge into fewer parts.
    Returns steps grouped by row band (bottom to top) so the build can be shown step by step.
    This is a colour mosaic of the photo, not a 3D reconstruction of the scene.
    """
    img = image.convert("RGB")
    w, h = img.size
    grid_w = max(4, min(64, int(grid_w)))
    grid_h = max(4, int(round(grid_w * h / max(1, w))))
    grid_h = min(grid_h, 64)
    img = img.resize((grid_w, grid_h), Image.BILINEAR)

    step_size = 256 >> quant_bits
    px = img.load()

    def q(v: int) -> int:
        return min(255, (v // step_size) * step_size + step_size // 2)

    parts: List[Dict[str, Any]] = []
    rows_bands: List[List[Dict[str, Any]]] = []
    pid = 0
    # Row 0 of the image is the top of the picture; build from the bottom (grid_h-1) up.
    for band_index, img_row in enumerate(range(grid_h - 1, -1, -1)):
        y = band_index  # height level of this row
        band: List[Dict[str, Any]] = []
        x = 0
        while x < grid_w:
            col = _rgb_to_hex(tuple(q(c) for c in px[x, img_row]))
            run = 1
            while x + run < grid_w and _rgb_to_hex(tuple(q(c) for c in px[x + run, img_row])) == col:
                run += 1
            cx = x + run / 2 - grid_w / 2
            part = _part(f"px_{pid}", [cx, y + 0.5, 0], [run, 1, depth], col)
            pid += 1
            band.append(part)
            x += run
        rows_bands.append(band)
        parts.extend(band)

    per = math.ceil(len(rows_bands) / max(1, max_steps))
    steps = []
    for start in range(0, len(rows_bands), per):
        group = rows_bands[start:start + per]
        lo_row, hi_row = start + 1, start + len(group)
        steps.append({"name": f"Rows {lo_row}-{hi_row} of {len(rows_bands)}" if per > 1 else f"Row {lo_row}",
                      "parts": [p for band in group for p in band]})

    return {
        "matched": True,
        "recipe": "photo_mosaic",
        "steps": steps,
        "notes": [f"Photo converted to a {grid_w}x{grid_h} colour mosaic with {len(parts)} blocks.",
                  "This is a colour mosaic of the photo, not a 3D reconstruction."],
    }


# ------------------------------------------------------------------ steps and checks

def cumulative_parts(steps: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Snapshot after each step: the model as it stands once that step is finished."""
    snaps, acc = [], []
    for s in steps:
        acc = acc + list(s["parts"])
        snaps.append(list(acc))
    return snaps


def check_model(parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Checks on a box model:
      floating: parts not connected (by a shared face) to the ground through other parts
      overlaps: pairs of parts whose volumes intersect
    Uses a spatial grid so large models stay fast.
    """
    from collections import defaultdict, deque

    boxes = []
    for p in parts:
        (x, y, z), (w, h, d) = p["pos"], p["size"]
        boxes.append(((x - w / 2, y - h / 2, z - d / 2), (x + w / 2, y + h / 2, z + d / 2), p["id"]))

    cell = 2.0
    grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i, (lo, hi, _) in enumerate(boxes):
        for cx in range(math.floor(lo[0] / cell), math.floor(hi[0] / cell) + 1):
            for cy in range(math.floor(lo[1] / cell), math.floor(hi[1] / cell) + 1):
                for cz in range(math.floor(lo[2] / cell), math.floor(hi[2] / cell) + 1):
                    grid[(cx, cy, cz)].append(i)

    def near(i):
        lo, hi, _ = boxes[i]
        seen = set()
        for cx in range(math.floor((lo[0] - 0.1) / cell), math.floor((hi[0] + 0.1) / cell) + 1):
            for cy in range(math.floor((lo[1] - 0.1) / cell), math.floor((hi[1] + 0.1) / cell) + 1):
                for cz in range(math.floor((lo[2] - 0.1) / cell), math.floor((hi[2] + 0.1) / cell) + 1):
                    seen.update(grid.get((cx, cy, cz), ()))
        seen.discard(i)
        return seen

    adj: Dict[int, List[int]] = defaultdict(list)
    overlaps = 0
    tol = 0.05
    for i in range(len(boxes)):
        lo, hi, _ = boxes[i]
        for j in near(i):
            if j < i:
                continue
            lo2, hi2, _ = boxes[j]
            ov = [min(hi[k], hi2[k]) - max(lo[k], lo2[k]) for k in range(3)]
            if all(o > 0.01 for o in ov):
                overlaps += 1
                continue
            # face contact: one axis touches, the other two overlap by a real area
            touching = [abs(hi2[k] - lo[k]) < tol or abs(hi[k] - lo2[k]) < tol for k in range(3)]
            if sum(touching) == 1:
                k = touching.index(True)
                others = [o for idx, o in enumerate(ov) if idx != k]
                if all(o > 0.01 for o in others):
                    adj[i].append(j)
                    adj[j].append(i)

    reached = set()
    q = deque(i for i, (lo, _, _) in enumerate(boxes) if lo[1] <= 0.01)
    reached.update(q)
    while q:
        i = q.popleft()
        for j in adj[i]:
            if j not in reached:
                reached.add(j)
                q.append(j)
    floating = [boxes[i][2] for i in range(len(boxes)) if i not in reached]
    return {"parts": len(parts), "floating": floating[:20], "floating_count": len(floating),
            "overlaps": overlaps, "ok": not floating and overlaps == 0}


# ------------------------------------------------------------------ renderer (isometric, Pillow only)

COS30, SIN30 = math.cos(math.radians(30)), 0.5


def _project(x: float, y: float, z: float, scale: float, ox: float, oy: float) -> Tuple[float, float]:
    u = (x - z) * COS30
    v = (x + z) * SIN30 - y
    return ox + u * scale, oy + v * scale


def _bounds(parts: Sequence[Dict[str, Any]]):
    if not parts:
        return (-1, 0, -1), (1, 1, 1)
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for p in parts:
        for k in range(3):
            lo[k] = min(lo[k], p["pos"][k] - p["size"][k] / 2)
            hi[k] = max(hi[k], p["pos"][k] + p["size"][k] / 2)
    return tuple(lo), tuple(hi)


def render_step(parts: Sequence[Dict[str, Any]], frame_bounds, width: int = 640, height: int = 480,
                title: Optional[str] = None) -> bytes:
    """
    Render parts as PNG bytes with a fixed camera. Pass the bounds of the FINAL model as
    frame_bounds so every step uses the same framing and camera.
    """
    img = Image.new("RGB", (width, height), (22, 27, 36))
    draw = ImageDraw.Draw(img)

    lo, hi = frame_bounds
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    pts = [(_project(x, y, z, 1.0, 0, 0)) for x, y, z in corners]
    minu = min(p[0] for p in pts); maxu = max(p[0] for p in pts)
    minv = min(p[1] for p in pts); maxv = max(p[1] for p in pts)
    span = max(maxu - minu, maxv - minv, 1e-6)
    scale = min(width, height) * 0.78 / span
    ox = width / 2 - (minu + maxu) / 2 * scale
    oy = height / 2 - (minv + maxv) / 2 * scale

    # Painter's order: draw far boxes first. Camera sits at (+x, +y, +z).
    order = sorted(parts, key=lambda p: sum(p["pos"]))
    for p in order:
        (x, y, z), (w, h, d) = p["pos"], p["size"]
        x0, x1 = x - w / 2, x + w / 2
        y0, y1 = y - h / 2, y + h / 2
        z0, z1 = z - d / 2, z + d / 2
        c = p["color"]
        top = [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
        right = [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
        front = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        for face, col in ((top, shade(c, 1.0)), (right, shade(c, 0.78)), (front, shade(c, 0.6))):
            poly = [_project(*v, scale, ox, oy) for v in face]
            draw.polygon(poly, fill=col, outline=shade(c, 0.4))

    if title:
        draw.text((12, 10), title, fill=(230, 235, 245))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_all(snapshots: List[List[Dict[str, Any]]], width: int = 640, height: int = 480,
               titles: Optional[List[str]] = None) -> List[bytes]:
    final = snapshots[-1] if snapshots else []
    fb = _bounds(final)
    out = []
    for i, snap in enumerate(snapshots):
        t = titles[i] if titles and i < len(titles) else None
        out.append(render_step(snap, fb, width, height, t))
    return out
