from __future__ import annotations

import base64
import json
import math
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components


WIDTH = 820
HEIGHT = 500
FPS = 15
MAX_FRAMES = 96
TABLE_HEIGHT = 0.75

OBJECT_CLASSES = ("cube", "cup", "bottle", "box")
BIN_LABELS = {"bin_a": "Bin A", "bin_b": "Bin B"}
DEFAULT_COMMAND = "put the cup in the bin"

DISPLAY_HEIGHT_M = {
    "cube": 0.13,
    "cup": 0.15,
    "bottle": 0.22,
    "box": 0.17,
}

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(APP_DIR, "assets")
ASSET_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 10px 12px;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}

.header-card {
    background: linear-gradient(135deg, #0f172a 0%, #172554 55%, #0b1020 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 14px 16px;
    color: #e5e7eb;
    margin-bottom: 12px;
}

.header-title {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 0.2px;
    color: #f8fafc;
}

.header-sub {
    font-size: 0.86rem;
    color: #94a3b8;
    margin-top: 4px;
}

.pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.3px;
    color: #0b1020;
}

.pill-success { background: #22c55e; }
.pill-failure { background: #ef4444; color: #fff; }
.pill-info { background: #38bdf8; }
.pill-warn { background: #f59e0b; }

.industrial-player {
    width: 100%;
    background: #0b1020;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 10px;
    box-sizing: border-box;
}

.player-top {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    align-items: center;
    color: #dbeafe;
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 12px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}

.player-badge {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 4px 9px;
}

.player-img-wrap {
    width: 100%;
    background: #000;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1f2937;
}

.player-img-wrap img {
    width: 100%;
    display: block;
    background: #000;
}

.player-controls {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 9px;
    font-family: ui-sans-serif, system-ui, sans-serif;
    color: #dbeafe;
    font-size: 12px;
}

.player-controls button,
.player-controls select {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 7px 11px;
    cursor: pointer;
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 12px;
}

.player-controls button.secondary {
    background: #2ea043;
}

.player-controls input[type="range"] {
    flex: 1 1 220px;
    min-width: 160px;
}

.chart-card {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 10px;
    margin: 8px 0;
}

.chart-title {
    color: #e5e7eb;
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 6px;
}

.health-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 7px 0;
    font-family: ui-sans-serif, system-ui, sans-serif;
    color: #e5e7eb;
    font-size: 13px;
}

.health-label {
    width: 135px;
    color: #cbd5e1;
}

.health-track {
    flex: 1;
    height: 10px;
    background: #1f2937;
    border-radius: 999px;
    overflow: hidden;
}

.health-fill {
    height: 100%;
    border-radius: 999px;
}

.health-value {
    width: 52px;
    text-align: right;
    color: #f8fafc;
    font-weight: 700;
}

.alarm-card {
    border-left: 4px solid #f59e0b;
    background: #111827;
    color: #fde68a;
    border-radius: 10px;
    padding: 8px 10px;
    margin: 6px 0;
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 13px;
}

.alarm-error {
    border-left-color: #ef4444;
    color: #fecaca;
}

.empty-note {
    color: #94a3b8;
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 13px;
    padding: 8px 0;
}
</style>
"""


PLAYER_HTML = """
<div class="industrial-player">
  <div class="player-top">
    <div>
      <span class="player-badge">LIVE SIMULATION</span>
      <span class="player-badge">SYNTHETIC CELL</span>
      <span class="player-badge">NO API KEY</span>
    </div>
    <div id="sim-clock-__UID__" class="player-badge">t=0.00s</div>
  </div>

  <div class="player-img-wrap">
    <img
      id="sim-img-__UID__"
      src="data:image/jpeg;base64,__FIRST_FRAME__"
      alt="Live simulation frame"
    />
  </div>

  <div class="player-controls">
    <button id="sim-toggle-__UID__">Pause</button>
    <button id="sim-restart-__UID__" class="secondary">Restart</button>
    <label>
      Speed
      <select id="sim-speed-__UID__">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="1.5">1.5x</option>
        <option value="2">2x</option>
      </select>
    </label>
    <input id="sim-range-__UID__" type="range" min="0" max="0" value="0" />
    <span id="sim-status-__UID__">Ready</span>
  </div>
</div>

<script>
(function() {
  var payload = __PAYLOAD_JSON__;
  var frames = payload.frames || [];
  var meta = payload.meta || [];
  var idx = 0;
  var playing = true;
  var timer = null;
  var speed = 1.0;
  var baseInterval = __BASE_INTERVAL__;

  var img = document.getElementById("sim-img-__UID__");
  var toggle = document.getElementById("sim-toggle-__UID__");
  var restart = document.getElementById("sim-restart-__UID__");
  var range = document.getElementById("sim-range-__UID__");
  var status = document.getElementById("sim-status-__UID__");
  var clock = document.getElementById("sim-clock-__UID__");
  var speedSelect = document.getElementById("sim-speed-__UID__");

  function show(i) {
    if (frames.length === 0) return;
    i = Math.max(0, Math.min(frames.length - 1, i));
    idx = i;
    img.src = "data:image/jpeg;base64," + frames[i];
    range.value = i;

    var m = meta[i] || {phase: "UNKNOWN", time: 0};
    status.textContent = "Frame " + (i + 1) + "/" + frames.length + " • " + m.phase;
    clock.textContent = "t=" + Number(m.time || 0).toFixed(2) + "s";
  }

  function tick() {
    idx = (idx + 1) % frames.length;
    show(idx);
  }

  function play() {
    if (timer !== null || frames.length === 0) return;
    playing = true;
    toggle.textContent = "Pause";
    var interval = Math.max(16, Math.round(baseInterval / speed));
    timer = setInterval(tick, interval);
  }

  function pause() {
    playing = false;
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
    toggle.textContent = "Play";
  }

  toggle.addEventListener("click", function() {
    if (playing) pause();
    else play();
  });

  restart.addEventListener("click", function() {
    show(0);
    if (!playing) play();
  });

  range.addEventListener("input", function() {
    pause();
    show(parseInt(range.value, 10));
  });

  speedSelect.addEventListener("change", function() {
    speed = parseFloat(speedSelect.value) || 1.0;
    if (playing) {
      clearInterval(timer);
      timer = null;
      play();
    }
  });

  if (frames.length > 0) {
    range.max = String(frames.length - 1);
    show(0);
    play();
  } else {
    status.textContent = "No frames generated";
  }
})();
</script>
"""


@dataclass
class ParsedCommand:
    ok: bool
    object_name: str = "cup"
    bin_key: str = "bin_a"
    error: str = ""


@dataclass
class SimObject:
    name: str
    true_base: np.ndarray
    est_base: np.ndarray
    display_h: float
    sprite: np.ndarray
    confidence: float
    detected: bool
    held: bool = False
    placed: bool = False
    base_pos: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.base_pos = self.true_base.copy()


@dataclass
class Bin:
    key: str
    label: str
    pos: np.ndarray
    color: tuple[int, int, int]
    width: float = 0.22
    depth: float = 0.17
    height: float = 0.13


@dataclass
class Camera:
    pos: np.ndarray
    R: np.ndarray
    K: np.ndarray


@dataclass
class Segment:
    phase: str
    duration: int
    waypoint: np.ndarray
    action: str = ""
    action_at: float = 0.0


def parse_command(text: str) -> ParsedCommand:
    raw = str(text or "")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", raw.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ParsedCommand(
            ok=False,
            error="Empty command. Try: put the cup in the bin.",
        )

    synonyms = {
        "cube": ["cube", "block", "dice"],
        "cup": ["cup", "mug", "glass"],
        "bottle": ["bottle", "jar"],
        "box": ["box", "carton", "package"],
    }

    object_name = ""
    for name, words in synonyms.items():
        if any(re.search(rf"\b{re.escape(word)}\b", cleaned) for word in words):
            object_name = name
            break

    if not object_name:
        return ParsedCommand(
            ok=False,
            error=(
                "Unknown object. Allowed objects are cube, cup, bottle, box. "
                "Try: put the cup in the bin."
            ),
        )

    bin_key = "bin_a"
    if re.search(r"\b(?:b|right|second|far)\b", cleaned):
        bin_key = "bin_b"
    elif re.search(r"\b(?:a|left|first|near)\b", cleaned):
        bin_key = "bin_a"

    has_action_or_destination = bool(
        re.search(
            r"\b(?:put|place|move|transfer|send|drop|set|pick|grab|retrieve|fetch|bin|container|target)\b",
            cleaned,
        )
    )

    if not has_action_or_destination:
        return ParsedCommand(
            ok=False,
            error="Missing action or destination. Try: put the cup in the bin.",
        )

    return ParsedCommand(
        ok=True,
        object_name=object_name,
        bin_key=bin_key,
    )


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec
    return vec / norm


def ease(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def lerp_vec(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def clamp_xy(point: np.ndarray) -> np.ndarray:
    out = point.copy()
    out[0] = float(np.clip(out[0], -0.36, 0.36))
    out[1] = float(np.clip(out[1], 0.42, 0.88))
    return out


def make_camera() -> Camera:
    cam_pos = np.array([0.0, 0.0, 1.48], dtype=float)
    look_at = np.array([0.0, 0.66, TABLE_HEIGHT], dtype=float)

    forward = normalize(look_at - cam_pos)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = normalize(np.cross(forward, world_up))
    down = normalize(np.cross(forward, right))

    rotation = np.vstack([right, down, forward])
    intrinsic = np.array(
        [
            [720.0, 0.0, WIDTH / 2.0],
            [0.0, 720.0, HEIGHT / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return Camera(pos=cam_pos, R=rotation, K=intrinsic)


def project(point: np.ndarray, cam: Camera) -> Optional[tuple[int, int, float]]:
    point = np.asarray(point, dtype=float)
    cam_point = cam.R @ (point - cam.pos)
    depth = float(cam_point[2])
    if depth <= 0.08:
        return None

    u = cam.K[0, 0] * cam_point[0] / depth + cam.K[0, 2]
    v = cam.K[1, 1] * cam_point[1] / depth + cam.K[1, 2]
    if not (math.isfinite(u) and math.isfinite(v)):
        return None

    return int(round(u)), int(round(v)), depth


def ensure_bgra(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        bgr = img.copy()
    elif img.shape[2] == 4:
        return img.copy()
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    return np.dstack([bgr, alpha]).astype(np.uint8)


def remove_light_background(bgra: np.ndarray) -> np.ndarray:
    bgr = bgra[:, :, :3].copy()
    h, w = bgr.shape[:2]

    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    corner_values = [bgr[y, x] for x, y in corners]

    light_border = all(
        float(np.mean(v)) > 165.0 and float(np.max(v) - np.min(v)) < 55.0
        for v in corner_values
    )
    if not light_border:
        return bgra

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    lo = (30, 30, 30)
    up = (30, 30, 30)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE

    for seed in corners:
        cv2.floodFill(bgr, mask, seed, 0, lo, up, flags)

    bg = mask[1:-1, 1:-1] == 255
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    _, alpha = cv2.threshold(alpha, 12, 255, cv2.THRESH_BINARY)

    out = bgra.copy()
    out[:, :, 3] = alpha
    return out


def trim_sprite(img: np.ndarray, pad: int = 5) -> np.ndarray:
    alpha = img[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)
    return img[y : y + h, x : x + w].copy()


def prepare_real_sprite(img: np.ndarray) -> np.ndarray:
    bgra = ensure_bgra(img)

    if np.all(bgra[:, :, 3] == 255):
        bgra = remove_light_background(bgra)

    bgra = trim_sprite(bgra)

    if bgra.shape[0] < 40:
        scale = 40.0 / float(bgra.shape[0])
        bgra = cv2.resize(
            bgra,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )

    return bgra


def make_cup_sprite() -> np.ndarray:
    h, w = 210, 175
    cx = w // 2

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, 45), (55, 22), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (cx - 55, 45), (cx + 55, 168), 255, -1)
    cv2.ellipse(mask, (cx, 168), (55, 22), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (cx + 64, 105), (31, 43), 0, -75, 75, 255, 14)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:] = (242, 242, 242)

    yy, xx = np.mgrid[0:h, 0:w]
    shade = 1.0 - 0.25 * (yy / float(h)) + 0.10 * np.cos((xx - cx) / float(w) * math.pi)
    bgr = np.clip(bgr * shade[:, :, None], 0, 255).astype(np.uint8)

    cv2.ellipse(bgr, (cx, 45), (47, 16), 0, 0, 360, (45, 35, 80), -1)
    cv2.ellipse(bgr, (cx, 45), (38, 12), 0, 0, 360, (25, 20, 55), -1)

    highlight = np.zeros_like(bgr)
    cv2.ellipse(highlight, (cx - 25, 105), (11, 52), 0, 0, 360, (255, 255, 255), -1)
    highlight = cv2.GaussianBlur(highlight, (17, 17), 0)

    alpha_h = np.clip(highlight.max(axis=2) / 255.0 * 0.45, 0.0, 1.0)
    bgr = (bgr * (1.0 - alpha_h[:, :, None]) + highlight * alpha_h[:, :, None]).astype(np.uint8)

    return np.dstack([bgr, mask]).astype(np.uint8)


def make_bottle_sprite() -> np.ndarray:
    h, w = 255, 135
    cx = w // 2

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (cx - 19, 20), (cx + 19, 62), 255, -1)
    cv2.ellipse(mask, (cx, 66), (45, 28), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (cx - 45, 66), (cx + 45, 218), 255, -1)
    cv2.ellipse(mask, (cx, 218), (45, 22), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (cx - 23, 4), (cx + 23, 24), 255, -1)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:] = (120, 195, 80)

    yy, xx = np.mgrid[0:h, 0:w]
    shade = 1.0 - 0.22 * (yy / float(h)) + 0.12 * np.cos((xx - cx) / float(w) * math.pi)
    bgr = np.clip(bgr * shade[:, :, None], 0, 255).astype(np.uint8)

    cv2.rectangle(bgr, (cx - 40, 116), (cx + 40, 182), (248, 248, 248), -1)
    cv2.rectangle(bgr, (cx - 30, 133), (cx + 30, 147), (80, 80, 80), -1)
    cv2.rectangle(bgr, (cx - 23, 4), (cx + 23, 24), (60, 60, 220), -1)

    highlight = np.zeros_like(bgr)
    cv2.rectangle(highlight, (cx - 30, 80), (cx - 17, 208), (255, 255, 255), -1)
    highlight = cv2.GaussianBlur(highlight, (15, 15), 0)

    alpha_h = np.clip(highlight.max(axis=2) / 255.0 * 0.35, 0.0, 1.0)
    bgr = (bgr * (1.0 - alpha_h[:, :, None]) + highlight * alpha_h[:, :, None]).astype(np.uint8)

    return np.dstack([bgr, mask]).astype(np.uint8)


def make_cube_sprite() -> np.ndarray:
    h, w = 175, 180

    front = np.array([[30, 62], [118, 62], [118, 150], [30, 150]], dtype=np.int32)
    top = np.array([[30, 62], [72, 26], [160, 26], [118, 62]], dtype=np.int32)
    side = np.array([[118, 62], [160, 26], [160, 114], [118, 150]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [front], 255)
    cv2.fillPoly(mask, [top], 255)
    cv2.fillPoly(mask, [side], 255)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.fillPoly(bgr, [front], (238, 188, 88))
    cv2.fillPoly(bgr, [top], (255, 224, 135))
    cv2.fillPoly(bgr, [side], (186, 136, 56))

    for pts in (front, top, side):
        cv2.polylines(bgr, [pts], True, (35, 35, 35), 2)

    return np.dstack([bgr, mask]).astype(np.uint8)


def make_box_sprite() -> np.ndarray:
    h, w = 190, 230

    front = np.array([[38, 82], [145, 82], [145, 174], [38, 174]], dtype=np.int32)
    side = np.array([[145, 82], [194, 52], [194, 144], [145, 174]], dtype=np.int32)
    top = np.array([[38, 82], [87, 52], [194, 52], [145, 82]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [front], 255)
    cv2.fillPoly(mask, [side], 255)
    cv2.fillPoly(mask, [top], 255)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.fillPoly(bgr, [front], (90, 155, 205))
    cv2.fillPoly(bgr, [side], (65, 120, 170))
    cv2.fillPoly(bgr, [top], (120, 180, 225))

    cv2.rectangle(bgr, (82, 82), (103, 174), (190, 220, 245), -1)

    for pts in (front, side, top):
        cv2.polylines(bgr, [pts], True, (45, 45, 45), 2)

    return np.dstack([bgr, mask]).astype(np.uint8)


def make_fallback_sprites() -> dict[str, np.ndarray]:
    return {
        "cube": make_cube_sprite(),
        "cup": make_cup_sprite(),
        "bottle": make_bottle_sprite(),
        "box": make_box_sprite(),
    }


@st.cache_resource(show_spinner=False)
def get_default_sprites() -> dict[str, np.ndarray]:
    sprites = make_fallback_sprites()

    for name in OBJECT_CLASSES:
        for ext in ASSET_EXTS:
            path = os.path.join(ASSET_DIR, f"{name}{ext}")
            if not os.path.isfile(path):
                continue

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            try:
                sprites[name] = prepare_real_sprite(img)
            except Exception:
                pass
            break

    return sprites


def current_sprites() -> dict[str, np.ndarray]:
    sprites = get_default_sprites().copy()
    uploaded = st.session_state.get("uploaded_sprites", {})
    sprites.update(uploaded)
    return sprites


def scenario_config(name: str) -> dict[str, Any]:
    cfg = {
        "spacing": 0.17,
        "conf_bias": 0.0,
        "noise_mult": 1.0,
        "bad_class": None,
        "place_bias_mm": 0.0,
        "force_inject": False,
        "description": "Standard tabletop manipulation scenario.",
    }

    if name == "High clutter":
        cfg.update(
            {
                "spacing": 0.135,
                "conf_bias": -0.045,
                "noise_mult": 1.12,
                "description": "Objects placed closer together; higher occlusion risk.",
            }
        )
    elif name == "Low light / noisy depth":
        cfg.update(
            {
                "conf_bias": -0.12,
                "noise_mult": 1.85,
                "description": "Reduced detection confidence and higher depth noise.",
            }
        )
    elif name == "Transparent cup risk":
        cfg.update(
            {
                "conf_bias": -0.035,
                "noise_mult": 1.25,
                "bad_class": "cup",
                "description": "Cup depth is less reliable, mimicking transparent/reflective failure.",
            }
        )
    elif name == "Tight bin placement":
        cfg.update(
            {
                "place_bias_mm": 18.0,
                "description": "Placement tolerance is tighter; higher placement error.",
            }
        )
    elif name == "Recovery stress test":
        cfg.update(
            {
                "noise_mult": 1.35,
                "conf_bias": -0.05,
                "force_inject": True,
                "description": "Forces an injected slip to exercise recovery logic.",
            }
        )

    return cfg


def make_scene(
    rng: np.random.Generator,
    target_name: str,
    target_bin: str,
    noise_m: float,
    sprites: dict[str, np.ndarray],
    sc: dict[str, Any],
) -> tuple[list[SimObject], list[Bin], int]:
    target_name = target_name if target_name in OBJECT_CLASSES else "cup"

    objects: list[SimObject] = []
    occupied: list[tuple[float, float]] = []
    bin_positions = [(-0.33, 0.69), (0.33, 0.69)]

    for name in OBJECT_CLASSES:
        x, y = 0.0, 0.0
        for _ in range(350):
            x = float(rng.uniform(-0.28, 0.28))
            y = float(rng.uniform(0.50, 0.80))

            ok = True
            for px, py in occupied:
                if math.hypot(x - px, y - py) <= float(sc["spacing"]):
                    ok = False
                    break

            if ok:
                for bx, by in bin_positions:
                    if math.hypot(x - bx, y - by) <= 0.21:
                        ok = False
                        break

            if ok:
                break

        occupied.append((x, y))

        display_h = DISPLAY_HEIGHT_M[name]
        true_base = np.array([x, y, TABLE_HEIGHT], dtype=float)

        noise_scale = max(float(noise_m), 1e-4) * float(sc["noise_mult"])
        conf_mean = 0.89 + float(sc["conf_bias"])

        if sc.get("bad_class") == name:
            noise_scale *= 1.45
            conf_mean -= 0.08

        est_base = true_base + rng.normal(0.0, noise_scale, 3)
        est_base = clamp_xy(est_base)
        est_base[2] = TABLE_HEIGHT

        confidence = float(np.clip(rng.normal(conf_mean, 0.045), 0.18, 0.99))
        detected = confidence >= 0.40

        if name == target_name:
            confidence = max(confidence, 0.76)
            detected = True

        objects.append(
            SimObject(
                name=name,
                true_base=true_base,
                est_base=est_base,
                display_h=display_h,
                sprite=sprites[name],
                confidence=confidence,
                detected=detected,
            )
        )

    bins = [
        Bin(
            key="bin_a",
            label=BIN_LABELS["bin_a"],
            pos=np.array([-0.33, 0.69, TABLE_HEIGHT], dtype=float),
            color=(190, 130, 50),
        ),
        Bin(
            key="bin_b",
            label=BIN_LABELS["bin_b"],
            pos=np.array([0.33, 0.69, TABLE_HEIGHT], dtype=float),
            color=(80, 160, 220),
        ),
    ]

    target_idx = next(i for i, obj in enumerate(objects) if obj.name == target_name)
    _ = target_bin
    return objects, bins, target_idx


def fill_poly_alpha(
    img: np.ndarray,
    pts: list[tuple[int, int]],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if len(pts) < 3:
        return

    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
    blended = cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0.0)
    img[:] = blended


def draw_bin(img: np.ndarray, bin_obj: Bin, cam: Camera) -> None:
    x, y, z = bin_obj.pos
    hw = bin_obj.width / 2.0
    hd = bin_obj.depth / 2.0
    hh = bin_obj.height

    world_points = {
        "fbl": [x - hw, y - hd, z],
        "fbr": [x + hw, y - hd, z],
        "bbl": [x - hw, y + hd, z],
        "bbr": [x + hw, y + hd, z],
        "ftl": [x - hw, y - hd, z + hh],
        "ftr": [x + hw, y - hd, z + hh],
        "btl": [x - hw, y + hd, z + hh],
        "btr": [x + hw, y + hd, z + hh],
    }

    proj: dict[str, tuple[int, int]] = {}
    ok = True
    for key, point in world_points.items():
        p = project(np.array(point, dtype=float), cam)
        if p is None:
            ok = False
            break
        proj[key] = (p[0], p[1])

    if not ok:
        center = project(np.array([x, y, z + hh * 0.5], dtype=float), cam)
        if center is None:
            return

        u, v, depth = center
        size = max(34, int(0.20 * cam.K[0, 0] / max(depth, 0.1)))
        w, h = size, int(size * 0.72)
        x1, y1 = u - w // 2, v - h // 2
        x2, y2 = u + w // 2, v + h // 2

        fill_poly_alpha(img, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], bin_obj.color, 0.38)
        cv2.rectangle(img, (x1, y1), (x2, y2), (35, 35, 35), 2)
        cv2.putText(
            img,
            bin_obj.label,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        return

    top = [proj["ftl"], proj["ftr"], proj["btr"], proj["btl"]]
    front = [proj["fbl"], proj["fbr"], proj["ftr"], proj["ftl"]]
    left = [proj["fbl"], proj["ftl"], proj["btl"], proj["bbl"]]
    right = [proj["fbr"], proj["ftr"], proj["btr"], proj["bbr"]]

    darker = (
        max(0, bin_obj.color[0] - 45),
        max(0, bin_obj.color[1] - 45),
        max(0, bin_obj.color[2] - 45),
    )
    lighter = (
        min(255, bin_obj.color[0] + 35),
        min(255, bin_obj.color[1] + 35),
        min(255, bin_obj.color[2] + 35),
    )

    fill_poly_alpha(img, left, darker, 0.64)
    fill_poly_alpha(img, right, darker, 0.64)
    fill_poly_alpha(img, front, bin_obj.color, 0.80)
    fill_poly_alpha(img, top, lighter, 0.40)

    for poly in (top, front, left, right):
        cv2.polylines(img, [np.array(poly, dtype=np.int32)], True, (25, 25, 25), 2)

    label_pt = proj["ftl"]
    cv2.putText(
        img,
        bin_obj.label,
        (label_pt[0], max(14, label_pt[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def make_base_image(cam: Camera, bins: list[Bin]) -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for yy in range(HEIGHT):
        t = yy / max(1, HEIGHT - 1)
        if yy < 260:
            img[yy, :, 0] = int(188 - 80 * t)
            img[yy, :, 1] = int(158 - 65 * t)
            img[yy, :, 2] = int(132 - 55 * t)
        else:
            tt = (yy - 260) / max(1, HEIGHT - 261)
            img[yy, :, 0] = int(95 - 30 * tt)
            img[yy, :, 1] = int(105 - 35 * tt)
            img[yy, :, 2] = int(120 - 40 * tt)

    tex_rng = np.random.default_rng(17)
    noise = tex_rng.normal(0.0, 1.6, img.shape[:2])
    img = np.clip(img.astype(np.float32) + noise[:, :, None] * 0.45, 0, 255).astype(np.uint8)

    world_corners = [
        np.array([-0.48, 0.35, TABLE_HEIGHT], dtype=float),
        np.array([0.48, 0.35, TABLE_HEIGHT], dtype=float),
        np.array([0.48, 0.97, TABLE_HEIGHT], dtype=float),
        np.array([-0.48, 0.97, TABLE_HEIGHT], dtype=float),
    ]

    pts: list[tuple[int, int]] = []
    for corner in world_corners:
        p = project(corner, cam)
        if p is None:
            pts = []
            break
        pts.append((p[0], p[1]))

    if len(pts) != 4:
        pts = [(55, 305), (765, 305), (805, 480), (15, 480)]

    pts_array = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts_array], (62, 92, 138))

    for i in range(1, 8):
        yy = int(305 + i * 22)
        cv2.line(img, (35, yy), (WIDTH - 35, yy), (75, 105, 150), 1)

    cv2.polylines(img, [pts_array], True, (35, 45, 65), 2)

    for bin_obj in bins:
        draw_bin(img, bin_obj, cam)

    return img


def draw_shadow(
    img: np.ndarray,
    center: tuple[int, int],
    axes: tuple[int, int],
    opacity: float,
) -> None:
    cx, cy = center
    ax, ay = axes

    x1 = max(0, cx - ax)
    x2 = min(WIDTH, cx + ax)
    y1 = max(0, cy - ay)
    y2 = min(HEIGHT, cy + ay)

    if x1 >= x2 or y1 >= y2:
        return

    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.ellipse(overlay, (cx - x1, cy - y1), (ax, ay), 0, 0, 360, (0, 0, 0), -1)
    img[y1:y2, x1:x2] = cv2.addWeighted(overlay, float(opacity), roi, 1.0 - float(opacity), 0.0)


def composite_bgra(
    canvas: np.ndarray,
    sprite: np.ndarray,
    x: int,
    y: int,
) -> None:
    sh, sw = sprite.shape[:2]
    ch, cw = canvas.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(cw, x + sw)
    y2 = min(ch, y + sh)

    if x1 >= x2 or y1 >= y2:
        return

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    roi = canvas[y1:y2, x1:x2]
    patch = sprite[sy1:sy2, sx1:sx2]

    alpha = patch[:, :, 3:4].astype(float) / 255.0
    rgb = patch[:, :, :3].astype(float)

    roi[:] = (roi * (1.0 - alpha) + rgb * alpha).astype(np.uint8)


def resize_sprite(sprite: np.ndarray, target_h: int) -> np.ndarray:
    sh, sw = sprite.shape[:2]
    target_h = max(24, int(target_h))
    target_w = max(16, int(sw * target_h / max(1, sh)))
    interp = cv2.INTER_AREA if target_h < sh else cv2.INTER_LINEAR
    return cv2.resize(sprite, (target_w, target_h), interpolation=interp)


def draw_sprite_contour(
    img: np.ndarray,
    sprite: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    alpha = sprite[:, :, 3]
    _, mask = cv2.threshold(alpha, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 40:
            continue
        shifted = c + np.array([[[x, y]]], dtype=np.int32)
        cv2.drawContours(img, [shifted], -1, color, 2)


def draw_detection_overlays(
    img: np.ndarray,
    objects: list[SimObject],
    placements: list[tuple[int, int, int, int, int]],
    target_idx: int,
) -> None:
    for idx, x, y, w, h in placements:
        obj = objects[idx]
        if not obj.detected:
            continue

        pad = 6
        if idx == target_idx:
            color = (0, 230, 0)
        elif obj.confidence < 0.55:
            color = (0, 80, 255)
        else:
            color = (0, 190, 255)

        pt1 = (x - pad, y - pad)
        pt2 = (x + w + pad, y + h + pad)

        cv2.rectangle(img, pt1, pt2, color, 3 if idx == target_idx else 2)

        label = f"{obj.name} {obj.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(img, pt1, (pt1[0] + tw + 5, pt1[1] - th - 7), color, -1)
        cv2.putText(
            img,
            label,
            (pt1[0] + 2, pt1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def draw_pose_overlay(img: np.ndarray, obj: SimObject, cam: Camera) -> None:
    est = project(obj.est_base, cam)
    true = project(obj.base_pos, cam)
    if est is None or true is None:
        return

    cv2.circle(img, est[:2], 6, (0, 255, 255), 2)
    cv2.circle(img, true[:2], 4, (255, 255, 255), 1)
    cv2.line(img, est[:2], true[:2], (0, 255, 255), 1)

    err_cm = float(np.linalg.norm(obj.est_base - obj.base_pos) * 100.0)
    cv2.putText(
        img,
        f"est err {err_cm:.1f} cm",
        (est[0] + 9, max(12, est[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_grasp_candidates(
    img: np.ndarray,
    obj: SimObject,
    cam: Camera,
    selected_idx: int,
) -> None:
    proj = project(obj.est_base, cam)
    if proj is None:
        return

    u, v, depth = proj
    s = max(16, int(obj.display_h * cam.K[0, 0] / max(depth, 0.1)))
    radius = max(20, int(s * 0.58))

    for i in range(8):
        angle = i * math.pi / 4.0
        x = u + int(radius * math.cos(angle))
        y = v - int(radius * 0.45 * math.sin(angle))
        selected = i == selected_idx
        color = (0, 255, 0) if selected else (0, 180, 255)
        cv2.circle(img, (x, y), 6 if selected else 3, color, -1)
        cv2.arrowedLine(img, (x, y), (u, v - int(s * 0.35)), color, 1, tipLength=0.25)


def draw_robot_arm(
    img: np.ndarray,
    ee_pos: np.ndarray,
    cam: Camera,
    arm_side: str,
    gripper_open: float,
) -> None:
    bases = {
        "left": np.array([-0.24, 0.04, 1.00], dtype=float),
        "right": np.array([0.24, 0.04, 1.00], dtype=float),
    }

    base_proj = project(bases[arm_side], cam)
    ee_proj = project(ee_pos, cam)

    if base_proj is None:
        base = np.array([230, 475] if arm_side == "left" else [590, 475], dtype=float)
    else:
        base = np.array(base_proj[:2], dtype=float)

    if ee_proj is None:
        target = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
    else:
        target = np.array(ee_proj[:2], dtype=float)

    l1, l2 = 95.0, 118.0
    dx = float(target[0] - base[0])
    dy = float(target[1] - base[1])
    raw_d = math.hypot(dx, dy)

    if raw_d < 1e-3:
        target_clamped = (int(base[0]), int(base[1]))
        angle = 0.0
        d = abs(l1 - l2) + 2.0
    else:
        d = min(max(raw_d, abs(l1 - l2) + 2.0), l1 + l2 - 2.0)
        scale = d / raw_d
        target_clamped = (int(base[0] + dx * scale), int(base[1] + dy * scale))
        angle = math.atan2(dy, dx)

    cos_phi = (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d) if d > 1e-6 else 1.0
    phi = math.acos(float(np.clip(cos_phi, -1.0, 1.0)))
    flip = 1.0 if arm_side == "left" else -1.0

    elbow = (
        int(base[0] + l1 * math.cos(angle + flip * phi)),
        int(base[1] + l1 * math.sin(angle + flip * phi)),
    )

    color = (70, 160, 235) if arm_side == "left" else (235, 160, 70)
    base_pt = (int(base[0]), int(base[1]))

    cv2.line(img, base_pt, elbow, color, 13, cv2.LINE_AA)
    cv2.line(img, elbow, target_clamped, color, 11, cv2.LINE_AA)
    cv2.circle(img, base_pt, 10, (40, 40, 40), -1)
    cv2.circle(img, elbow, 8, (40, 40, 40), -1)

    target_np = np.array(target_clamped, dtype=float)
    elbow_np = np.array(elbow, dtype=float)
    dirv = target_np - elbow_np
    dir_norm = float(np.linalg.norm(dirv))
    if dir_norm < 1e-6:
        dirv = np.array([0.0, -1.0], dtype=float)
    else:
        dirv = dirv / dir_norm

    perp = np.array([-dirv[1], dirv[0]], dtype=float)
    open_px = 5.0 + float(gripper_open) * 15.0
    finger_len = 20.0
    finger_base = target_np - dirv * 4.0

    for sign in (-1.0, 1.0):
        p1 = finger_base + perp * open_px * sign
        p2 = p1 + dirv * finger_len
        cv2.line(
            img,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (30, 30, 30),
            3,
            cv2.LINE_AA,
        )

    cv2.circle(img, target_clamped, 4, (0, 0, 0), -1)


def draw_trail(img: np.ndarray, trail: list[np.ndarray], cam: Camera) -> None:
    pts: list[tuple[int, int]] = []
    for p in trail:
        proj = project(p, cam)
        if proj is not None:
            pts.append((proj[0], proj[1]))

    if len(pts) >= 2:
        cv2.polylines(
            img,
            [np.array(pts, dtype=np.int32)],
            False,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_hud(img: np.ndarray, info: dict[str, Any]) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, 68), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, HEIGHT - 38), (WIDTH, HEIGHT), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.62, img, 0.38, 0.0)

    cv2.putText(
        img,
        f"{info['cell']} | {info['robot']}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"RUN {info['run_id']}",
        (10, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (150, 180, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"PHASE: {info['phase']}",
        (WIDTH // 2 - 90, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"STATE: {info['state']}",
        (WIDTH // 2 - 90, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 255, 180) if info["state"] == "RUNNING" else (255, 220, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"t={info['time_s']:.2f}s  f={info['frame']}/{info['total']}",
        (WIDTH - 205, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"CMD: {info['command']}"[:62],
        (WIDTH - 390, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    bottom = (
        f"det {info['det_conf']:.2f} | loc {info['loc_err']:.1f} cm | "
        f"grasp {info['grasp_score']:.2f} | grip {info['gripper_width']*1000:.0f} mm | "
        f"force {info['contact_force']:.1f} N | {info['arm']} arm | "
        f"{info['target']} -> {info['bin']} | alarm {info['alarm']}"
    )
    cv2.putText(
        img,
        bottom[:118],
        (8, HEIGHT - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def draw_scene(
    base: np.ndarray,
    objects: list[SimObject],
    target_idx: int,
    cam: Camera,
    ee: np.ndarray,
    arm_side: str,
    gripper_open: float,
    info: dict[str, Any],
    trail: list[np.ndarray],
) -> np.ndarray:
    img = base.copy()

    entries: list[tuple[float, int, tuple[int, int, float]]] = []
    for idx, obj in enumerate(objects):
        proj = project(obj.base_pos, cam)
        if proj is not None:
            entries.append((proj[2], idx, proj))

    entries.sort(key=lambda item: item[0], reverse=True)

    placements: list[tuple[int, int, int, int, int]] = []

    for depth, idx, proj in entries:
        obj = objects[idx]
        u, v, _ = proj

        pixel_h = int(obj.display_h * cam.K[0, 0] / max(depth, 0.1))
        sprite = resize_sprite(obj.sprite, pixel_h)
        sw, sh = sprite.shape[1], sprite.shape[0]

        x = u - sw // 2
        y = v - sh

        on_table = obj.base_pos[2] <= TABLE_HEIGHT + 0.04
        if on_table:
            draw_shadow(
                img,
                (u, v),
                (max(8, sw // 2), max(4, int(sh * 0.12))),
                0.26,
            )

        composite_bgra(img, sprite, x, y)

        if obj.detected:
            contour_color = (0, 230, 0) if idx == target_idx else (0, 190, 255)
            draw_sprite_contour(img, sprite, x, y, contour_color)

        placements.append((idx, x, y, sw, sh))

    draw_detection_overlays(img, objects, placements, target_idx)
    draw_trail(img, trail, cam)

    phase = str(info["phase"])
    if phase in {"LOCALIZE", "PLAN", "VERIFY", "RE_LOCALIZE", "RE_APPROACH"}:
        draw_pose_overlay(img, objects[target_idx], cam)

    if phase in {"PLAN", "APPROACH", "RE_APPROACH"}:
        selected_idx = 4 if arm_side == "left" else 0
        draw_grasp_candidates(img, objects[target_idx], cam, selected_idx)

    draw_robot_arm(img, ee, cam, arm_side, gripper_open)
    draw_hud(img, info)
    return img


def encode_jpeg_b64(frame: np.ndarray, quality: int = 70) -> Optional[str]:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_live_html(frames_b64: list[str], frame_meta: list[dict[str, Any]], fps: int, uid: str) -> str:
    if not frames_b64:
        return (
            "<div style='padding:16px; background:#0b1020; color:#fff; "
            "border-radius:12px; font-family:sans-serif;'>No frames generated.</div>"
        )

    payload = {"frames": frames_b64, "meta": frame_meta}
    base_interval = max(25, int(1000 / max(1, fps)))

    html = (
        PLAYER_HTML.replace("__UID__", uid)
        .replace("__FIRST_FRAME__", frames_b64[0])
        .replace("__PAYLOAD_JSON__", json.dumps(payload))
        .replace("__BASE_INTERVAL__", str(base_interval))
    )
    return html


def open_video_writer(tmpdir: str) -> tuple[Any, str, str, bool, str]:
    candidates = [
        ("demo.mp4", "avc1", "video/mp4", True, "mp4"),
        ("demo.mp4", "mp4v", "video/mp4", False, "mp4"),
        ("demo.avi", "XVID", "video/x-msvideo", False, "avi"),
    ]

    for filename, codec, mime, playable, ext in candidates:
        path = os.path.join(tmpdir, filename)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))
        if writer.isOpened():
            return writer, path, mime, playable, ext

    return None, "", "", False, "mp4"


def add_event(
    events: list[dict[str, Any]],
    time_s: float,
    frame: int,
    module: str,
    severity: str,
    code: str,
    message: str,
) -> None:
    events.append(
        {
            "time_s": round(float(time_s), 3),
            "frame": int(frame),
            "module": module,
            "severity": severity,
            "code": code,
            "message": message,
        }
    )


def phase_module(phase: str) -> str:
    if phase == "SEARCH":
        return "perception"
    if phase in {"LOCALIZE", "RE_LOCALIZE"}:
        return "calib"
    if phase == "PLAN":
        return "grasp"
    if phase in {"APPROACH", "RE_APPROACH", "TRANSFER"}:
        return "planning"
    if phase in {"GRASP", "RE_GRASP", "PLACE", "SLIP"}:
        return "exec"
    if phase in {"VERIFY", "RE_LIFT"}:
        return "verify"
    return "task"


def render_demo(
    facility: str,
    cell: str,
    robot: str,
    scenario: str,
    command: str,
    target_name: str,
    target_bin: str,
    seed: int,
    speed_scale: float,
    depth_noise_m: float,
    inject_failure: bool,
    sprites: dict[str, np.ndarray],
) -> dict[str, Any]:
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    sc = scenario_config(scenario)
    effective_inject = bool(inject_failure or sc.get("force_inject", False))
    effective_noise = float(depth_noise_m) * float(sc["noise_mult"])

    rng = np.random.default_rng(int(seed))
    objects, bins, target_idx = make_scene(
        rng=rng,
        target_name=target_name,
        target_bin=target_bin,
        noise_m=float(depth_noise_m),
        sprites=sprites,
        sc=sc,
    )

    cam = make_camera()
    base = make_base_image(cam, bins)

    target_obj = objects[target_idx]
    bin_obj = next(b for b in bins if b.key == target_bin)
    arm_side = "left" if target_obj.est_base[0] < 0.0 else "right"

    hold_offset = np.array([0.0, 0.0, target_obj.display_h * 0.55], dtype=float)

    home = np.array(
        [-0.07 if arm_side == "left" else 0.07, 0.16, 1.08],
        dtype=float,
    )

    grasp_ee = target_obj.est_base + np.array([0.0, 0.0, target_obj.display_h * 0.55], dtype=float)
    pre_ee = grasp_ee + np.array([0.0, 0.0, 0.10], dtype=float)
    lift_ee = grasp_ee + np.array([0.0, 0.0, 0.12], dtype=float)

    place_base = bin_obj.pos + np.array([0.0, 0.0, 0.025], dtype=float)
    place_ee = place_base + np.array([0.0, 0.0, target_obj.display_h * 0.55], dtype=float)
    place_hover_ee = place_ee + np.array([0.0, 0.0, 0.12], dtype=float)
    retreat_ee = place_hover_ee + np.array([0.0, 0.0, 0.06], dtype=float)

    mid_ee = lerp_vec(lift_ee, place_hover_ee, 0.45)

    dropped_object_base = mid_ee - hold_offset
    drop_land_base = np.array(
        [
            float(np.clip(dropped_object_base[0], -0.28, 0.28)),
            float(np.clip(dropped_object_base[1], 0.50, 0.78)),
            TABLE_HEIGHT,
        ],
        dtype=float,
    )
    drop_grasp_ee = drop_land_base + np.array([0.0, 0.0, target_obj.display_h * 0.55], dtype=float)
    drop_lift_ee = drop_grasp_ee + np.array([0.0, 0.0, 0.12], dtype=float)

    speed = float(np.clip(speed_scale, 0.6, 2.5))

    def dur(frames: int) -> int:
        return max(2, int(round(float(frames) / speed)))

    segments = [
        Segment("SEARCH", dur(10), home),
        Segment("LOCALIZE", dur(7), home),
        Segment("PLAN", dur(8), home),
        Segment("APPROACH", dur(16), pre_ee),
        Segment("GRASP", dur(6), grasp_ee, "grasp", 0.65),
        Segment("VERIFY", dur(5), lift_ee),
    ]

    if effective_inject:
        segments.extend(
            [
                Segment("TRANSFER", dur(12), mid_ee),
                Segment("SLIP", dur(6), mid_ee, "slip", 0.25),
                Segment("RE_LOCALIZE", dur(5), mid_ee),
                Segment("RE_APPROACH", dur(12), drop_grasp_ee),
                Segment("RE_GRASP", dur(6), drop_grasp_ee, "grasp", 0.65),
                Segment("RE_LIFT", dur(8), drop_lift_ee),
                Segment("TRANSFER", dur(16), place_hover_ee),
            ]
        )
    else:
        segments.append(Segment("TRANSFER", dur(20), place_hover_ee))

    segments.extend(
        [
            Segment("PLACE", dur(6), place_ee, "release", 0.55),
            Segment("RETREAT", dur(8), retreat_ee),
            Segment("COMPLETE", dur(3), retreat_ee),
        ]
    )

    total_frames = sum(seg.duration for seg in segments)
    if total_frames > MAX_FRAMES:
        factor = float(MAX_FRAMES) / float(total_frames)
        for seg in segments:
            seg.duration = max(2, int(round(seg.duration * factor)))
        total_frames = sum(seg.duration for seg in segments)

    loc_err_cm = float(np.linalg.norm(target_obj.est_base - target_obj.true_base) * 100.0)
    current_loc_err_cm = loc_err_cm
    grasp_score = float(
        np.clip(
            0.88
            - float(effective_noise) * 4.5
            + (0.02 if effective_inject else 0.0)
            + float(sc.get("conf_bias", 0.0)) * 0.25,
            0.20,
            0.99,
        )
    )

    frames_b64: list[str] = []
    frame_meta: list[dict[str, Any]] = []
    timeseries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    final_frame = base.copy()

    video_bytes: Optional[bytes] = None
    video_mime = ""
    video_playable = False
    video_ext = "mp4"

    frame_idx = 0
    ee = home.copy()
    gripper_open = 1.0
    gripper_target = 1.0
    held_idx: Optional[int] = None
    falling = False
    fall_start = np.zeros(3, dtype=float)
    fall_end = np.zeros(3, dtype=float)
    fall_frames = 1
    fall_t = 0
    ee_trail: list[np.ndarray] = []
    recovery_attempts = 0
    slip_detected = False

    add_event(
        events,
        0.0,
        0,
        "task",
        "INFO",
        "RUN_STARTED",
        f"Run started: {facility} / {cell} / {robot}",
    )
    add_event(
        events,
        0.0,
        0,
        "task",
        "INFO",
        "COMMAND_ACCEPTED",
        f"Command accepted: {command}",
    )
    add_event(
        events,
        0.0,
        0,
        "perception",
        "INFO",
        "TARGET_DETECTED",
        f"Target detected: {target_obj.name}, confidence={target_obj.confidence:.2f}",
    )

    if scenario == "Low light / noisy depth":
        add_event(
            events,
            0.0,
            0,
            "perception",
            "WARN",
            "LOW_LIGHT_CONDITION",
            "Low-light scenario increases depth noise and reduces detection confidence.",
        )

    if scenario == "Transparent cup risk" and target_name == "cup":
        add_event(
            events,
            0.0,
            0,
            "perception",
            "WARN",
            "TRANSPARENT_DEPTH_RISK",
            "Transparent/reflective cup may produce unstable depth localization.",
        )

    if target_obj.confidence < 0.55:
        add_event(
            events,
            0.0,
            0,
            "perception",
            "WARN",
            "LOW_DETECTION_CONFIDENCE",
            f"Detection confidence is low: {target_obj.confidence:.2f}",
        )

    if current_loc_err_cm > 4.0:
        add_event(
            events,
            0.0,
            0,
            "calib",
            "WARN",
            "HIGH_LOCALIZATION_ERROR",
            f"Initial 3D localization error is high: {current_loc_err_cm:.1f} cm",
        )

    with tempfile.TemporaryDirectory(prefix="p02_industrial_") as tmpdir:
        writer, video_path, candidate_mime, candidate_playable, candidate_ext = (
            open_video_writer(tmpdir)
        )

        try:
            for seg in segments:
                if seg.phase == "RE_LOCALIZE":
                    recovery_attempts += 1
                    target_obj.est_base = target_obj.base_pos + rng.normal(
                        0.0,
                        max(float(effective_noise), 1e-4) * 0.75,
                        3,
                    )
                    target_obj.est_base = clamp_xy(target_obj.est_base)
                    target_obj.est_base[2] = TABLE_HEIGHT
                    target_obj.confidence = float(
                        np.clip(target_obj.confidence + rng.normal(0.0, 0.02), 0.35, 0.99)
                    )
                    target_obj.detected = target_obj.confidence >= 0.40
                    current_loc_err_cm = float(
                        np.linalg.norm(target_obj.est_base - target_obj.base_pos) * 100.0
                    )

                add_event(
                    events,
                    float(frame_idx) / float(FPS),
                    frame_idx,
                    phase_module(seg.phase),
                    "INFO",
                    f"ENTER_{seg.phase}",
                    f"State machine entered {seg.phase}.",
                )

                trigger_frame = int(seg.duration * seg.action_at) if seg.action else -1
                start_ee = ee.copy()
                denom = max(1, seg.duration - 1)

                for i in range(seg.duration):
                    if seg.action and i == trigger_frame:
                        if seg.action == "grasp":
                            held_idx = target_idx
                            target_obj.held = True
                            target_obj.placed = False
                            gripper_target = 0.15
                            add_event(
                                events,
                                float(frame_idx) / float(FPS),
                                frame_idx,
                                "exec",
                                "INFO",
                                "GRIPPER_CLOSED",
                                "Gripper closed; object attached to end-effector.",
                            )
                        elif seg.action == "release":
                            held_idx = None
                            target_obj.held = False
                            target_obj.placed = True
                            target_obj.base_pos = place_base.copy()
                            gripper_target = 1.0
                            add_event(
                                events,
                                float(frame_idx) / float(FPS),
                                frame_idx,
                                "verify",
                                "INFO",
                                "PLACEMENT_VERIFIED",
                                "Object released; bin placement verified by simulator ground truth.",
                            )
                        elif seg.action == "slip":
                            held_idx = None
                            target_obj.held = False
                            falling = True
                            slip_detected = True
                            fall_start = target_obj.base_pos.copy()
                            fall_end = drop_land_base.copy()
                            fall_frames = max(1, seg.duration - i)
                            fall_t = 0
                            add_event(
                                events,
                                float(frame_idx) / float(FPS),
                                frame_idx,
                                "exec",
                                "ERROR",
                                "GRASP_SLIPPAGE",
                                "Injected failure: grasp slipped during transfer.",
                            )

                    alpha = ease(float(i) / float(denom))
                    ee = lerp_vec(start_ee, seg.waypoint, alpha)
                    gripper_open += (gripper_target - gripper_open) * 0.35

                    if falling:
                        fall_t += 1
                        target_obj.base_pos = lerp_vec(
                            fall_start,
                            fall_end,
                            min(1.0, float(fall_t) / float(fall_frames)),
                        )
                        if fall_t >= fall_frames:
                            falling = False
                            add_event(
                                events,
                                float(frame_idx) / float(FPS),
                                frame_idx,
                                "perception",
                                "INFO",
                                "OBJECT_REDETECTED",
                                "Dropped object re-detected on table.",
                            )
                    elif held_idx == target_idx:
                        target_obj.base_pos = ee - hold_offset

                    ee_trail.append(ee.copy())
                    if len(ee_trail) > 85:
                        ee_trail.pop(0)

                    if held_idx == target_idx and seg.phase != "SLIP" and not falling:
                        contact_force = float(np.clip(rng.normal(18.0, 1.2), 0.0, 35.0))
                    else:
                        contact_force = 0.0

                    gripper_width_m = 0.015 + float(gripper_open) * 0.065
                    clearance_mm = float(
                        np.clip(28.0 - float(effective_noise) * 900.0 + rng.normal(0.0, 1.2), 3.0, 80.0)
                    )

                    timeseries.append(
                        {
                            "time_s": round(float(frame_idx) / float(FPS), 3),
                            "frame": int(frame_idx),
                            "phase": seg.phase,
                            "ee_z_m": round(float(ee[2]), 4),
                            "object_z_m": round(float(target_obj.base_pos[2]), 4),
                            "gripper_open": round(float(gripper_open), 4),
                            "gripper_width_m": round(float(gripper_width_m), 4),
                            "confidence": round(float(target_obj.confidence), 4),
                            "loc_err_cm": round(float(current_loc_err_cm), 3),
                            "grasp_score": round(float(grasp_score), 4),
                            "contact_force_n": round(float(contact_force), 3),
                            "clearance_mm": round(float(clearance_mm), 3),
                            "held": bool(held_idx == target_idx),
                            "falling": bool(falling),
                            "placed": bool(target_obj.placed),
                        }
                    )

                    frame_meta.append(
                        {
                            "phase": seg.phase,
                            "time": round(float(frame_idx) / float(FPS), 3),
                        }
                    )

                    alarm = "NONE"
                    if slip_detected and not target_obj.placed:
                        alarm = "SLIP"
                    elif target_obj.confidence < 0.55:
                        alarm = "LOW_DET"
                    elif current_loc_err_cm > 5.0:
                        alarm = "LOC_ERR"

                    info = {
                        "facility": facility,
                        "cell": cell,
                        "robot": robot,
                        "run_id": run_id,
                        "phase": seg.phase,
                        "command": command,
                        "frame": frame_idx,
                        "total": total_frames,
                        "time_s": float(frame_idx) / float(FPS),
                        "state": "RUNNING",
                        "alarm": alarm,
                        "arm": arm_side,
                        "target": target_name,
                        "bin": BIN_LABELS[target_bin],
                        "det_conf": float(target_obj.confidence),
                        "loc_err": float(current_loc_err_cm),
                        "grasp_score": float(grasp_score),
                        "collision": "OK",
                        "gripper_width": float(gripper_width_m),
                        "contact_force": float(contact_force),
                    }

                    final_frame = draw_scene(
                        base=base,
                        objects=objects,
                        target_idx=target_idx,
                        cam=cam,
                        ee=ee,
                        arm_side=arm_side,
                        gripper_open=gripper_open,
                        info=info,
                        trail=ee_trail,
                    )

                    b64 = encode_jpeg_b64(final_frame, quality=68)
                    if b64 is not None:
                        frames_b64.append(b64)

                    if writer is not None:
                        writer.write(final_frame)

                    frame_idx += 1
        finally:
            if writer is not None:
                writer.release()
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    with open(video_path, "rb") as fh:
                        video_bytes = fh.read()
                    video_mime = candidate_mime
                    video_playable = candidate_playable
                    video_ext = candidate_ext

    if not frames_b64:
        b64 = encode_jpeg_b64(final_frame, quality=75)
        if b64 is not None:
            frames_b64 = [b64]
            frame_meta = [{"phase": "COMPLETE", "time": 0.0}]

    uid = uuid.uuid4().hex[:10]
    html = build_live_html(frames_b64, frame_meta, FPS, uid)

    success = bool(target_obj.placed)
    terminal_state = "SUCCESS" if success else "FAILURE"

    if success:
        placement_error_mm = float(
            np.clip(
                rng.normal(
                    9.0 + float(sc.get("place_bias_mm", 0.0)) + float(effective_noise) * 900.0,
                    3.8,
                ),
                1.0,
                120.0,
            )
        )
    else:
        placement_error_mm = 999.0

    if slip_detected and success:
        add_event(
            events,
            float(frame_idx) / float(FPS),
            frame_idx,
            "task",
            "INFO",
            "RECOVERY_SUCCESS",
            "Recovery succeeded after injected slippage.",
        )
    elif not success:
        add_event(
            events,
            float(frame_idx) / float(FPS),
            frame_idx,
            "task",
            "ERROR",
            "TASK_FAILED",
            "Terminal failure: object was not verified in target bin.",
        )
    else:
        add_event(
            events,
            float(frame_idx) / float(FPS),
            frame_idx,
            "task",
            "INFO",
            "TASK_SUCCESS",
            "Terminal success: correct object verified in target bin.",
        )

    phase_counts: dict[str, int] = {}
    for m in frame_meta:
        phase_counts[m["phase"]] = phase_counts.get(m["phase"], 0) + 1
    phase_durations = [(p, float(c) / float(FPS)) for p, c in phase_counts.items()]

    perception_health = float(np.clip(target_obj.confidence * 100.0 - (8.0 if target_obj.confidence < 0.55 else 0.0), 0.0, 100.0))
    localization_health = float(np.clip(100.0 - max(0.0, current_loc_err_cm - 1.0) * 8.0, 0.0, 100.0))
    grasp_health = float(np.clip(grasp_score * 100.0, 0.0, 100.0))
    planning_health = 98.0
    execution_health = 100.0 if success and not slip_detected else (82.0 if success else 35.0)
    verification_health = 100.0 if success else 0.0
    recovery_health = 100.0 if not effective_inject else (100.0 if success else 0.0)

    subsystem_health = [
        {"label": "Perception", "value": perception_health, "color": "#22c55e"},
        {"label": "Localization", "value": localization_health, "color": "#38bdf8"},
        {"label": "Grasp scoring", "value": grasp_health, "color": "#a78bfa"},
        {"label": "Motion planning", "value": planning_health, "color": "#f59e0b"},
        {"label": "Execution", "value": execution_health, "color": "#fb7185"},
        {"label": "Verification", "value": verification_health, "color": "#34d399"},
        {"label": "Recovery", "value": recovery_health, "color": "#60a5fa"},
    ]

    failure_counts: dict[str, int] = {}
    for e in events:
        if e["severity"] in ("WARN", "ERROR", "CRITICAL"):
            failure_counts[e["module"]] = failure_counts.get(e["module"], 0) + 1

    failure_breakdown = [
        {"Stage": module, "Warnings/Errors": count}
        for module, count in sorted(failure_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]

    metrics = {
        "run_id": run_id,
        "facility": facility,
        "cell": cell,
        "robot": robot,
        "scenario": scenario,
        "command": command,
        "target_object": target_name,
        "target_bin": BIN_LABELS[target_bin],
        "detection_confidence": float(target_obj.confidence),
        "localization_error_cm": float(current_loc_err_cm),
        "grasp_score": float(grasp_score),
        "ik_success": True,
        "collision_free_execution_rate": 1.0,
        "grasp_success": success,
        "placement_error_mm": float(placement_error_mm),
        "task_time_s": float(frame_idx) / float(FPS),
        "injected_failure": bool(effective_inject),
        "slip_detected": bool(slip_detected),
        "recovery_attempts": int(recovery_attempts),
        "recovery_success": bool(effective_inject and success),
        "terminal_state": terminal_state,
        "frames_rendered": int(frame_idx),
        "fps": int(FPS),
        "mean_contact_force_n": float(
            np.mean([row["contact_force_n"] for row in timeseries if row["contact_force_n"] > 0.0])
            if any(row["contact_force_n"] > 0.0 for row in timeseries)
            else 0.0
        ),
        "max_localization_error_cm": float(max((row["loc_err_cm"] for row in timeseries), default=current_loc_err_cm)),
    }

    config_snapshot = {
        "run_id": run_id,
        "facility": facility,
        "cell": cell,
        "robot": robot,
        "scenario": scenario,
        "scenario_description": sc.get("description", ""),
        "command": command,
        "target_object": target_name,
        "target_bin": target_bin,
        "seed": int(seed),
        "speed_scale": float(speed_scale),
        "depth_noise_m": float(depth_noise_m),
        "effective_noise_m": float(effective_noise),
        "inject_failure_requested": bool(inject_failure),
        "inject_failure_effective": bool(effective_inject),
        "image_source": "uploaded/assets/fallback sprites",
    }

    return {
        "run_id": run_id,
        "html": html,
        "frames_b64": frames_b64,
        "frame_meta": frame_meta,
        "timeseries": timeseries,
        "events": events,
        "metrics": metrics,
        "config_snapshot": config_snapshot,
        "phase_durations": phase_durations,
        "phases": [seg.phase for seg in segments],
        "subsystem_health": subsystem_health,
        "failure_breakdown": failure_breakdown,
        "video_bytes": video_bytes,
        "video_mime": video_mime,
        "video_playable": video_playable,
        "video_ext": video_ext,
        "final_frame": final_frame,
        "final_frame_rgb": cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB),
    }


def svg_metric_chart(
    timeseries: list[dict[str, Any]],
    key: str,
    title: str,
    color: str,
    unit: str = "",
    width: int = 780,
    height: int = 220,
) -> str:
    if not timeseries:
        return "<div class='empty-note'>No telemetry available.</div>"

    xs = [float(row.get("time_s", i / float(FPS))) for i, row in enumerate(timeseries)]
    ys = [float(row.get(key, 0.0)) for row in timeseries]

    xmin, xmax = min(xs), max(xs)
    if xmax - xmin < 1e-6:
        xmax = xmin + 1.0

    ymin, ymax = min(ys), max(ys)
    if ymax - ymin < 1e-6:
        ymax = ymin + 1.0

    pad_l, pad_r, pad_t, pad_b = 58, 22, 28, 34

    def sx(x: float) -> float:
        return pad_l + ((x - xmin) / (xmax - xmin)) * (width - pad_l - pad_r)

    def sy(y: float) -> float:
        return height - pad_b - ((y - ymin) / (ymax - ymin)) * (height - pad_t - pad_b)

    grid = []
    for i in range(5):
        yval = ymin + i * (ymax - ymin) / 4.0
        yy = sy(yval)
        grid.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" stroke="#334155" stroke-width="1"/>'
            f'<text x="6" y="{yy+4:.1f}" fill="#94a3b8" font-size="11" font-family="sans-serif">{yval:.2f}</text>'
        )

    points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    last = ys[-1]

    return f"""
<div class="chart-card">
  <div class="chart-title">{title}</div>
  <svg width="100%" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#0b1020" rx="10" />
    {''.join(grid)}
    <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />
    <text x="{width-pad_r}" y="18" fill="#e5e7eb" font-size="12" font-family="sans-serif" text-anchor="end">
      last: {last:.2f}{unit}
    </text>
    <text x="{pad_l}" y="{height-8}" fill="#94a3b8" font-size="11" font-family="sans-serif">time (s)</text>
  </svg>
</div>
"""


def svg_bar_chart(
    items: list[tuple[str, float]],
    title: str,
    width: int = 780,
    bar_height: int = 18,
) -> str:
    if not items:
        return "<div class='empty-note'>No phase data available.</div>"

    max_val = max(value for _, value in items) or 1.0
    height = len(items) * (bar_height + 10) + 42
    rows = []

    for i, (label, value) in enumerate(items):
        y = 28 + i * (bar_height + 10)
        w = max(3.0, (value / max_val) * (width - 210))
        rows.append(
            f'<text x="8" y="{y+13}" fill="#cbd5e1" font-size="12" font-family="sans-serif">{label}</text>'
            f'<rect x="145" y="{y}" width="{w:.1f}" height="{bar_height}" fill="#38bdf8" rx="5" />'
            f'<text x="{150+w:.1f}" y="{y+13}" fill="#e5e7eb" font-size="12" font-family="sans-serif">{value:.2f}s</text>'
        )

    return f"""
<div class="chart-card">
  <div class="chart-title">{title}</div>
  <svg width="100%" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#0b1020" rx="10" />
    {''.join(rows)}
  </svg>
</div>
"""


def health_html(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        value = float(row["value"])
        color = str(row["color"])
        out.append(
            f"""
            <div class="health-row">
              <div class="health-label">{row['label']}</div>
              <div class="health-track">
                <div class="health-fill" style="width:{value:.1f}%; background:{color};"></div>
              </div>
              <div class="health-value">{value:.0f}%</div>
            </div>
            """
        )
    return "".join(out)


def html_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def dict_list_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""

    headers = list(rows[0].keys())
    lines = [",".join(headers)]

    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, "")
            s = str(v).replace('"', '""')
            if "," in s or '"' in s or "\n" in s:
                s = f'"{s}"'
            vals.append(s)
        lines.append(",".join(vals))

    return "\n".join(lines)


def build_html_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    config = result["config_snapshot"]
    events = result["events"]
    final_b64 = encode_jpeg_b64(result["final_frame"], quality=82)

    metric_rows = "".join(
        f"<tr><td>{html_escape(k)}</td><td>{html_escape(v)}</td></tr>"
        for k, v in metrics.items()
    )

    config_rows = "".join(
        f"<tr><td>{html_escape(k)}</td><td>{html_escape(v)}</td></tr>"
        for k, v in config.items()
    )

    event_rows = "".join(
        f"<tr><td>{html_escape(e['time_s'])}</td><td>{html_escape(e['module'])}</td>"
        f"<td>{html_escape(e['severity'])}</td><td>{html_escape(e['code'])}</td>"
        f"<td>{html_escape(e['message'])}</td></tr>"
        for e in events
    )

    image_block = ""
    if final_b64:
        image_block = f'<h3>Final annotated frame</h3><img src="data:image/jpeg;base64,{final_b64}" style="max-width:100%; border:1px solid #ccc; border-radius:10px;" />'

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Industrial Manipulation Demo Report {html_escape(metrics['run_id'])}</title>
</head>
<body style="font-family:Arial,Helvetica,sans-serif; margin:24px; color:#111827;">
<h1>Autonomous Object Detection and Manipulation — Synthetic Demo Report</h1>
<p><strong>Run ID:</strong> {html_escape(metrics['run_id'])}</p>
<p><strong>Generated:</strong> {html_escape(time.strftime('%Y-%m-%d %H:%M:%S'))}</p>
<p><strong>Notice:</strong> All results are from a local synthetic simulation. They are not physical-robot benchmark measurements.</p>

<h2>Metrics</h2>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
<tr><th>Key</th><th>Value</th></tr>
{metric_rows}
</table>

<h2>Configuration snapshot</h2>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
<tr><th>Key</th><th>Value</th></tr>
{config_rows}
</table>

{image_block}

<h2>Event log</h2>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%;">
<tr><th>Time (s)</th><th>Module</th><th>Severity</th><th>Code</th><th>Message</th></tr>
{event_rows}
</table>

</body>
</html>
"""


def init_session() -> None:
    defaults = {
        "facility": "Plant A",
        "cell": "CELL-ROBO-01",
        "robot": "HR-UB-0001",
        "scenario": "Standard",
        "command": DEFAULT_COMMAND,
        "valid_command": DEFAULT_COMMAND,
        "target_object": "cup",
        "target_bin": "bin_a",
        "command_error": "",
        "seed": 7,
        "speed": 1.0,
        "noise": 0.003,
        "inject": True,
        "auto": True,
        "nonce": 0,
        "applied_nonce": -1,
        "applied_control_key": None,
        "uploaded_sprites": {},
        "sprite_version": 0,
        "run_history": [],
        "last_error": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    parsed = parse_command(str(st.session_state["command"]))
    if parsed.ok:
        st.session_state["valid_command"] = str(st.session_state["command"])
        st.session_state["target_object"] = parsed.object_name
        st.session_state["target_bin"] = parsed.bin_key
        st.session_state["command_error"] = ""
    else:
        st.session_state["command_error"] = parsed.error


def set_command_callback(cmd: str) -> None:
    st.session_state["command"] = cmd


def bump_nonce_callback() -> None:
    st.session_state["nonce"] = int(st.session_state.get("nonce", 0)) + 1


def handle_upload(name: str, uploaded_file: Any) -> None:
    if uploaded_file is None:
        return

    signature = f"{uploaded_file.name}:{uploaded_file.size}"
    processed_key = f"processed_upload_{name}"

    if st.session_state.get(processed_key) == signature:
        return

    data = uploaded_file.getvalue()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None:
        st.error(f"Could not decode uploaded image for {name}.")
        return

    try:
        sprite = prepare_real_sprite(img)
    except Exception as exc:
        st.error(f"Could not process uploaded image for {name}: {exc}")
        return

    st.session_state["uploaded_sprites"][name] = sprite
    st.session_state[processed_key] = signature
    st.session_state["sprite_version"] = int(st.session_state.get("sprite_version", 0)) + 1


st.set_page_config(
    page_title="Industrial Autonomous Manipulation Demo",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
init_session()

with st.sidebar:
    st.header("🏭 Site configuration")

    st.selectbox(
        "Facility",
        options=["Plant A", "Plant B", "Warehouse North", "R&D Lab"],
        key="facility",
    )
    st.text_input("Cell ID", key="cell")
    st.text_input("Robot serial", key="robot")

    st.divider()
    st.header("🧪 Scenario")

    scenario = st.selectbox(
        "Process scenario",
        options=[
            "Standard",
            "High clutter",
            "Low light / noisy depth",
            "Transparent cup risk",
            "Tight bin placement",
            "Recovery stress test",
        ],
        key="scenario",
    )
    st.caption(scenario_config(scenario)["description"])

    st.divider()
    st.header("🗣️ Task command")

    st.text_input("Natural-language command", key="command")
    parsed = parse_command(str(st.session_state["command"]))

    if parsed.ok:
        st.success(f"Parsed: pick {parsed.object_name} → {BIN_LABELS[parsed.bin_key]}")
    else:
        st.error(st.session_state["command_error"])
        st.caption(f"Using last valid command: {st.session_state['valid_command']}")

    q1, q2 = st.columns(2)
    with q1:
        st.button(
            "Cup → Bin A",
            use_container_width=True,
            on_click=set_command_callback,
            args=("put the cup in the bin",),
        )
    with q2:
        st.button(
            "Bottle → Bin B",
            use_container_width=True,
            on_click=set_command_callback,
            args=("put the bottle in bin b",),
        )

    q3, q4 = st.columns(2)
    with q3:
        st.button(
            "Box → Bin A",
            use_container_width=True,
            on_click=set_command_callback,
            args=("place the box in left bin",),
        )
    with q4:
        st.button(
            "Cube → Bin B",
            use_container_width=True,
            on_click=set_command_callback,
            args=("move cube to right bin",),
        )

    st.divider()
    st.header("⚙️ Simulation controls")

    st.number_input("Scene seed", min_value=0, max_value=9999, step=1, key="seed")
    st.slider("Speed scale", min_value=0.75, max_value=2.0, step=0.05, key="speed")
    st.slider(
        "Depth noise std (m)",
        min_value=0.0,
        max_value=0.02,
        step=0.0005,
        key="noise",
    )
    st.checkbox("Inject slip failure", key="inject")
    st.checkbox("Auto-regenerate on control change", key="auto")

    st.divider()
    st.header("🖼️ Real object images")
    st.caption(
        "PNG with transparent background is best. JPG is accepted and the app will try "
        "to remove a white/light background. You can also place files in assets/."
    )

    for name in OBJECT_CLASSES:
        uploaded = st.file_uploader(
            f"{name.capitalize()} image",
            key=f"upload_{name}",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
        )
        handle_upload(name, uploaded)

    if st.button("Regenerate now", use_container_width=True, on_click=bump_nonce_callback):
        pass

control_params = (
    int(st.session_state["seed"]),
    str(st.session_state["valid_command"]),
    str(st.session_state["target_object"]),
    str(st.session_state["target_bin"]),
    float(st.session_state["speed"]),
    float(st.session_state["noise"]),
    bool(st.session_state["inject"]),
    str(st.session_state["scenario"]),
    str(st.session_state["facility"]),
    str(st.session_state["cell"]),
    str(st.session_state["robot"]),
    int(st.session_state["sprite_version"]),
)

control_key = (int(st.session_state["nonce"]),) + control_params

should_generate = False
if "result" not in st.session_state:
    should_generate = True
elif bool(st.session_state["auto"]):
    should_generate = st.session_state.get("applied_control_key") != control_key
else:
    should_generate = st.session_state.get("applied_nonce", -1) != int(st.session_state["nonce"])

if should_generate:
    try:
        with st.spinner("Generating industrial live simulation..."):
            result = render_demo(
                facility=str(st.session_state["facility"]),
                cell=str(st.session_state["cell"]),
                robot=str(st.session_state["robot"]),
                scenario=str(st.session_state["scenario"]),
                command=str(st.session_state["valid_command"]),
                target_name=str(st.session_state["target_object"]),
                target_bin=str(st.session_state["target_bin"]),
                seed=int(st.session_state["seed"]),
                speed_scale=float(st.session_state["speed"]),
                depth_noise_m=float(st.session_state["noise"]),
                inject_failure=bool(st.session_state["inject"]),
                sprites=current_sprites(),
            )
        st.session_state["result"] = result
        st.session_state["last_error"] = ""

        summary = {
            "run_id": result["run_id"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "facility": result["metrics"]["facility"],
            "cell": result["metrics"]["cell"],
            "robot": result["metrics"]["robot"],
            "scenario": result["metrics"]["scenario"],
            "command": result["metrics"]["command"],
            "target": result["metrics"]["target_object"],
            "bin": result["metrics"]["target_bin"],
            "terminal_state": result["metrics"]["terminal_state"],
            "detection_confidence": result["metrics"]["detection_confidence"],
            "localization_error_cm": result["metrics"]["localization_error_cm"],
            "grasp_score": result["metrics"]["grasp_score"],
            "placement_error_mm": result["metrics"]["placement_error_mm"],
            "task_time_s": result["metrics"]["task_time_s"],
            "injected_failure": result["metrics"]["injected_failure"],
            "recovery_success": result["metrics"]["recovery_success"],
        }
        st.session_state["run_history"].insert(0, summary)
        st.session_state["run_history"] = st.session_state["run_history"][:100]

    except Exception as exc:
        st.session_state["last_error"] = f"{type(exc).__name__}: {exc}"
        result = st.session_state.get("result")
    finally:
        st.session_state["applied_control_key"] = control_key
        st.session_state["applied_nonce"] = int(st.session_state["nonce"])
else:
    result = st.session_state.get("result")

if st.session_state.get("last_error"):
    st.error(f"Last generation error: {st.session_state['last_error']}")

if result is None:
    st.warning("No generated demo is currently available. Adjust controls or press Regenerate.")
    st.stop()

metrics = result["metrics"]
state_color = "pill-success" if metrics["terminal_state"] == "SUCCESS" else "pill-failure"

st.markdown(
    f"""
    <div class="header-card">
      <div style="display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; align-items:center;">
        <div>
          <div class="header-title">🏭 Industrial Autonomous Manipulation — Synthetic Live Demo</div>
          <div class="header-sub">
            {html_escape(metrics['facility'])} • {html_escape(metrics['cell'])} • {html_escape(metrics['robot'])} •
            Run {html_escape(metrics['run_id'])} • Scenario {html_escape(metrics['scenario'])}
          </div>
        </div>
        <div>
          <span class="pill {state_color}">{html_escape(metrics['terminal_state'])}</span>
          <span class="pill pill-info">{html_escape(metrics['target_object'])} → {html_escape(metrics['target_bin'])}</span>
        </div>
      </div>
      <div class="header-sub" style="margin-top:8px;">
        Command: <strong>{html_escape(metrics['command'])}</strong>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "🎥 Live Simulation",
        "📊 Analytics",
        "🧠 Diagnostics",
        "🗂️ Run History",
        "📤 Reports & Export",
        "🛡️ Safety & Help",
    ]
)

with tabs[0]:
    left, right = st.columns([2.35, 1])

    with left:
        st.subheader("Live rendered simulation")
        components.html(result["html"], height=HEIGHT + 215, scrolling=False)

        st.subheader("Final annotated frame")
        st.image(result["final_frame_rgb"], use_container_width=True)

        if result["video_bytes"]:
            if result["video_playable"]:
                st.subheader("Generated video file")
                st.video(result["video_bytes"], format=result["video_mime"])

            st.download_button(
                "Download generated video file",
                data=result["video_bytes"],
                file_name=f"industrial_demo_{result['run_id']}.{result['video_ext']}",
                mime=result["video_mime"] or "video/mp4",
                use_container_width=True,
            )

        st.subheader("Frame inspector")
        if result["frames_b64"]:
            max_idx = len(result["frames_b64"]) - 1
            selected = st.slider(
                "Select frame",
                min_value=0,
                max_value=max_idx,
                value=max_idx,
                key=f"frame_inspector_{result['run_id']}",
            )
            b64 = result["frames_b64"][selected]
            frame_bytes = base64.b64decode(b64)
            frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_img = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
            if frame_img is not None:
                meta = result["frame_meta"][selected]
                st.image(
                    cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {selected + 1}/{len(result['frames_b64'])} • {meta['phase']} • t={meta['time']:.2f}s",
                    use_container_width=True,
                )

    with right:
        st.subheader("Live telemetry")
        st.metric("Terminal state", metrics["terminal_state"])
        st.metric("Detection confidence", f"{metrics['detection_confidence']:.2f}")
        st.metric("3D localization error", f"{metrics['localization_error_cm']:.1f} cm")
        st.metric("Grasp score", f"{metrics['grasp_score']:.2f}")
        st.metric("Grasp success", "Yes" if metrics["grasp_success"] else "No")
        st.metric("Placement error", f"{metrics['placement_error_mm']:.0f} mm")
        st.metric("Task completion time", f"{metrics['task_time_s']:.1f} s")

        if metrics["injected_failure"]:
            recovery = "Yes" if metrics["recovery_success"] else "No"
        else:
            recovery = "N/A"
        st.metric("Recovery success", recovery)

        st.caption("Synthetic simulation telemetry. These are not physical-robot benchmark numbers.")

with tabs[1]:
    st.subheader("Process analytics")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Collision-free execution", "100%" if metrics["collision_free_execution_rate"] > 0.99 else "0%")
    with c2:
        st.metric("IK success", "Yes" if metrics["ik_success"] else "No")
    with c3:
        st.metric("Mean contact force", f"{metrics['mean_contact_force_n']:.1f} N")
    with c4:
        st.metric("Max localization error", f"{metrics['max_localization_error_cm']:.1f} cm")

    st.markdown(
        svg_metric_chart(
            result["timeseries"],
            "confidence",
            "Detection confidence over time",
            "#22c55e",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        svg_metric_chart(
            result["timeseries"],
            "loc_err_cm",
            "3D localization error over time",
            "#38bdf8",
            " cm",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        svg_metric_chart(
            result["timeseries"],
            "object_z_m",
            "Target object height over time",
            "#a78bfa",
            " m",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        svg_metric_chart(
            result["timeseries"],
            "contact_force_n",
            "Gripper contact force estimate",
            "#f59e0b",
            " N",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        svg_bar_chart(
            result["phase_durations"],
            "State-machine phase duration",
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Failure taxonomy for this run")
    if result["failure_breakdown"]:
        st.dataframe(result["failure_breakdown"], use_container_width=True, height=220)
    else:
        st.success("No warning/error events were recorded for this run.")

with tabs[2]:
    st.subheader("Subsystem health")
    st.markdown(health_html(result["subsystem_health"]), unsafe_allow_html=True)

    st.subheader("Alarms")
    alarms = [e for e in result["events"] if e["severity"] in ("WARN", "ERROR", "CRITICAL")]
    if alarms:
        for a in alarms:
            css_class = "alarm-card alarm-error" if a["severity"] in ("ERROR", "CRITICAL") else "alarm-card"
            st.markdown(
                f"""
                <div class="{css_class}">
                  <strong>{html_escape(a['severity'])}</strong> •
                  {html_escape(a['time_s'])}s •
                  {html_escape(a['module'])} •
                  {html_escape(a['code'])}:
                  {html_escape(a['message'])}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("No active alarms.")

    st.subheader("Structured event log")
    event_rows = [
        {
            "Time (s)": e["time_s"],
            "Frame": e["frame"],
            "Module": e["module"],
            "Severity": e["severity"],
            "Code": e["code"],
            "Message": e["message"],
        }
        for e in result["events"]
    ]
    st.dataframe(event_rows, use_container_width=True, height=330)

    st.subheader("State machine sequence")
    st.write(" → ".join(result["phases"]))

with tabs[3]:
    st.subheader("Run history")

    if st.session_state["run_history"]:
        st.dataframe(st.session_state["run_history"], use_container_width=True, height=390)

        if st.button("Clear run history"):
            st.session_state["run_history"] = []
            st.rerun()
    else:
        st.info("No runs recorded yet.")

with tabs[4]:
    st.subheader("Export artifacts")

    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.download_button(
            "Download metrics JSON",
            data=json.dumps(result["metrics"], indent=2),
            file_name=f"metrics_{result['run_id']}.json",
            mime="application/json",
            use_container_width=True,
        )

    with e2:
        st.download_button(
            "Download events CSV",
            data=dict_list_to_csv(result["events"]),
            file_name=f"events_{result['run_id']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with e3:
        st.download_button(
            "Download telemetry CSV",
            data=dict_list_to_csv(result["timeseries"]),
            file_name=f"telemetry_{result['run_id']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with e4:
        st.download_button(
            "Download HTML report",
            data=build_html_report(result),
            file_name=f"report_{result['run_id']}.html",
            mime="text/html",
            use_container_width=True,
        )

    st.subheader("Configuration snapshot")
    st.json(result["config_snapshot"])

with tabs[5]:
    st.subheader("Safety statement")
    st.markdown(
        """
        - This application is a **synthetic simulation demo only**.
        - It must not be used to command a real robot without a separate real-robot safety gate.
        - Natural-language commands are parsed and validated before any motion logic is used.
        - Unknown objects, malformed commands, and missing destinations are rejected.
        - In a real deployment, workspace limits, speed limits, force limits, emergency stop,
          supervised first runs, and hardware safety certification would be mandatory.
        """
    )

    st.subheader("Accepted command examples")
    st.code(
        """
put the cup in the bin
Put the cup in the bin.
place bottle in right bin
move cube to bin b
cup in left bin
pick the box
        """.strip(),
        language="text",
    )

    st.subheader("Real image asset setup")
    st.markdown(
        """
        For customer/site-specific imagery, create an `assets/` folder beside `app.py`:

        ```text
        project/
        ├── app.py
        ├── requirements.txt
        └── assets/
            ├── cube.png
            ├── cup.png
            ├── bottle.png
            └── box.png
        ```

        Recommended format:
        - PNG with transparent background
        - object centered
        - clean crop
        - no background clutter

        JPG is also accepted. The app will attempt to remove a white/light background.
        """
    )

    st.subheader("Troubleshooting")
    st.markdown(
        """
        - If the command shows an error, use one of the sidebar quick buttons.
        - If no video download appears, the live HTML player still works; video encoding backend may be unavailable.
        - If uploaded images do not appear, use PNG with transparency or place files in `assets/`.
        - If the app feels slow, reduce speed scale or disable auto-regeneration.
        """
    )