# emotion_color.py
# Emotion → RGB mapping with JSON-configurable colors & white center

import json
import os

# --------------------------------------------------
# Load configuration
# --------------------------------------------------

DEFAULT_CONFIG = {
    "white_center": {
        "color": [255, 255, 255],
        "radius": 0.25
    },
    "emotions": {
        "anger":   [255,   0,   0],
        "joy":     [255, 255,   0],
        "calm":    [0,   255,   0],
        "sadness": [0,     0, 255]
    }
}


def load_emotion_color_config(path="config/emotion_colors.json"):
    if not os.path.exists(path):
        return DEFAULT_CONFIG

    with open(path, "r") as f:
        cfg = json.load(f)

    # Merge with defaults (safety)
    out = DEFAULT_CONFIG.copy()
    out["white_center"].update(cfg.get("white_center", {}))
    out["emotions"].update(cfg.get("emotions", {}))

    return out


_CONFIG = load_emotion_color_config()

WHITE_RGB   = tuple(_CONFIG["white_center"]["color"])
WHITE_RADIUS = float(_CONFIG["white_center"]["radius"])

ANGER_RGB   = tuple(_CONFIG["emotions"]["anger"])
JOY_RGB     = tuple(_CONFIG["emotions"]["joy"])
CALM_RGB    = tuple(_CONFIG["emotions"]["calm"])
SADNESS_RGB = tuple(_CONFIG["emotions"]["sadness"])


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _lerp(c1, c2, t):
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


# --------------------------------------------------
# Public API
# --------------------------------------------------

def emotion_to_rgb(valence, arousal, intensity=1.0):
    """
    Emotion → RGB using JSON-configured colors.

    valence   ∈ [-1, 1]
    arousal   ∈ [-1, 1]
    intensity ∈ [0, 1]
    """

    # Clamp inputs
    valence   = _clamp(valence, -1.0, 1.0)
    arousal   = _clamp(arousal, -1.0, 1.0)
    intensity = _clamp(intensity, 0.0, 1.0)

    # --------------------------------------------------
    # 1. Emotion corner interpolation
    x = (valence + 1.0) * 0.5
    y = (arousal + 1.0) * 0.5

    top    = _lerp(ANGER_RGB, JOY_RGB, x)
    bottom = _lerp(SADNESS_RGB, CALM_RGB, x)
    emotion_rgb = _lerp(bottom, top, y)

    # --------------------------------------------------
    # 2. White center blending
    radius = min(1.0, (valence * valence + arousal * arousal) ** 0.10)

    if radius <= WHITE_RADIUS:
        mix = 0.0
    else:
        mix = (radius - WHITE_RADIUS) / (1.0 - WHITE_RADIUS)

    r, g, b = _lerp(WHITE_RGB, emotion_rgb, mix)

    # --------------------------------------------------
    # 3. Apply intensity

    intensity = remap_intensity(intensity, in_max=1.0)

    return (
        int(r * intensity),
        int(g * intensity),
        int(b * intensity),
    )


def remap_intensity(value, in_max=100.0):
    """
    Remap intensity so that:
      - [0 .. 70%]   -> 0.7
      - [70 .. 100%] -> linear 0.7 .. 1.0

    value  : raw intensity (0..in_max)
    returns: intensity (0..1)
    """

    if in_max <= 0:
        return 0.7

    v = _clamp(value / in_max, 0.0, 1.0)

    if v <= 0.7:
        return 0.7

    # linear ramp from 0.7 → 1.0
    return 0.7 + (v - 0.7) * 1.0
