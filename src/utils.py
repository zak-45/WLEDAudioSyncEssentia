import soxr
import numpy as np
import colorsys

from src.genre_hues import *

def resample(audio, sr_in, sr_out):
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)

    return soxr.resample(
        audio.astype(np.float32, copy=False),
        sr_in,
        sr_out,
        quality="HQ"
    )

def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def compute_color(top_genre, top_prob, energy):
    macro = top_genre.split("---")[0]
    hue = GENRE_HUES.get(macro, 0)  # fallback red

    saturation = min(1.0, top_prob * 1.5)
    brightness = max(0.1, min(1.0, energy))

    return hsv_to_rgb(hue, saturation, brightness)
