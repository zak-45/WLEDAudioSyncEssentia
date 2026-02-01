# src/genre_color_profile_loader.py
"""Loader for genre-specific color profile settings.

This module reads JSON configuration that describes how each music genre should
map to hue, brightness, saturation, and accent behaviour for the lighting
system. It returns a lookup of named profiles plus a required DEFAULT profile
so the analysis core can always fall back to sensible visual parameters.
"""

import json
from src.genre_color_profile import GenreColorProfile

def load_genre_color_profiles(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    profiles = {}

    for name, data in raw.items():
        if name.startswith("_"):
            continue

        profiles[name] = GenreColorProfile(
            hue=float(data["hue"]),
            sat_floor=float(data["sat_floor"]),
            bright_floor=float(data["bright_floor"]),
            mood_hue_weight=float(data["mood_hue_weight"]),
            energy_boost=float(data["energy_boost"]),
            accent_gain=float(data["accent_gain"]),
            flash_decay=float(data["flash_decay"]),
            emotion_sat_gain=float(data.get("emotion_sat_gain", 1.0)),
            emotion_bright_gain=float(data.get("emotion_bright_gain", 1.0)),
            emotion_white_gain=float(data.get("emotion_white_gain", 1.0)),
            emotion_intensity_curve=data.get("emotion_intensity_curve", "linear")
        )

    default = profiles.get("DEFAULT")
    if default is None:
        raise ValueError("DEFAULT genre profile is required")

    return profiles, default
