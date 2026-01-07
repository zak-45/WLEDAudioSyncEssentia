# src/genre_color_profile_loader.py

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
        )

    default = profiles.get("DEFAULT")
    if default is None:
        raise ValueError("DEFAULT genre profile is required")

    return profiles, default
