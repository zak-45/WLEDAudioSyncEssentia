# src/genre_color_profile.py

from dataclasses import dataclass

@dataclass(frozen=True)
class GenreColorProfile:
    hue: float
    sat_floor: float
    bright_floor: float
    mood_hue_weight: float
    energy_boost: float
    accent_gain: float
    flash_decay: float
