# src/genre_color_profile.py
"""Data structures for describing how each music genre should look as light.

This module defines a compact configuration object that encodes hue, minimum
brightness and saturation, and various gain and decay factors for genre-driven
lighting. It is used by the analysis pipeline to turn genre and energy values
into consistent, genre-specific color and accent behaviours on the LEDs.
"""

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
