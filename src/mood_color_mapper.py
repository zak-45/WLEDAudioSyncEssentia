import json
import colorsys
from collections import defaultdict
import numpy as np

class MoodColorMapper:
    """
    Maps music mood to RGB color using:
    - Valence (derived from genre labels + JSON metadata)
    - Energy (danceability / RMS / confidence)
    """

    def __init__(self, genre_json_path):
        self.valence_weights = self._build_valence_weights(genre_json_path)

    # ------------------------------------------------------------
    # Valence weights auto-derived from JSON
    # ------------------------------------------------------------
    @staticmethod
    def _build_valence_weights(json_path):
        """
        Builds valence weights automatically from model JSON.
        Strategy:
        - Uses label names (macro genres)
        - Applies heuristic sentiment mapping
        """

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        labels = meta.get("classes") or meta.get("labels")
        weights = defaultdict(float)

        POSITIVE = {
            "Pop", "Dance", "Disco", "Latin", "Reggae",
            "Funk", "Soul", "House", "Electro"
        }

        NEGATIVE = {
            "Metal", "Noise", "Industrial", "Dark",
            "Experimental", "Drone"
        }

        CALM = {
            "Ambient", "Classical", "Jazz", "Soundtrack"
        }

        for label in labels:
            macro = label.split("---")[0]

            if macro in POSITIVE:
                weights[macro] = 0.6
            elif macro in NEGATIVE:
                weights[macro] = -0.6
            elif macro in CALM:
                weights[macro] = -0.2
            else:
                weights[macro] = 0.0

        return dict(weights)

    # ------------------------------------------------------------
    # Compute valence from genre predictions
    # ------------------------------------------------------------
    def compute_valence(self, genre_probs):
        """
        genre_probs: list of (label, probability)
        returns: float in [-1, 1]
        """
        valence = 0.0
        total = 0.0

        for label, prob in genre_probs:
            macro = label.split("---")[0]
            weight = self.valence_weights.get(macro, 0.0)
            valence += weight * prob
            total += prob

        if total > 0:
            valence /= total

        return max(-1.0, min(1.0, valence))

    # ------------------------------------------------------------
    # Energy → [-1, 1]
    # ------------------------------------------------------------
    @staticmethod
    def compute_energy(danceability):
        """
        danceability ∈ [0,1]
        """
        return 2.0 * danceability - 1.0

    # ------------------------------------------------------------
    # Valence + Energy → RGB
    # ------------------------------------------------------------
    @staticmethod
    def mood_to_rgb(valence, energy, confidence):
        """
        confidence: top-1 genre probability
        """

        # Hue: sad(240=blue) → happy(0=red/yellow)
        hue = 240.0 - (valence + 1.0) * 120.0

        # Brightness: calm → energetic
        value = 0.3 + 0.7 * ((energy + 1.0) / 2.0)

        # Saturation: confidence-driven
        saturation = min(1.0, confidence * 1.5)

        r, g, b = colorsys.hsv_to_rgb(
            hue / 360.0,
            saturation,
            value
        )

        return int(r * 255), int(g * 255), int(b * 255)

    @staticmethod
    def mood_to_hue(valence, energy):
        """
        Map valence (-1..1) and energy (0..1) to a hue angle.
        """
        # Valence controls warm ↔ cool
        # Energy controls brightness but slightly shifts hue
        base_hue = np.interp(valence, [-1, 1], [220, 20])  # sad→happy

        energy_shift = np.interp(energy, [0, 1], [-20, 20])

        return (base_hue + energy_shift) % 360