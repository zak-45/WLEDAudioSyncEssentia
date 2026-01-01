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

    def __init__(self, genre_json_path, mood_config_path, genre_energy_path="configs/genre_energy.json"):

        self.valence_weights = self._build_valence_weights(genre_json_path, mood_config_path)
        self.genre_energy = self._load_genre_energy(genre_energy_path)

    # ------------------------------------------------------------
    # Valence weights auto-derived from JSON
    # ------------------------------------------------------------
    @staticmethod
    def _build_valence_weights(model_json_path, mood_config_path):
        """
        Builds valence weights automatically from:
        - model JSON (labels)
        - mood config JSON (sentiment groups)
        """

        # Load model labels
        with open(model_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        labels = meta.get("classes") or meta.get("labels")

        # Load mood config
        with open(mood_config_path, "r", encoding="utf-8") as f:
            mood_cfg = json.load(f)

        weights = defaultdict(float)

        # Build lookup table: macro → weight
        macro_weight = {}

        for group_name, group in mood_cfg.items():
            w = float(group.get("weight", 0.0))
            for genre in group.get("genres", []):
                macro_weight[genre] = w

        # Assign weights per macro genre found in labels
        for label in labels:
            macro = label.split("---")[0]
            weights[macro] = macro_weight.get(macro, 0.0)

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
    def compute_energy(
            self,
            danceability,
            rms_rt,
            top_genre,
            confidence
    ):
        genre_energy = self.genre_energy.get(top_genre, 0.5)
        rms_energy = self.rms_to_energy(rms_rt)

        energy = (
                0.45 * danceability +
                0.35 * genre_energy +
                0.20 * rms_energy
        )

        # boost if stable & confident
        energy *= 0.7 + 0.3 * confidence

        return float(np.clip(energy, 0.0, 1.0))

    @staticmethod
    def rms_to_energy(rms, min_rms=0.002, max_rms=0.05):
        x = (rms - min_rms) / (max_rms - min_rms)
        return float(np.clip(x, 0.0, 1.0))



    def _load_genre_energy(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # ------------------------------------------------------------
    # Energy → [-1, 1]
    # ------------------------------------------------------------
    """
    @staticmethod
    def compute_energy(danceability):

        #danceability ∈ [0,1]
   
        return 2.0 * danceability - 1.0
    """

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

    # --------------------------------------------------
    # Hue fusion
    # --------------------------------------------------

    def fuse_hues(
        self,
        genre_hue,
        mood_hue,
        confidence
    ):
        """
        Circular blend between genre hue and mood hue.
        Returns final hue in [0, 360)
        """

        """
        # how much mood influences final hue
        mood_weight = 0.2 + 0.5 * confidence
        mood_weight = float(np.clip(mood_weight, 0.2, 0.7))
        """

        # mood dominates when genre confidence is LOW
        mood_weight = 0.2 + 0.5 * (1.0 - confidence)
        mood_weight = np.clip(mood_weight, 0.2, 0.7)

        return self._circular_lerp(
            genre_hue,
            mood_hue,
            mood_weight
        )

    @staticmethod
    def _circular_lerp(h1, h2, t):
        """
        Interpolates hues on a color wheel
        """
        delta = ((h2 - h1 + 180) % 360) - 180
        return (h1 + t * delta) % 360


    def final_color(
        self,
        genre_hue,
        mood_hue,
        confidence,
        energy
    ):
        # --- Hue fusion ---
        final_hue = self.fuse_hues(
            genre_hue,
            mood_hue,
            confidence
        )

        # --- Saturation ---
        # genre confidence drives saturation, but bounded
        saturation = 0.55 + 0.35 * confidence
        saturation = np.clip(saturation, 0.5, 0.9)

        # --- Value (brightness) ---
        # energy already includes danceability + genre bias
        value = 0.35 + 0.65 * energy
        value = np.clip(value, 0.35, 1.0)

        return self._hsv_to_rgb(
            final_hue,
            int(saturation * 255),
            int(value * 255)
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        s /= 255.0
        v /= 255.0

        c = v * s
        x = c * (1 - abs((h / 60.0) % 2 - 1))
        m = v - c

        if h < 60:
            rp, gp, bp = c, x, 0
        elif h < 120:
            rp, gp, bp = x, c, 0
        elif h < 180:
            rp, gp, bp = 0, c, x
        elif h < 240:
            rp, gp, bp = 0, x, c
        elif h < 300:
            rp, gp, bp = x, 0, c
        else:
            rp, gp, bp = c, 0, x

        r = int((rp + m) * 255)
        g = int((gp + m) * 255)
        b = int((bp + m) * 255)

        return r, g, b
