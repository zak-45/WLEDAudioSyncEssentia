import json
import numpy as np
from collections import defaultdict


class MoodColorMapper:
    """
    Maps music mood & energy to RGB colors.

    Definitions:
    - activity_energy  → physical motion (0..1)
    - emotional_energy   → emotional intensity (0..1)
    - valence          → sadness ↔ happiness (-1..1)
    """

    # ==================================================
    # INIT
    # ==================================================

    def __init__(self, genre_json_path, mood_config_path):
        self._valence = 0.0
        self.valence_weights = self._build_valence_weights(
            genre_json_path,
            mood_config_path
        )

        with open(mood_config_path, "r", encoding="utf-8") as f:
            mood_cfg = json.load(f)
        self.genre_valence = mood_cfg

        self._genre_to_valence = {}

        for mood, block in self.genre_valence.items():
            w = block["weight"]
            for g in block["genres"]:
                self._genre_to_valence[g] = 0.5 + w

    # ==================================================
    # VALENCE
    # ==================================================

    @staticmethod
    def _build_valence_weights(model_json_path, mood_config_path):
        with open(model_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        labels = meta.get("classes") or meta.get("labels")

        with open(mood_config_path, "r", encoding="utf-8") as f:
            mood_cfg = json.load(f)

        macro_weight = {}
        for group in mood_cfg.values():
            w = float(group.get("weight", 0.0))
            for genre in group.get("genres", []):
                macro_weight[genre] = w

        weights = defaultdict(float)
        for label in labels:
            macro = label.split("---")[0]
            weights[macro] = macro_weight.get(macro, 0.0)

        return dict(weights)

    # ==================================================
    # VALENCE ESTIMATION
    # ==================================================

    # --------------------------------------------------
    def compute_valence(
        self,
        top5,
        brightness: float,
        activity_energy: float,
        confidence: float
    ) -> float:
        """
        Robust valence estimation in [0,1]

        Inputs:
        - top5: genre probabilities
        - brightness: perceptual brightness [0,1]
        - activity_energy: physical energy [0,1]
        - confidence: genre certainty [0,1]
        """

        # --------------------------------------------------
        # 1. Genre prior (existing logic)
        # --------------------------------------------------
        genre_valence = 0.0
        weight_sum = 0.0

        for label, prob in top5:
            macro = label.split("---")[0]
            gv = self._genre_to_valence.get(macro, 0.5)
            genre_valence += gv * prob
            weight_sum += prob
            # print("[GENRE LOOKUP]", macro, "→", gv)

        if weight_sum > 0:
            genre_valence /= weight_sum
        else:
            genre_valence = 0.5

        genre_valence = float(np.clip(genre_valence, 0.0, 1.0))

        # --------------------------------------------------
        # 2. Mode proxy (major / minor feeling)
        # --------------------------------------------------
        # Bright + stable = major-ish

        mode_proxy = brightness * (0.6 + 0.4 * activity_energy)
        mode_proxy = float(np.clip(mode_proxy, 0.0, 1.0))

        # --------------------------------------------------
        # 3. Activity bias (directional, not additive)
        # --------------------------------------------------
        if brightness > 0.5:
            activity_bias = +0.1 * activity_energy
        else:
            activity_bias = -0.1 * activity_energy

        # --------------------------------------------------
        # 4. Confidence stabilizer
        # --------------------------------------------------
        confidence_bias = 0.1 * (confidence - 0.5)

        # --------------------------------------------------
        # 5. Weighted fusion
        # --------------------------------------------------
        raw_valence = (
            0.35 * genre_valence +
            0.25 * brightness +
            0.20 * mode_proxy +
            0.10 * activity_bias +
            0.10 * confidence_bias
        )

        raw_valence = float(np.clip(raw_valence, 0.0, 1.0))

        # --------------------------------------------------
        # DEBUG
        # --------------------------------------------------
        """
        print(
            f"[VALENCE DEBUG] "
            f"raw={raw_valence:.3f} "
            f"prev={self._valence:.3f} "
            f"bright={brightness:.3f} "
            f"energy={activity_energy:.3f} "
            f"conf={confidence:.3f}"
        )

        print(
            "[GENRE VALENCE DEBUG]",
            f"genre_valence={genre_valence:.3f}",
            f"top5={[(l.split('---')[0], round(p, 3)) for l, p in top5]}"
        )

        """
        # --------------------------------------------------
        # 6. Temporal smoothing (important!)
        # --------------------------------------------------
        alpha = 0.2
        self._valence = (
            alpha * raw_valence +
            (1.0 - alpha) * self._valence
        )

        return float(self._valence)

    # ==================================================
    # HUE
    # ==================================================

    @staticmethod
    def mood_to_hue(valence, emotional_energy):
        """
        Valence controls warm ↔ cool
        Emotion slightly bends hue
        """
        base_hue = np.interp(valence, [-1, 1], [220, 20])
        energy_shift = np.interp(emotional_energy, [0, 1], [-20, 20])
        return (base_hue + energy_shift) % 360

    # ==================================================
    # HUE FUSION
    # ==================================================

    def fuse_hues(self, genre_hue, mood_hue, confidence):
        """
        Mood dominates when genre confidence is low
        """
        mood_weight = 0.2 + 0.5 * (1.0 - confidence)
        mood_weight = float(np.clip(mood_weight, 0.2, 0.7))
        return self._circular_lerp(genre_hue, mood_hue, mood_weight)

    @staticmethod
    def _circular_lerp(h1, h2, t):
        delta = ((h2 - h1 + 180) % 360) - 180
        return (h1 + t * delta) % 360

    # ==================================================
    # FINAL COLOR (FIXED + ALIGNED)
    # ==================================================

    def final_color(
        self,
        genre_hue,
        mood_hue,
        confidence,
        activity_energy,
        emotional_energy
    ):
        """
        Correct mapping:
        - activity_energy → saturation
        - emotional_energy  → brightness
        """

        # --- Hue ---
        final_hue = self.fuse_hues(
            genre_hue,
            mood_hue,
            confidence
        )

        # --- Saturation (physical motion) ---
        sat_floor = 0.25
        saturation = sat_floor + activity_energy * (1.0 - sat_floor)
        saturation = np.clip(saturation, 0.25, 0.95)

        # --- Brightness (emotional intensity) ---
        bright_floor = 0.20
        value = bright_floor + emotional_energy * (1.0 - bright_floor)
        value = np.clip(value, 0.20, 1.0)

        return self._hsv_to_rgb(final_hue, saturation, value)

    # ==================================================
    # ACCENT COLOR
    # ==================================================

    def accent_color(self, final_hue, activity_energy, confidence):
        if activity_energy < 0.35:
            accent_hue = (final_hue + 30) % 360
        elif activity_energy < 0.65:
            accent_hue = (final_hue + 150) % 360
        else:
            accent_hue = (final_hue + 180) % 360

        saturation = 0.45 + 0.35 * confidence
        saturation = np.clip(saturation, 0.45, 0.85)

        value = 0.25 + 0.55 * activity_energy
        value = np.clip(value, 0.25, 0.8)

        return self._hsv_to_rgb(accent_hue, saturation, value)

    # ==================================================
    # FLASH
    # ==================================================

    @staticmethod
    def apply_flash(rgb, flash_strength):
        if flash_strength <= 0.0:
            return rgb

        r, g, b = rgb
        boost = 1.0 + 0.8 * flash_strength

        return (
            min(255, int(r * boost)),
            min(255, int(g * boost)),
            min(255, int(b * boost)),
        )

    # ==================================================
    # HSV → RGB
    # ==================================================

    @staticmethod
    def _hsv_to_rgb(h, s, v):
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

        return (
            int((rp + m) * 255),
            int((gp + m) * 255),
            int((bp + m) * 255),
        )
