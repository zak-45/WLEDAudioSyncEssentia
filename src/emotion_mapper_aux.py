import json
from src.emotion_color import emotion_to_rgb   # authoritative color logic


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class EmotionMapperAUX:
    """
    Maps Essentia AUX classifiers (0..1)
    into valence, arousal, intensity
    and then into WLED parameters.
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.effects = config_manager.load("default")
        self.debug = False

    # --------------------------------------------------

    def compute_emotion(self, aux: dict):
        """
        aux example:
        {
          "danceable": 0.1377,
          "aggressive": 0.06529,
          "happy": 0.02156,
          "relaxed": 0.94882,
          "sad": 0.67817
        }
        """

        happy = aux.get("happy", 0.0)
        relaxed = aux.get("relaxed", 0.0)
        sad = aux.get("sad", 0.0)
        aggressive = aux.get("aggressive", 0.0)
        danceable = aux.get("danceable", 0.0)

        # --------------------------------------------------
        # VALENCE (pleasant ↔ unpleasant)
        valence = (
            0.6 * happy +
            0.4 * relaxed -
            0.7 * sad -
            0.5 * aggressive
        )
        valence = clamp(valence, -1.0, 1.0)

        # --------------------------------------------------
        # AROUSAL (low ↔ high energy)
        arousal = (
            0.6 * aggressive +
            0.4 * happy -
            0.6 * relaxed -
            0.5 * sad
        )
        arousal = clamp(arousal, -1.0, 1.0)

        # --------------------------------------------------
        # INTENSITY (energy / drive)
        intensity = clamp(danceable, 0.0, 1.0)

        label, quadrant = self.emotion_label(valence, arousal)

        if self.debug:
            print(
                f"[EMOTION] {label:18s} | "
                f"V={valence:+.2f} "
                f"A={arousal:+.2f} | "
                f"{quadrant}"
            )

        return valence, arousal, intensity

    # --------------------------------------------------

    def emotion_to_wled(self, valence, arousal, intensity):
        """
        Converts emotion space → WLED parameters
        using the NEW correct emotion→RGB logic.
        """

        # --------------------------------------------------
        # COLOR (authoritative)
        r, g, b = emotion_to_rgb(valence, arousal, intensity)

        # --------------------------------------------------
        # EFFECT selection (unchanged)
        effect, effect_label, effect_index, _ = self.effects.select_effect(valence, arousal)

        # --------------------------------------------------
        # MOTION parameters
        speed = int(40 + 200 * abs(arousal))
        intensity_param = int(50 + 200 * intensity)

        if self.debug:
            print(
                f"[WLED] {effect_label:8s} | "
                f"V={valence:+.2f} "
                f"A={arousal:+.2f} "
                f"RGB=({r:3d},{g:3d},{b:3d}) "
                f"FX={effect}"
            )

        return {
            "rgb": (r, g, b),
            "effect": effect,
            "index": effect_index,
            "speed": speed,
            "intensity": intensity_param,
        }

    # --------------------------------------------------

    def update_context(self, genre=None, bpm=None, hour=None):
        self.effects = self.config_manager.select(
            genre=genre,
            bpm=bpm,
            hour=hour
        )

    # --------------------------------------------------

    @staticmethod
    def emotion_label(valence, arousal):
        """
        Returns (label, quadrant_name)
        """

        if abs(valence) < 0.15 and abs(arousal) < 0.15:
            return "Neutral", "Center"

        if arousal >= 0:
            if valence >= 0:
                return "Joy / Elation", "Pleasant + High Energy"
            else:
                return "Anger / Tension", "Unpleasant + High Energy"
        else:
            if valence >= 0:
                return "Calm / Relief", "Pleasant + Low Energy"
            else:
                return "Sadness / Withdrawal", "Unpleasant + Low Energy"


# ------------------------------------------------------
# EFFECT CONFIG (UNCHANGED)
# ------------------------------------------------------

class EffectConfig:
    def __init__(self, path):
        with open(path, "r") as f:
            self.config = json.load(f)

        self.thresholds = self.config["thresholds"]
        self.effects = self.config["effects"]

    def select_effect(self, valence, arousal):
        ctx = {
            "valence": valence,
            "arousal": arousal,
            **self.thresholds
        }

        for name, entry in self.effects.items():
            if eval(entry["condition"], {}, ctx):
                return entry["effect"], name, entry["index"], "index"

        return "Solid", "fallback", 0, "index"


# -------------------------------------------------
# STROBE
#if strobe_ctrl.update(valence, arousal, intensity, now_ms):
#    effect = "Strobe"
#else:
#    effect = normal_effect
#
# -------------------------------------------------

class StrobeController:
    def __init__(self):
        self.last_arousal = 0.0
        self.last_time = 0
        self.active_until = 0

    def update(self, valence, arousal, intensity, now_ms):
        # Stop strobe if active and expired
        if now_ms > self.active_until:
            self.active_until = 0

        # Compute arousal delta
        delta = arousal - self.last_arousal
        self.last_arousal = arousal

        # Conditions
        allow = (
            arousal > 0.6 and
            delta > 0.25 and
            not (valence > 0 and arousal < 0)
        )

        if allow and self.active_until == 0:
            self.active_until = now_ms + 180  # ms
            return True  # START STROBE

        return self.active_until > now_ms  # CONTINUE or OFF
