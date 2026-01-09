# src/analysis_process_core.py
import json
import queue
import time
from collections import deque

import numpy as np

from src.effnet_classifier import EffnetClassifier, AuxClassifier
from config.genre_flash_shape import GENRE_FLASH_SHAPES
from src.macro_genres import collapse_to_macro
from src.model_loader import discover_models
from src.smoothing import GenreSmoother
from src.mood_color_mapper import MoodColorMapper
from src.adaptive_buffer import AdaptiveBuffer
from src.utils import compute_color
from src.genre_color_profile_loader import load_genre_color_profiles

# fetch all models from folder
models = discover_models("models")
#
DEFAULT_FLASH_SHAPE = "pulse"
#

class AnalysisCore:
    def __init__(
        self,
        audio_queue,
        cfg,
        osc,
        visual,
        use_macro,
        macro_agg,
        color1,
        debug,
        aux,
        last_beat_time,
        activate_buffer,
    ):
        self.motion_history = deque(maxlen=120)
        self._activity_energy = 0.0
        self._motion_hist = deque(maxlen=60)
        self.accent_strength = 0.0
        self.aux = aux
        self.danceability = 0.5
        self.aux_classifiers = []
        self.last_non_silent_time = time.time()
        self.audio_queue = audio_queue
        self.cfg = cfg
        self.osc = osc
        self.visual = visual
        self.debug = debug
        self.color1 = color1
        self.use_macro = use_macro
        self.macro_agg = macro_agg
        self.is_silent = False
        self.last_beat_time = last_beat_time
        self.activate_buffer = activate_buffer

        # -------- MODELS (SAFE HERE) --------
        self.clf = EffnetClassifier()
        self.smooth = GenreSmoother(
            self.clf.labels,
            cfg.SMOOTHING_ALPHA
        )

        if self.aux:
            self.load_aux()

        self.mood_mapper = MoodColorMapper(
            "models/genre_discogs400-discogs-effnet-1.json",
            "config/mood_valence.json"
        )

        self.adaptive = AdaptiveBuffer(
            cfg.MIN_BUFFER_SECONDS,
            cfg.MAX_BUFFER_SECONDS,
            cfg.BUFFER_SECONDS,
            cfg.CONFIDENCE_THRESHOLD,
            cfg.STABILITY_FRAMES,
            cfg.BUFFER_GROWTH_RATE,
            cfg.BUFFER_SHRINK_RATE,
        )

        self.buffer = np.zeros(0, dtype=np.float32)
        self.last_analysis_time = 0.0
        self.fast_buffer = np.zeros(0, dtype=np.float32)

        self.genre_profiles, self.default_profile = load_genre_color_profiles(
            "config/genre_color_profiles.json"
        )

        self.prev_segment = None
        self.activity_smooth = 0.0

    # -----------------------------------------------------

    def run(self):
        hop = int(self.cfg.MODEL_SAMPLE_RATE * self.cfg.HOP_SECONDS)

        while True:
            try:
                audio, rms_rt, ts = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # --------------------------------------------------
            # Silence detection
            # --------------------------------------------------
            now = time.time()

            if rms_rt < self.cfg.MIN_RMS:
                if not self.is_silent and (now - self.last_non_silent_time) >= self.cfg.SILENCE_TIMEOUT:
                    self._enter_silence()
                continue
            else:
                self.last_non_silent_time = now
                if self.is_silent:
                    self._exit_silence()

            # --------------------------------------------------
            # FAST BUFFER (activity_energy)
            # --------------------------------------------------

            self.fast_buffer = np.concatenate([self.fast_buffer, audio])

            fast_needed = int(self.cfg.MODEL_SAMPLE_RATE * self.cfg.MIN_BUFFER_SECONDS)

            if len(self.fast_buffer) < fast_needed:
                continue

            fast_segment = self.fast_buffer[-fast_needed:]
            activity_energy = float(self.compute_activity_energy(fast_segment))

            # keep fast buffer tight
            self.fast_buffer = self.fast_buffer[-fast_needed:]

            # --------------------------------------------------
            # ADAPTIVE GENRE BUFFER
            # --------------------------------------------------

            self.buffer = np.concatenate([self.buffer, audio])

            if self.activate_buffer:
                needed = int(self.cfg.MODEL_SAMPLE_RATE * self.adaptive.current)
            else:
                needed = int(self.cfg.MODEL_SAMPLE_RATE * self.cfg.MIN_BUFFER_SECONDS)

            max_samples = int(self.cfg.MAX_BUFFER_SECONDS * self.cfg.MODEL_SAMPLE_RATE)
            self.buffer = self.buffer[-max_samples:]

            if len(self.buffer) < needed:
                continue

            segment = self.buffer[-needed:]

            # --------------------------------------------------
            # Genre classification
            # --------------------------------------------------
            probs = self.clf.classify(segment)
            if probs is None:
                continue

            self.smooth.update(probs)
            top5 = self.smooth.top_n(5)

            top_label, top_conf = top5[0]
            macro_label = top_label.split("---")[0]

            # adapt buffer size if not use macro genre
            if not self.use_macro and self.activate_buffer:
                self.adaptive.update(
                    top_label=top_label,
                    confidence=top_conf,
                    silent=False
                )

            # load genre profile params from JSON
            profile = self.genre_profiles.get(macro_label, self.default_profile)

            #color
            genre_hue = profile.hue

            #energy boost
            activity_energy = np.clip(
                activity_energy * profile.energy_boost,
                0.0, 1.0
            )

            if self.debug:
                print("GENRE TOP5: ",
                      " | ".join(f"{g}:{v:.5f}" for g, v in top5))
                print("GENRE CONF: ", top_conf)

            # --------------------------------------------------
            # OSC genre labels
            # --------------------------------------------------
            for i, (label, _) in enumerate(top5):
                self.osc.send(f"/WASEssentia/genre/top{i}", label)

            # --------------------------------------------------
            # Macro genres
            # --------------------------------------------------
            if self.use_macro:
                macro_probs = collapse_to_macro(probs, self.clf.labels, agg=self.macro_agg)

                total = sum(macro_probs.values())
                if total > 0:
                    for k in macro_probs:
                        macro_probs[k] /= total

                # top-5 macro genres
                top5_macro = sorted(
                    macro_probs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                top_label_macro, top_conf_macro = top5_macro[0]

                # update buffer size
                if self.activate_buffer:
                    self.adaptive.update(
                        top_label=top_label_macro,
                        confidence=top_conf_macro,
                        silent=False
                    )

                if self.debug:
                    print("MACRO TOP5: ",
                          " | ".join(f"{g}:{v:.5f}" for g, v in top5_macro))

                    print("MACRO CONF: ", top_conf_macro)

                # --------------------------------------------------
                # OSC macro genre labels
                # --------------------------------------------------
                for i, (label, _) in enumerate(top5_macro):
                    self.osc.send(f"/WASEssentia/genre/macro_top{i}", label)


            # --------------------------------------------------
            # AUX classifiers
            # --------------------------------------------------
            if self.aux_classifiers:
                embeddings = self.clf.compute_embeddings(segment)
                for aux in self.aux_classifiers:
                    results = aux.classify(embeddings)
                    if results is None:
                        continue

                    if aux.name == "danceability classifier":
                        self.danceability = float(results.get("danceable", 0.5))

                    if self.debug:
                        print('_|_')
                        print(f"AUX {aux.name} =", " | ".join(f"{k}:{v:.5f}" for k, v in results.items()))

                    # --------------------------------------------------
                    # OSC aux labels
                    # --------------------------------------------------
                    for label, value in results.items():
                        path = f"/WASEssentia/aux/{aux.name.replace(' ', '_')}/{label.replace(' ', '_')}"
                        self.osc.send(path, float(value))

            # --------------------------------------------------
            # Mood computation
            # --------------------------------------------------
            valence = self.mood_mapper.compute_valence(top5)

            emotional_weight = abs(valence - 0.5) * 2.0

            emotional_energy = (
                    0.5 * emotional_weight +  # emotional intensity
                    0.3 * activity_energy +  # physical support
                    0.2 * top_conf  # certainty
            )

            emotional_energy = float(np.clip(emotional_energy, 0.0, 1.0))

            # brightness & saturation

            brightness = (
                    profile.bright_floor +
                    emotional_energy * (1.0 - profile.bright_floor)
            )

            saturation = (
                    profile.sat_floor +
                    activity_energy * (1.0 - profile.sat_floor)
            )

            brightness = float(np.clip(brightness, 0.0, 1.0))
            saturation = float(np.clip(saturation, 0.0, 1.0))


            # --------------------------------------------------
            # Mood color
            # --------------------------------------------------

            mood_hue = self.mood_mapper.mood_to_hue(valence, emotional_energy)

            # --------------------------------------------------
            # Genre color
            # --------------------------------------------------

            r, g, b = compute_color(genre_hue, saturation, brightness)

            if self.debug:
                print("Genre color:", r, g, b)

            self.osc.send("/WASEssentia/genre/color/r", r / 255.0)
            self.osc.send("/WASEssentia/genre/color/g", g / 255.0)
            self.osc.send("/WASEssentia/genre/color/b", b / 255.0)

            # --------------------------------------------------
            # Mood color genre-centric override
            # --------------------------------------------------
            #
            r, g, b = compute_color(mood_hue, saturation, brightness)

            if self.debug:
                print("Mood color:", r, g, b)

            self.osc.send("/WASEssentia/mood/color/r", r / 255.0)
            self.osc.send("/WASEssentia/mood/color/g", g / 255.0)
            self.osc.send("/WASEssentia/mood/color/b", b / 255.0)

            # --------------------------------------------------
            # Final hue + colors
            # --------------------------------------------------

            final_hue = self.mood_mapper.fuse_hues(
                genre_hue=profile.hue,
                mood_hue=mood_hue,
                confidence=top_conf * profile.mood_hue_weight
            )

            # Authoritative production color
            r, g, b = self.mood_mapper.final_color(
                genre_hue=genre_hue,
                mood_hue=mood_hue,
                confidence=top_conf,
                activity_energy=activity_energy,
                emotional_energy=emotional_energy
            )

            if self.debug:
                print("Final color:", r, g, b)

            # Debug / genre-centric override
            if self.color1:
                brightness = max(0.1, min(1.0, activity_energy))
                saturation = min(1.0, top_conf * 1.5)
                r, g, b = compute_color(final_hue, saturation, brightness)
                print('Debug final :', r, g, b)

            # Accent color
            accent_r, accent_g, accent_b = self.mood_mapper.accent_color(
                final_hue=final_hue,
                activity_energy=activity_energy,
                confidence=top_conf
            )

            # Sync beats with analysis window
            segment_start = ts - (len(segment) / self.cfg.MODEL_SAMPLE_RATE)
            segment_end = ts

            beat_in_window = (
                    self.last_beat_time >= segment_start
                    and self.last_beat_time <= segment_end
            )

            # Update accent strength once
            self.update_accent_strength(
                beat=beat_in_window,
                energy=activity_energy,
                genre=macro_label
            )

            # Apply artistic gain
            self.accent_strength *= profile.accent_gain
            self.accent_strength = min(1.0, self.accent_strength)

            # always apply flash if there's any remaining strength
            if self.accent_strength > 0.01:
                accent_r, accent_g, accent_b = self.mood_mapper.apply_flash(
                    (accent_r, accent_g, accent_b),
                    flash_strength=self.accent_strength
                )

            if self.debug:
                print("Accent color:", accent_r, accent_g, accent_b)

            # --------------------------------------------------
            # OSC output
            # --------------------------------------------------
            self.osc.send("/WASEssentia/final/color/r", r / 255.0)
            self.osc.send("/WASEssentia/final/color/g", g / 255.0)
            self.osc.send("/WASEssentia/final/color/b", b / 255.0)

            self.osc.send("/WASEssentia/accent/color/r", accent_r / 255.0)
            self.osc.send("/WASEssentia/accent/color/g", accent_g / 255.0)
            self.osc.send("/WASEssentia/accent/color/b", accent_b / 255.0)

            self.osc.send(
                "/WASEssentia/mood/data",
                json.dumps({
                    "valence": round(valence, 3),
                    "activity_energy": round(activity_energy, 3),
                    "emotional_energy": round(emotional_energy, 3),
                    "R": r,
                    "G": g,
                    "B": b
                })
            )

            mood_data = json.dumps({
                    "valence": round(valence, 3),
                    "activity_energy": round(activity_energy, 3),
                    "emotional_energy": round(emotional_energy, 3),
                    "R": r,
                    "G": g,
                    "B": b
                })

            if self.debug:
                print(mood_data)

            # --------------------------------------------------
            # Visual debug
            # --------------------------------------------------
            if self.visual:
                self.visual.update(
                    genre=macro_label,
                    genre_hue=genre_hue,
                    mood_hue=mood_hue,
                    final_hue=final_hue,
                    valence=valence,
                    energy=activity_energy,
                    rgb=(r, g, b),
                    rgb_accent=(accent_r, accent_g, accent_b)
                )

                self.visual.render()

            # --------------------------------------------------
            # Advance hop
            # --------------------------------------------------
            self.buffer = self.buffer[-hop:]

    # ==================================================
    def _enter_silence(self):
        self.is_silent = True
        self.buffer = np.zeros(0, dtype=np.float32)
        self.fast_buffer = np.zeros(0, dtype=np.float32)
        self.smooth.reset()
        self.adaptive.reset()
        self.adaptive.current = self.cfg.MIN_BUFFER_SECONDS
        self.osc.send_silence(0)

        if self.debug:
            print("🔇 SILENCE → analysis reset")

    def _exit_silence(self):
        self.is_silent = False
        self.osc.send_silence(1)

        if self.debug:
            print("🎵 AUDIO RESUMED")

    # ==================================================
    def load_aux(self):
        self.aux_classifiers = []
        # load models and set them to list for type AUX
        for mod in models:
            if mod["type"] == "genre":

                if self.debug:
                    print(f"🎵 Genre model loaded: {mod['name']}")

            else:

                if self.aux:
                    self.aux_classifiers.append(
                        AuxClassifier(mod["name"], mod["pb"], mod["json"], mod["output_name"], agg=self.macro_agg)
                    )

                    if self.debug:
                        print(f"🎛 Aux model loaded: {mod['name']}")

    # ==================================================
    def update_accent_strength(self, beat, energy, genre):
        shape = GENRE_FLASH_SHAPES.get(genre, DEFAULT_FLASH_SHAPE)
        decay = 0.0

        if beat:
            if shape == "punch":
                self.accent_strength = 1.0

            elif shape == "hold":
                self.accent_strength = max(self.accent_strength, 1.0)

            elif shape == "glow":
                self.accent_strength += 0.6
                self.accent_strength = min(self.accent_strength, 1.0)

            elif shape == "bounce":
                self.accent_strength = 1.0

            elif shape == "pulse":
                self.accent_strength = 0.8

        # ---- decay behavior ----
        if shape == "punch":
            decay = 0.75 - 0.25 * energy

        elif shape == "hold":
            decay = 0.92 - 0.10 * energy

        elif shape == "glow":
            decay = 0.97 - 0.05 * energy

        elif shape == "bounce":
            decay = 0.85

        elif shape == "pulse":
            decay = 0.88 - 0.12 * energy

        if shape == "none":
            self.accent_strength = 0.0
            return

        self.accent_strength *= decay
        self.accent_strength = max(0.0, min(1.0, self.accent_strength))



    def compute_activity_energy(self, segment: np.ndarray) -> float:
        """
        Activity energy from FAST buffer.
        Depends on BOTH loudness and envelope motion.
        Returns [0.0, 1.0]
        """

        sr = self.cfg.MODEL_SAMPLE_RATE

        # --------------------------------------------------
        # 1. Short-term RMS envelope
        # --------------------------------------------------
        frame = int(0.02 * sr)  # 20 ms
        hop = int(0.01 * sr)  # 10 ms

        if len(segment) < frame * 3:
            return 0.0

        frames = np.lib.stride_tricks.sliding_window_view(
            segment, frame
        )[::hop]

        env = np.sqrt(np.mean(frames ** 2, axis=1))

        rms = float(np.mean(env))
        if rms < self.cfg.MIN_RMS:
            self._activity_energy = 0.0
            return 0.0

        # --------------------------------------------------
        # 2. Envelope motion (FAST)
        # --------------------------------------------------
        motion = np.abs(np.diff(env))
        env_motion = float(np.mean(motion))

        # --------------------------------------------------
        # 3. Normalize loudness (FIXED SCALE)
        # --------------------------------------------------
        RMS_FLOOR = self.cfg.MIN_RMS  # e.g. 0.002
        RMS_REF = self.cfg.REF_RMS  # e.g. 0.05

        rms_norm = (rms - RMS_FLOOR) / (RMS_REF - RMS_FLOOR)
        rms_norm = np.clip(rms_norm, 0.0, 1.0)

        # --------------------------------------------------
        # 4. Normalize motion (FIXED SCALE)
        # --------------------------------------------------
        MOTION_REF = self.cfg.MOTION_REF
        # MOTION_FLOOR = 0.0010
        MOTION_FLOOR = self.compute_adaptive_motion_floor(env_motion)

        motion_norm = max(0.0, env_motion - MOTION_FLOOR)
        motion_norm = motion_norm / MOTION_REF
        motion_norm = np.clip(motion_norm, 0.0, 1.0)

        # --------------------------------------------------
        # 5. Activity = loud AND moving
        # --------------------------------------------------
        raw_activity = rms_norm * motion_norm

        # --------------------------------------------------
        # 6. Temporal smoothing (no accumulation!)
        # --------------------------------------------------
        if not hasattr(self, "_activity_energy"):
            self._activity_energy = raw_activity
        else:
            alpha = self.cfg.SMOOTHING_ALPHA
            self._activity_energy = (
                    alpha * raw_activity +
                    (1.0 - alpha) * self._activity_energy
            )

        # --------------------------------------------------
        # 7. Debug
        # --------------------------------------------------
        if getattr(self, "debug", False):
            print(
                f"activity | rms={rms:.4f} "
                f"rms_n={rms_norm:.3f} "
                f"motion={env_motion:.6f} "
                f"motion_n={motion_norm:.3f} "
                f"activity={self._activity_energy:.3f}"
            )

        return float(self._activity_energy)

    def compute_adaptive_motion_floor(self, env_motion):
        # collect history
        self.motion_history.append(env_motion)

        if len(self.motion_history) < 20:
            # bootstrap: fallback fixed floor
            return 0.0010

        hist = np.array(self.motion_history)

        # estimate noise floor from quiet moments
        noise_motion = np.percentile(hist, 20)

        # adaptive floor with safety margin
        floor = noise_motion * 1.8

        # hard safety clamps
        floor = np.clip(floor, 0.0005, 0.0030)

        return floor
