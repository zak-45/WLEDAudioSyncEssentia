# src/analysis_process_core.py

"""Core engine for turning live audio into genres, mood, and lighting colors.

This module implements the main analysis loop that consumes audio chunks,
classifies genres, estimates mood and energy, and drives OSC-controlled
lighting in real time. It combines neural network models, adaptive buffering,
and color mapping logic to produce stable yet responsive visual feedback that
tracks the character of the music.
"""

import json
import queue
import time

import numpy as np

from configmanager import root_path
from src.effnet_classifier import EffnetClassifier, AuxClassifier
from src.macro_genres import collapse_to_macro
from src.model_loader import discover_models
from src.smoothing import GenreSmoother
from src.mood_color_mapper import MoodColorMapper
from src.adaptive_buffer import AdaptiveBuffer
from src.utils import compute_color
from src.genre_color_profile_loader import load_genre_color_profiles
from src.emotion_color_mapper import EmotionColorMapper

rt_color_mapper = EmotionColorMapper(
    mood_image_path="assets/music_color_mood.png",
    smoothing=0.85   # recommended for LEDs
)

with open(root_path("config/genre_flash_shape.json"), "r") as f:
    GENRE_FLASH_SHAPES = json.load(f)

# fetch all models from folder
models = discover_models(root_path("models"))
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
        activate_buffer,
        rt_mood_lift,
    ):
        self.rt_mood_lift = rt_mood_lift
        self.accent_strength = 0.0
        self.aux = aux
        self.danceability = None
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
                audio, rms_rt, ts, activity_energy, beat, is_silent = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # --------------------------------------------------
            # Silence detection
            # --------------------------------------------------
            if is_silent:
                self._enter_silence()
                continue


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
                if self.debug:
                    print('Prob is None')
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

            if self.debug:
                print(
                    "[PROFILE DEBUG]",
                    f"macro={macro_label}",
                    f"hue={profile.hue}",
                    f"bright_floor={profile.bright_floor}",
                    f"sat_floor={profile.sat_floor}",
                )

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


            # --------------------------------------------------
            # Perceptual brightness proxy (for valence only)
            # --------------------------------------------------

            # brightness must NOT be energy-driven

            if self.danceability is not None:
                perceptual_brightness = np.clip(
                    profile.bright_floor +
                    0.6 * top_conf +  # genre certainty
                    0.4 * self.danceability,  # musical feel (if available)
                    0.0, 1.0
                )
            else:
                perceptual_brightness = np.clip(
                    profile.bright_floor +
                    0.5 * top_conf,
                    0.0, 1.0
                )

            if self.debug:
                print(
                    "[BRIGHTNESS DEBUG]",
                    f"profile_floor={profile.bright_floor:.3f}",
                    f"activity_energy={activity_energy:.3f}",
                    f"computed={perceptual_brightness:.3f}",
                )

                print(
                    f"[INPUT DEBUG] "
                    f"p_bright={perceptual_brightness:.3f} "
                    f"energy={activity_energy:.3f} "
                    f"profile_floor={profile.bright_floor:.3f} "
                    f"boost={profile.energy_boost:.3f}"
                )

            valence = self.mood_mapper.compute_valence(top5, perceptual_brightness, activity_energy, top_conf)

            if self.debug:
                print(
                    f"VALENCE INPUTS | "
                    f"p_bright={perceptual_brightness:.3f} "
                    f"energy={activity_energy:.3f} "
                    f"conf={top_conf:.3f}"
                )

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

            # Update accent strength once
            self.update_accent_strength(
                beat=beat,
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
            # Mood like RTMood
            # --------------------------------------------------
            # --------------------------------------------------
            # RTMood perceptual expansion
            # Valence is intentionally conservative (~0.4–0.6),
            # so we expand contrast ONLY for RTMood mapping.
            # --------------------------------------------------

            # --------------------------------------------------
            # RTMood energy protection
            # RTMood collapses to black at energy == -1
            # So we keep a perceptual floor
            # --------------------------------------------------

            v = float(valence)

            # expand around neutral (0.5 → 0.0)
            v_expanded = (0.5 + np.sign(v - 0.5) * abs(v - 0.5) ** 0.6)
            v_expanded = float(np.clip(v_expanded, 0.0, 1.0))

            soft_valence = (v_expanded * 2.0) - 1.0

            # --------------------------------------------------
            # RTMood-safe energy mapping
            # RTMood collapses chroma near ±1
            # --------------------------------------------------

            # map [0,1] → [-1,1]
            soft_energy = (activity_energy * 2.0) - 1.0

            # perceptual clamp (CRITICAL)
            SOFT_E_MAX = 0.75
            SOFT_E_MIN = -0.75

            soft_energy = float(np.clip(soft_energy, SOFT_E_MIN, SOFT_E_MAX))

            if self.debug:
                print(
                    f"[RTMOOD INPUT] "
                    f"valence={valence:.3f} "
                    f"expanded={v_expanded:.3f} "
                    f"soft_v={soft_valence:.3f} "
                    f"soft_e={soft_energy:.3f}"
                )

            rtr, rtg, rtb = rt_color_mapper.get_rgb(soft_valence, soft_energy)

            if self.rt_mood_lift:
                # optional RTMood luminance lift
                rt_lift = 0.4 + 0.6 * activity_energy

                rtr = int(rtr * rt_lift)
                rtg = int(rtg * rt_lift)
                rtb = int(rtb * rt_lift)

                rtr = min(255, rtr)
                rtg = min(255, rtg)
                rtb = min(255, rtb)

            self.osc.send("/WASEssentia/mood/rt/color/r", rtr / 255.0)
            self.osc.send("/WASEssentia/mood/rt/color/g", rtg / 255.0)
            self.osc.send("/WASEssentia/mood/rt/color/b", rtb / 255.0)

            if self.debug:
                print("RTMood color:", rtr, rtg, rtb)

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
        self.smooth.reset()
        self.adaptive.reset()
        self.adaptive.current = self.cfg.MIN_BUFFER_SECONDS
        self.mood_mapper.reset_valence()
        self.osc.send_silence(0)

        if self.debug:
            print("🔇 SILENCE → analysis reset")

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
        shape = GENRE_FLASH_SHAPES.get(genre, GENRE_FLASH_SHAPES.get("DEFAULT", "pulse"))
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

