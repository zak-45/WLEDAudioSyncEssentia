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

from src.ring_buffer import RingBuffer

from src.emotion_mapper_aux import EmotionMapperAUX, clamp, StrobeController
from src.effect_config_manager import EffectConfigManager

config_mgr = EffectConfigManager(root_path("config/effects"))
emotion = EmotionMapperAUX(config_mgr)

strobe_ctrl = StrobeController()

from src.emotion_debug_cv2 import EmotionDebugCV2

with open(root_path("config/genre_flash_shape.json"), "r") as f:
    GENRE_FLASH_SHAPES = json.load(f)


class AnalysisCore:
    def __init__(
            self,
            audio_queue,
            cfg,
            osc,
            visual,
            use_macro,
            macro_agg,
            debug,
            aux,
            activate_buffer,
            aux_mood,
            shutdown_event=None,
            silent_event=None,
    ):
        self.shutdown_event = shutdown_event
        self.silent_event = silent_event
        self.aux_mood = aux_mood
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
        self.use_macro = use_macro
        self.macro_agg = macro_agg
        self.is_silent = False
        self.adaptive_buffer = activate_buffer
        self.models = []

        # -------- MODELS (SAFE HERE) --------
        self.clf = EffnetClassifier()

        self.smooth = GenreSmoother(
            self.clf.labels,
            cfg.SMOOTHING_ALPHA
        )

        if self.aux_mood:
            self.models = discover_models(root_path("models/mood"))
            self.load_aux()
            self.smooth.alpha = 0.2
            if self.cfg.AUX_MOOD_VISUAL:
                self.emotion_visual = EmotionDebugCV2(size=520)

        elif self.aux:

            self.models = discover_models(root_path("models"))
            self.load_aux()


        self.mood_mapper = MoodColorMapper(
            root_path("models/genre_discogs400-discogs-effnet-1.json"),
            root_path("config/mood_valence.json")
        )

        self.buffer = np.zeros(0, dtype=np.float32)

        self.adaptive = AdaptiveBuffer(
            cfg.MIN_BUFFER_SECONDS,
            cfg.MAX_BUFFER_SECONDS,
            cfg.BUFFER_SECONDS,
            cfg.CONFIDENCE_THRESHOLD,
            cfg.STABILITY_FRAMES,
            cfg.BUFFER_GROWTH_RATE,
            cfg.BUFFER_SHRINK_RATE,
        )

        self.ring_buffer = RingBuffer(
            capacity_seconds=cfg.RING_BUFFER_CAPACITY,
            sample_rate=cfg.MODEL_SAMPLE_RATE,
            min_analysis_seconds=cfg.EFFNET_MIN_DURATION,
            hop_seconds=cfg.RING_BUFFER_HOP
        )

        self.fast_ring_buffer = RingBuffer(
            capacity_seconds=2.1,
            sample_rate=cfg.MODEL_SAMPLE_RATE,
            min_analysis_seconds=cfg.EFFNET_MIN_DURATION,
            hop_seconds=0.2
        )

        self.genre_profiles, self.default_profile = load_genre_color_profiles(
            root_path("config/genre_color_profiles.json")
        )

    # -----------------------------------------------------

    def run(self):

        model_rate = self.cfg.MODEL_SAMPLE_RATE

        hop = int(model_rate * self.cfg.HOP_SECONDS)
        base_prefix = 'MOOD' if self.aux_mood else ''
        needed_seconds = self.cfg.EFFNET_MIN_DURATION
        use_ring_buffer = self.cfg.RING_BUFFER_ACTIVATE

        adaptive_max_samples = int(self.cfg.MAX_BUFFER_SECONDS * model_rate)
        aux_mood_visual = self.cfg.AUX_MOOD_VISUAL

        if not self.adaptive_buffer and not use_ring_buffer:
            self.smooth.alpha = 0.0

        if use_ring_buffer:
            self.adaptive_buffer = False

        while True:

            # Check shutdown event
            if self.shutdown_event and self.shutdown_event.is_set():
                print("🛑 Analysis process shutting down...")
                if self.visual:
                    try:
                        self.visual.close()
                        self.emotion_visual.close()
                    except:
                        pass
                break

            if self.silent_event and self.silent_event.is_set():
                self._enter_silence()
                continue

            try:
                audio, rms_rt, ts, activity_energy, beat, is_silent = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            current_time = time.strftime("%H:%M:%S")
            prefix = f"[{current_time}] {base_prefix}" if base_prefix else f"[{current_time}]"

            # --------------------------------------------------
            # Silence detection
            # --------------------------------------------------
            if is_silent:
                self._enter_silence()
                continue

            # --------------------------------------------------
            # Buffer selection
            # --------------------------------------------------

            if use_ring_buffer:
                # ========================================
                # Ring Buffer
                # ========================================
                # add audio data to buffer
                self.ring_buffer.append(audio)

                # enough data to analyze ?
                if not self.ring_buffer.should_analyze():
                    # not enough
                    continue

                # choose data to analyze
                segment = self.ring_buffer.get_analysis_segment(needed_seconds)

                if segment is None:
                    if self.debug:
                        print(f'{prefix} Segment is None -- waiting for more data')
                    continue

            elif self.adaptive_buffer:

                # --------------------------------------------------
                # ADAPTIVE GENRE BUFFER
                # --------------------------------------------------

                self.buffer = np.concatenate([self.buffer, audio])

                adaptive_needed = int(model_rate * self.adaptive.current)

                if len(self.buffer) < adaptive_needed:
                    continue

                segment = self.buffer[-adaptive_needed:]

            else:

                # --------------------------------------------------
                # FAST DEFAULT BUFFER
                # --------------------------------------------------
                # add audio data to buffer
                self.fast_ring_buffer.append(audio)

                # enough data to analyze ?
                if not self.fast_ring_buffer.should_analyze():
                    # not enough
                    continue

                # choose data to analyze
                segment = self.fast_ring_buffer.get_analysis_segment(needed_seconds)

                if segment is None:
                    if self.debug:
                        print(f'{prefix} Segment is None -- waiting for more data')
                    continue

            # --------------------------------------------------
            # Genre classification
            # --------------------------------------------------
            probs = self.clf.classify(segment)
            if probs is None:
                if self.debug:
                    print('Prob is None -- check buffer size ???')
                continue

            self.smooth.update(probs)
            top5 = self.smooth.top_n(5)

            top_label, top_conf = top5[0]
            macro_label = top_label.split("---")[0]

            # adapt buffer size if not use macro genre
            if not self.use_macro and self.adaptive_buffer:
                self.adaptive.update(
                    top_label=top_label,
                    confidence=top_conf,
                    silent=False
                )

            # load genre profile params from JSON
            profile = self.genre_profiles.get(macro_label, self.default_profile)

            if self.debug:
                print(
                    f"[{prefix} PROFILE DEBUG]",
                    f"macro={macro_label}",
                    f"hue={profile.hue}",
                    f"bright_floor={profile.bright_floor}",
                    f"sat_floor={profile.sat_floor}",
                )

            # color
            genre_hue = profile.hue

            # energy boost
            activity_energy = np.clip(
                activity_energy * profile.energy_boost,
                0.0, 1.0
            )

            if self.debug:
                print(f"{prefix} GENRE TOP5: ",
                      " | ".join(f"{g}:{v:.5f}" for g, v in top5))
                print(f"{prefix} GENRE CONF: ", top_conf)

            # --------------------------------------------------
            # OSC genre labels
            # --------------------------------------------------
            path = "/WASEssentia/genre/mood_top" if self.aux_mood else "/WASEssentia/genre/top"
            for i, (label, _) in enumerate(top5):
                self.osc.send(f"{path}{i}", label)

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
                if self.adaptive_buffer:
                    self.adaptive.update(
                        top_label=top_label_macro,
                        confidence=top_conf_macro,
                        silent=False
                    )

                if self.debug:
                    print(f"{prefix} MACRO TOP5: ",
                          " | ".join(f"{g}:{v:.5f}" for g, v in top5_macro))

                    print(f"{prefix} MACRO CONF: ", top_conf_macro)

                # --------------------------------------------------
                # OSC macro genre labels
                # --------------------------------------------------
                path = "/WASEssentia/genre/macro_top" if self.aux_mood else "/WASEssentia/genre/mood_macro_top"
                for i, (label, _) in enumerate(top5_macro):
                    self.osc.send(f"{path}{i}", label)

            # --------------------------------------------------
            # AUX classifiers
            # --------------------------------------------------
            if self.aux_classifiers:
                aux_dict = {}
                embeddings = self.clf.compute_embeddings(segment)
                for aux in self.aux_classifiers:
                    results = aux.classify(embeddings)
                    if results is None:
                        continue

                    if aux.name == "danceability classifier":
                        self.danceability = float(results.get("danceable", 0.5))

                    if self.aux_mood:
                        aux_intensity = (
                                0.2 * activity_energy +
                                0.8 * self.danceability
                        )
                        aux_intensity = clamp(aux_intensity, 0, 1.0)
                        aux_dict["danceable"] = aux_intensity
                        if aux.name == "mood aggressive":
                            aux_dict["aggressive"] = float(results.get("aggressive", 0.5))
                        elif aux.name == "mood happy":
                            aux_dict["happy"] = float(results.get("happy", 0.5))
                        elif aux.name == "mood relaxed":
                            aux_dict["relaxed"] = float(results.get("relaxed", 0.5))
                        elif aux.name == "mood sad":
                            aux_dict["sad"] = float(results.get("sad", 0.5))

                    if self.debug:
                        print('_|_')
                        print(f"{prefix} AUX {aux.name} =", " | ".join(f"{k}:{v:.5f}" for k, v in results.items()))

                    # --------------------------------------------------
                    # OSC aux labels
                    # --------------------------------------------------
                    for label, value in results.items():
                        path = f"/WASEssentia/aux/{aux.name.replace(' ', '_')}/{label.replace(' ', '_')}"
                        self.osc.send(path, float(value))

                if self.aux_mood:
                    # Update context when known
                    emotion.update_context(
                        genre=macro_label.lower()
                    )
                    if self.debug:
                        print(f'[{prefix} PRESET] {emotion.effects.config["_meta"]}')

                    valence, arousal, intensity = emotion.compute_emotion(aux_dict)
                    wled_data = emotion.emotion_to_wled(valence, arousal, intensity)

                    now_ms = time.time() * 1000
                    if strobe_ctrl.update(valence, arousal, intensity, now_ms):
                        effect = "Strobe"
                        print('-----------STROBE----------')
                    # else:
                    #    effect = normal_effect

                    if aux_mood_visual:
                        self.emotion_visual.draw(
                            genre=top_label,
                            valence=valence,
                            arousal=arousal,
                            intensity=intensity,
                            emotion_label=emotion.emotion_label(valence, arousal)[0],
                            effect=wled_data["effect"],
                            profile=profile
                        )

                    path_r = "/WASEssentia/aux/mood/color/r"
                    value_r = wled_data['rgb'][0]
                    self.osc.send(path_r, value_r / 255.0)
                    path_g = "/WASEssentia/aux/mood/color/g"
                    value_g = wled_data['rgb'][1]
                    self.osc.send(path_g, value_g / 255.0)
                    path_b = "/WASEssentia/aux/mood/color/b"
                    value_b = wled_data['rgb'][2]
                    self.osc.send(path_b, value_b / 255.0)

                    aux_mood_data = json.dumps({
                        "valence": round(valence, 3),
                        "arousal": round(arousal, 3),
                        "intensity": round(intensity, 3),
                        "R": value_r,
                        "G": value_g,
                        "B": value_b
                    })

                    if self.debug:
                        print(f"[{prefix} AUX MOOD DATA] {aux_mood_data}")

                    self.osc.send(
                        "/WASEssentia/aux/mood/data",
                        aux_mood_data
                    )

                    self.osc.send(
                        "/WASEssentia/aux/mood/effect",
                        wled_data["effect"]
                    )

                    self.osc.send(
                        "/WASEssentia/aux/mood/index",
                        wled_data["index"]
                    )

            if not self.aux_mood:
                # --------------------------------------------------
                # Mood computation (other than AUX)
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
                        f"[{prefix} BRIGHTNESS DEBUG]",
                        f"profile_floor={profile.bright_floor:.3f}",
                        f"activity_energy={activity_energy:.3f}",
                        f"computed={perceptual_brightness:.3f}",
                    )

                    print(
                        f"[{prefix} INPUT DEBUG] "
                        f"p_bright={perceptual_brightness:.3f} "
                        f"energy={activity_energy:.3f} "
                        f"profile_floor={profile.bright_floor:.3f} "
                        f"boost={profile.energy_boost:.3f}"
                    )

                valence = self.mood_mapper.compute_valence(top5, perceptual_brightness, activity_energy, top_conf)

                if self.debug:
                    print(
                        f"{prefix} VALENCE INPUTS | "
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
                    print(f"{prefix} Genre color:", r, g, b)

                self.osc.send("/WASEssentia/genre/color/r", r / 255.0)
                self.osc.send("/WASEssentia/genre/color/g", g / 255.0)
                self.osc.send("/WASEssentia/genre/color/b", b / 255.0)

                # --------------------------------------------------
                # Mood color genre-centric override
                # --------------------------------------------------
                #
                r, g, b = compute_color(mood_hue, saturation, brightness)

                if self.debug:
                    print(f"{prefix} Mood color:", r, g, b)

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
                    print(f"{prefix} Final color:", r, g, b)

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
                    print(f"{prefix} Accent color:", accent_r, accent_g, accent_b)

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
                    print(f"[{prefix} MOOD DATA] {mood_data}")


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
            if not use_ring_buffer:
                if self.adaptive_buffer:
                    # Preserve enough history for confirmed classification
                    self.buffer = self.buffer[-hop:]
                self.buffer = self.buffer[-adaptive_max_samples:]

    # ==================================================
    def _enter_silence(self):
        self.is_silent = True
        self.silent_event.clear()

        while True:
            try:
                # Non-blocking get
                audio, rms_rt, ts, activity_energy, beat, is_silent = self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.buffer = np.zeros(0, dtype=np.float32)

        self.smooth.reset()
        self.adaptive.reset()
        self.adaptive.current = self.cfg.MIN_BUFFER_SECONDS
        self.mood_mapper.reset_valence()
        self.osc.send_silence(0)

        self.ring_buffer.reset()

        if self.debug:
            print("🔇 SILENCE → analysis reset")

    # ==================================================
    def load_aux(self):
        self.aux_classifiers = []
        # load models and set them to list for type AUX
        for mod in self.models:
            if mod["type"] == "genre":

                if self.debug:
                    print(f"🎵 Genre model loaded: {mod['name']}")

            else:

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