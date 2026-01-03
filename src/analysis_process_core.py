# src/analysis_process_core.py
import json
import queue
import time
import numpy as np

from src.effnet_classifier import EffnetClassifier, AuxClassifier
from src.genre_hues import GENRE_HUES
from src.macro_genres import collapse_to_macro
from src.model_loader import discover_models
from src.smoothing import GenreSmoother
from src.mood_color_mapper import MoodColorMapper
from src.adaptive_buffer import AdaptiveBuffer
from src.utils import compute_color

# fetch all models from folder
models = discover_models("models")
#
DEBUG_DATA = True
AUX = True
MACRO_AGG = "mean"

aux_classifiers = []
# load models and set them to list for type AUX
for m in models:
    if m["type"] == "genre":

        if DEBUG_DATA:
            print(f"🎵 Genre model loaded: {m['name']}")

    else:

        if AUX:
            aux_classifiers.append(
                AuxClassifier(m["name"], m["pb"], m["json"], m["output_name"], agg=MACRO_AGG)
            )

            if DEBUG_DATA:
                print(f"🎛 Aux model loaded: {m['name']}")


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
    ):
        self.danceability = 0.5
        self.aux_classifiers = aux_classifiers
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

        # -------- MODELS (SAFE HERE) --------
        self.clf = EffnetClassifier()
        self.smooth = GenreSmoother(
            self.clf.labels,
            cfg.SMOOTHING_ALPHA
        )

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
            # Buffering
            # --------------------------------------------------
            self.buffer = np.concatenate([self.buffer, audio])

            needed = int(self.cfg.MODEL_SAMPLE_RATE * self.adaptive.current)
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
            macro = top_label.split("---")[0]
            genre_hue = GENRE_HUES.get(macro, 270)

            self.adaptive.update(
                top_label=top_label,
                confidence=top_conf,
                silent=False
            )

            if self.debug:
                print("GENRE TOP5:",
                      " | ".join(f"{g}:{v:.5f}" for g, v in top5))

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

                # top-5 macro genres
                top5_macro = sorted(
                    macro_probs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                top_label_macro, top_conf_macro = top5_macro[0]

                self.adaptive.update(
                    top_label=top_label_macro,
                    confidence=top_conf_macro,
                    silent=False
                )

                if self.debug:
                    print("MACRO TOP5:",
                          " | ".join(f"{g}:{v:.5f}" for g, v in top5_macro))

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

            energy = self.mood_mapper.compute_energy(
                danceability=self.danceability,
                rms_rt=rms_rt,
                top_genre=macro,
                confidence=top_conf
            )

            mood_hue = self.mood_mapper.mood_to_hue(valence, energy)

            # --------------------------------------------------
            # Genre color
            # --------------------------------------------------
            #
            genre_brightness = max(0.1, min(1.0, energy))
            genre_saturation = min(1.0, top_conf * 1.5)

            r, g, b = compute_color(genre_hue, genre_saturation, genre_brightness)

            if self.debug:
                print("Genre color:", r, g, b)

            self.osc.send("/WASEssentia/genre/color/r", r / 255.0)
            self.osc.send("/WASEssentia/genre/color/g", g / 255.0)
            self.osc.send("/WASEssentia/genre/color/b", b / 255.0)

            # --------------------------------------------------
            # Mood color genre-centric override
            # --------------------------------------------------
            #
            r, g, b = compute_color(mood_hue, genre_saturation, genre_brightness)

            if self.debug:
                print("Mood color:", r, g, b)

            self.osc.send("/WASEssentia/mood/color/r", r / 255.0)
            self.osc.send("/WASEssentia/mood/color/g", g / 255.0)
            self.osc.send("/WASEssentia/mood/color/b", b / 255.0)

            # --------------------------------------------------
            # Final hue + colors
            # --------------------------------------------------
            final_hue = self.mood_mapper.fuse_hues(
                genre_hue=genre_hue,
                mood_hue=mood_hue,
                confidence=top_conf
            )

            # Authoritative production color
            r, g, b = self.mood_mapper.final_color(
                genre_hue=genre_hue,
                mood_hue=mood_hue,
                confidence=top_conf,
                energy=energy
            )

            if self.debug:
                print("Final color:", r, g, b)

            # Accent color
            accent_r, accent_g, accent_b = self.mood_mapper.accent_color(
                final_hue=final_hue,
                energy=energy,
                confidence=top_conf
            )

            # Debug / genre-centric override
            if self.color1:
                genre_brightness = max(0.1, min(1.0, energy))
                genre_saturation = min(1.0, top_conf * 1.5)
                r, g, b = compute_color(final_hue, genre_saturation, genre_brightness)

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
                    "energy": round(energy, 3),
                    "R": r,
                    "G": g,
                    "B": b
                })
            )

            # --------------------------------------------------
            # Visual debug
            # --------------------------------------------------
            if self.visual:
                self.visual.update(
                    genre=macro,
                    genre_hue=genre_hue,
                    mood_hue=mood_hue,
                    final_hue=final_hue,
                    valence=valence,
                    energy=energy,
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
        self.osc.send_silence(0)

        if self.debug:
            print("🔇 SILENCE → analysis reset")

    def _exit_silence(self):
        self.is_silent = False
        self.osc.send_silence(1)

        if self.debug:
            print("🎵 AUDIO RESUMED")
