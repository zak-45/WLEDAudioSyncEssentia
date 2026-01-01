import json

import numpy as np
import argparse
import time
import threading

from configmanager import *

from src.utils import resample, compute_color
from src.audio_stream import AudioStream
from src.smoothing import GenreSmoother
from src.osc_sender import OSCSender
from src.macro_genres import collapse_to_macro
from src.model_loader import discover_models
from src.effnet_classifier import EffnetClassifier, AuxClassifier
from src.adaptive_buffer import AdaptiveBuffer
from src.mood_color_mapper import MoodColorMapper
from src.genre_hues import GENRE_HUES
from src.visual_debug import VisualDebugOverlay
from src.runtime_config import RuntimeConfig

cfg = RuntimeConfig("config/audio_runtime.json")

AUDIO_DEVICE_RATE = cfg.AUDIO_DEVICE_RATE
MODEL_SAMPLE_RATE = cfg.MODEL_SAMPLE_RATE
MIN_RMS = cfg.MIN_RMS

BUFFER_SECONDS = cfg.BUFFER_SECONDS
MIN_BUFFER_SECONDS = cfg.MIN_BUFFER_SECONDS
MAX_BUFFER_SECONDS = cfg.MAX_BUFFER_SECONDS
HOP_SECONDS = cfg.HOP_SECONDS

SMOOTHING_ALPHA = cfg.SMOOTHING_ALPHA

parser = argparse.ArgumentParser()

CONFIDENCE_THRESHOLD = cfg.CONFIDENCE_THRESHOLD
STABILITY_FRAMES = cfg.STABILITY_FRAMES
BUFFER_GROWTH_RATE = cfg.BUFFER_GROWTH_RATE
BUFFER_SHRINK_RATE = cfg.BUFFER_SHRINK_RATE

parser.add_argument(
    "--macro",
    action="store_true",
    help="Collapse Discogs subgenres into macro genres"
)

parser.add_argument(
    "--macro-agg",
    choices=["mean", "max"],
    default="mean",
    help="Aggregation method for macro genres (default: mean)"
)

parser.add_argument(
    "--osc-ip",
    default="127.0.0.1",
    help="OSC server IP address (default: 127.0.0.1)"
)

parser.add_argument(
    "--osc-port",
    type=int,
    default=12000,
    help="OSC server port (default: 12000)"
)

parser.add_argument(
    "--osc-path",
    default="/genre",
    help="OSC message path (default: /genre)"
)

parser.add_argument(
    "--device-index",
    type=int,
    default=None,
    help="PyAudio input device index (default: system default)"
)

parser.add_argument(
    "--channels",
    type=int,
    default=2,
    help="PyAudio input device channel number (default: 2)"
)

parser.add_argument(
    "--show_raw",
    action="store_true",
    help="If present, Print RAW values"
)


parser.add_argument(
    "--aux",
    action="store_true",
    help="If present, show AUX values"
)

parser.add_argument(
    "--visual",
    action="store_true",
    help="Enable visual debug overlay"
)

parser.add_argument(
    "--debug",
    action="store_true",
    help="Print debug datas"
)

parser.add_argument(
    "--color1",
    action="store_true",
    help="Choose color type 1 for final hue"
)
parser.add_argument(
    "--color2",
    action="store_true",
    help="Choose color type 2 for final hue"
)


args = parser.parse_args()

COLOR1=args.color1
COLOR2=args.color2

VISUAL_DEBUG = args.visual
DEBUG_DATA = args.debug

visual = VisualDebugOverlay() if VISUAL_DEBUG else None

is_silent = False
last_non_silent_time = time.time()

adaptive = AdaptiveBuffer(
    MIN_BUFFER_SECONDS,
    MAX_BUFFER_SECONDS,
    BUFFER_SECONDS,
    CONFIDENCE_THRESHOLD,
    STABILITY_FRAMES,
    BUFFER_GROWTH_RATE,
    BUFFER_SHRINK_RATE
)

NEEDED = int(MODEL_SAMPLE_RATE * adaptive.current)

HOP = int(MODEL_SAMPLE_RATE * HOP_SECONDS)

MODELS_DIR = root_path("models")

USE_MACRO_GENRES = args.macro
MACRO_AGG = args.macro_agg

OSC_IP = args.osc_ip
OSC_PORT = args.osc_port
OSC_PATH = args.osc_path

DEVICE_INDEX = args.device_index
CHANNELS = args.channels

PRINT_RAW = args.show_raw
AUX = args.aux

SILENCE_TIMEOUT = cfg.SILENCE_TIMEOUT  # seconds

buffer = np.zeros(0, dtype=np.float32)

# Main Effnet discogs model with Music Genre Detection
clf = EffnetClassifier()

# Smoother
smooth = GenreSmoother(clf.labels, SMOOTHING_ALPHA)

# OSC Sender
osc = OSCSender(
    ip=OSC_IP,
    port=OSC_PORT,
    path=OSC_PATH
)

if DEBUG_DATA:
    print(
        f"🎛 OSC → {OSC_IP}:{OSC_PORT} {OSC_PATH}"
    )

# Brightness & Saturation for Color
g_brightness = 0.5 # default, calculate from  Energy
g_saturation = 0.5 # default, calculate from top genre min(1.0, top1_prob * 1.5)

# mood color, read JSON for genre labels
mood_mapper = MoodColorMapper("models/genre_discogs400-discogs-effnet-1.json",
                              "config/mood_valence.json")

def extract_macro(label):
    return label.split("---")[0]

def top_n_from_probs(probs, labels, n=5):
    idx = probs.argsort()[::-1][:n]
    return [(labels[i], float(probs[i])) for i in idx]

def enter_silence():
    global buffer, is_silent
    is_silent = True

    buffer = np.zeros(0, dtype=np.float32)
    smooth.reset()
    adaptive.reset()
    adaptive.current = MIN_BUFFER_SECONDS

    osc.send_silence(0)
    print("🔇 SILENCE → state reset")

def exit_silence():
    global is_silent
    is_silent = False

    osc.send_silence(1)
    print("🎵 AUDIO RESUMED")

def on_audio(audio, rms_rt):
    global buffer, last_non_silent_time, g_saturation, g_brightness

    # stereo → mono
    if audio.size % 2 == 0:
        audio = audio.reshape(-1, 2).mean(axis=1)

    # resample
    audio = resample(audio, AUDIO_DEVICE_RATE, MODEL_SAMPLE_RATE)

    buffer = np.concatenate([buffer, audio])

    if len(buffer) < NEEDED:
        return

    segment = buffer[-NEEDED:]

    # silent
    now = time.time()

    if rms_rt < MIN_RMS:
        if not is_silent and (now - last_non_silent_time) >= SILENCE_TIMEOUT:
            enter_silence()
        return
    else:
        last_non_silent_time = now
        if is_silent:
            exit_silence()

    probs = clf.classify(segment)
    if probs is None:
        return

    if not is_silent:

        if DEBUG_DATA:
            print('_' * 80)

        if PRINT_RAW and DEBUG_DATA:
            print("RAW max:", probs.max())

            raw_top5 = top_n_from_probs(probs, clf.labels, 5)

            print("RAW TOP5:",
                  " | ".join(f"{g}:{v:.4f}" for g, v in raw_top5))

        smooth.update(probs)
        top5 = smooth.top_n(5)

        top_label, top_conf = top5[0]

        macro = extract_macro(top_label)
        genre_hue = GENRE_HUES.get(macro, 270)  # fallback purple

        adaptive.update(
            top_label=top_label,
            confidence=top_conf,
            silent=False
        )

        if DEBUG_DATA:
            print("GENRE TOP5:",
                  " | ".join(f"{g}:{v:.5f}" for g, v in top5))

        i=0
        for label, value in top5:
            osc.send(f"/WASEssentia/genre/top{i}", label)
            i+=1

        if USE_MACRO_GENRES:
            macro_probs = collapse_to_macro(probs, clf.labels, agg=MACRO_AGG)

            # top-5 macro genres
            top5_macro = sorted(
                macro_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            top_label_macro, top_conf_macro = top5_macro[0]

            adaptive.update(
                top_label=top_label_macro,
                confidence=top_conf_macro,
                silent=False
            )

            if DEBUG_DATA:
                print("MACRO TOP5:",
                      " | ".join(f"{g}:{v:.5f}" for g, v in top5_macro))

            i=0
            for label, value in top5:
                osc.send(f"/WASEssentia/genre/macro_top{i}/", label)
                i+=1

        # --------------------------------------------------
        # AUX classifiers
        # --------------------------------------------------
        if AUX:
            if DEBUG_DATA:
                print('_' * 80)

            aux_results = {}
            embeddings = clf.compute_embeddings(segment)
            for aux in aux_classifiers:
                results = aux.classify(embeddings)
                if results is not None:
                    aux_results.update(results)
                    # energy --> brightness
                    if aux.name == 'danceability classifier':
                        g_brightness = max(0.1, min(1.0, float(aux_results.get('danceable'))))
                    # sort by value
                    aux_results_sorted = sorted(
                        aux_results.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    if DEBUG_DATA:
                        print('_|_')
                        print(f"AUX {aux.name} =", " | ".join(f"{k}:{v:.5f}" for k, v in aux_results_sorted))

                    for label, value in aux_results.items():
                        path = f"/WASEssentia/aux/{aux.name.replace(' ', '_')}/{label.replace(' ', '_')}"
                        osc.send(path, float(value))

                    aux_results.clear()

        # --------------------------------------------------
        # Mood computation
        # --------------------------------------------------
        # --- TOP GENRES (already computed) ---
        # top_genres = [('Latin---Reggaeton', 0.623), ('Hip Hop---Trap', 0.135), ...]

        top1_label, top1_prob = top5[0]

        valence = mood_mapper.compute_valence(top5)

        energy = mood_mapper.compute_energy(
            danceability=g_brightness,
            rms_rt=rms_rt,
            top_genre=top1_label.split("---")[0],
            confidence=top1_prob
        )

        mood_hue = mood_mapper.mood_to_hue(valence, energy)

        r, g, b = mood_mapper.mood_to_rgb(
            valence=valence,
            energy=energy,
            confidence=top1_prob
        )

        # --------------------------------------------------
        # OSC JSON (Chataigne friendly)
        # --------------------------------------------------
        osc_color_data = {
            "valence": round(valence, 3),
            "energy": round(energy, 3),
            "R": r,
            "G": g,
            "B": b
        }

        if DEBUG_DATA:
            print('_|_')
            print('Mood  color:', r,g,b)

        osc.send(
            "/WASEssentia/mood/color",
            json.dumps(osc_color_data)
        )

        # Now, we have all datas to calculate the genre color
        # --------------------------------------------------
        # Genre color
        # --------------------------------------------------
        # saturation brightness  --> use default or data from Mood calculation
        g_brightness = max(0.1, min(1.0, float(energy)))
        # take the most relevant genre value
        g_saturation = min(1.0, top1_prob * 1.5)

        r, g, b = compute_color(genre_hue,g_saturation,g_brightness)

        if  DEBUG_DATA:
            print("Genre color:", r,g,b)

        osc.send("/WASEssentia/genre/color/r", r / 255.0)
        osc.send("/WASEssentia/genre/color/g", g / 255.0)
        osc.send("/WASEssentia/genre/color/b", b / 255.0)

        # --------------------------------------------------
        # Final blended hue
        # --------------------------------------------------

        # fuse hues
        # final_hue = (0.7 * genre_hue + 0.3 * mood_hue) % 360
        final_hue = mood_mapper.fuse_hues(
            genre_hue=genre_hue,
            mood_hue=mood_hue,
            confidence=top1_prob
        )

        # convert final hue to RGB

        if COLOR1:
            # saturation & brightness already computed
            r, g, b = compute_color(
                final_hue,
                g_saturation,
                g_brightness
            )

        elif COLOR2:
            # complex algo
            r, g, b = mood_mapper.final_color(
                genre_hue=genre_hue,
                mood_hue=mood_hue,
                confidence=top1_prob,
                energy=energy
            )

        else:
            # simple algo
            r, g, b = mood_mapper._hsv_to_rgb(
                final_hue,
                int(80 + 175 * top1_prob),
                int(60 + 195 * energy)
            )

        if DEBUG_DATA:
            print('Blended color:', r,g,b)

        osc.send("/WASEssentia/final/color/r", r / 255.0)
        osc.send("/WASEssentia/final/color/g", g / 255.0)
        osc.send("/WASEssentia/final/color/b", b / 255.0)

        # --------------------------------------------------
        # End color
        # --------------------------------------------------

        if VISUAL_DEBUG:
            visual.update(
                genre=macro,
                genre_hue=genre_hue,
                mood_hue=mood_hue,
                final_hue=final_hue,
                valence=valence,
                energy=energy,
                rgb=(r, g, b)
            )

    buffer = buffer[-HOP:]  # advance hop


if __name__ == "__main__":

    # fetch all models from folder
    models = discover_models(MODELS_DIR)
    #
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

    # read audio --> non-blocking call
    main_audio = AudioStream(
        on_audio,
        device_index=DEVICE_INDEX,
        channels=CHANNELS
    )

    audio_thread = threading.Thread(
        target=main_audio.start,
        daemon=True
    )
    audio_thread.start()

    # display visual debug overlay if enabled
    if VISUAL_DEBUG:
        try:
            while True:
                visual.render()
                time.sleep(0.16)  # 6 FPS
        except KeyboardInterrupt:
            print("Stopping…")
            main_audio.stop()
            visual.close()

    else:

        try:
            # blocking call
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping…")

    print('End WLEDAudioSyncEssentia')