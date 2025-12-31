import numpy as np
import argparse
import time

from configmanager import *
from config import *

from src.utils import resample, compute_color
from src.audio_stream import AudioStream
from src.smoothing import GenreSmoother
from src.osc_sender import OSCSender
from src.macro_genres import collapse_to_macro

from src.model_loader import discover_models
from src.effnet_classifier import EffnetClassifier, AuxClassifier
from src.osc_schema import OscModelSchema

from src.adaptive_buffer import AdaptiveBuffer

parser = argparse.ArgumentParser()

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

args = parser.parse_args()

is_silent = False
last_non_silent_time = time.time()

SILENCE_TIMEOUT = 0.75  # seconds

adaptive = AdaptiveBuffer(
    MIN_BUFFER_SECONDS,
    MAX_BUFFER_SECONDS,
    BUFFER_SECONDS
)

NEEDED = int(MODEL_SAMPLE_RATE * adaptive.current)

HOP = int(MODEL_SAMPLE_RATE * HOP_SECONDS)

MODELS_DIR = "models"

USE_MACRO_GENRES = args.macro
MACRO_AGG = args.macro_agg

OSC_IP = args.osc_ip
OSC_PORT = args.osc_port
OSC_PATH = args.osc_path

DEVICE_INDEX = args.device_index
CHANNELS = args.channels

PRINT_RAW = args.show_raw
AUX = args.aux

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

print(
    f"🎛 OSC → {OSC_IP}:{OSC_PORT} {OSC_PATH}"
)

# Genre Color
g_brightness = 255 # default, calculate from danceability if available Energy → brightness (value)
g_saturation = 255 # default, calculate from top genre min(1.0, top1_prob * 1.5)


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

        print('_' * 80)

        if PRINT_RAW:
            print("RAW max:", probs.max())

            raw_top5 = top_n_from_probs(probs, clf.labels, 5)

            print("RAW TOP5:",
                  " | ".join(f"{g}:{v:.4f}" for g, v in raw_top5))

        smooth.update(probs)
        top5 = smooth.top_n(5)

        top_label, top_conf = top5[0]
        genre_hue = top_label

        adaptive.update(
            top_label=top_label,
            confidence=top_conf,
            silent=False
        )

        print("GENRE TOP5:",
              " | ".join(f"{g}:{v:.5f}" for g, v in top5))

        i=0
        for label, value in top5:
            osc.send(f"/WASEssentia/genre/top{i}", label)
            if i == 0:
                g_saturation = min(1.0, value * 1.5)
            i+=1

        if USE_MACRO_GENRES:
            macro_probs = collapse_to_macro(probs, clf.labels, agg=MACRO_AGG)

            # top-5 macro genres
            top5 = sorted(
                macro_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            top_label, top_conf = top5[0]

            adaptive.update(
                top_label=top_label,
                confidence=top_conf,
                silent=False
            )

            print("MACRO TOP5:",
                  " | ".join(f"{g}:{v:.5f}" for g, v in top5))

            i=0
            for label, value in top5:
                osc.send(f"/WASEssentia/genre/macro_top{i}/", label)
                i+=1

        # AUX
        if AUX:
            print('_' * 80)
            aux_results = {}
            embeddings = clf.compute_embeddings(segment)
            for aux in aux_classifiers:
                results = aux.classify(embeddings)
                if results is not None:
                    aux_results.update(results)
                    if aux.name == 'danceability':
                        g_brightness = max(0.1, min(1.0, aux_results[0][0]))
                    # sort by value
                    aux_results_sorted = sorted(
                        aux_results.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    print('_|_')
                    print(f"AUX {aux.name} =", " | ".join(f"{k}:{v:.5f}" for k, v in aux_results_sorted))

                    for label, value in aux_results.items():
                        path = f"/WASEssentia/aux/{aux.name.replace(' ', '_')}/{label.replace(' ', '_')}"
                        osc.send(path, float(value))

                    aux_results.clear()

            # Genre color
            r, g, b = compute_color(genre_hue,g_saturation,g_brightness)
            print('_|_')
            print("Genre color:", r,g,b)
            osc.send("/WASEssentia/genre/color/r", r / 255.0)
            osc.send("/WASEssentia/genre/color/g", g / 255.0)
            osc.send("/WASEssentia/genre/color/b", b / 255.0)

    buffer = buffer[-HOP:]  # advance hop


if __name__ == "__main__":

    models = discover_models(MODELS_DIR)

    aux_classifiers = []

    for m in models:
        if m["type"] == "genre":

            print(f"🎵 Genre model loaded: {m['name']}")

        else:
            aux_classifiers.append(
                AuxClassifier(m["name"], m["pb"], m["json"], m["output_name"], agg=MACRO_AGG)
            )

            print(f"🎛 Aux model loaded: {m['name']}")

    AudioStream(
        on_audio,
        device_index=DEVICE_INDEX,
        channels=CHANNELS
    ).start()
