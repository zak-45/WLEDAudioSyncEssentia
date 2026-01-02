import json

import numpy as np
import argparse
import time
import threading

from configmanager import *

from src.analysis_worker import AnalysisWorker
from src.utils import resample
from src.audio_stream import AudioStream
from src.smoothing import GenreSmoother
from src.osc_sender import OSCSender
from src.model_loader import discover_models
from src.effnet_classifier import EffnetClassifier, AuxClassifier
from src.adaptive_buffer import AdaptiveBuffer
from src.mood_color_mapper import MoodColorMapper
from src.visual_debug import VisualDebugOverlay

from src.aubio_beat_detector import AubioBeatDetector

from src.runtime_config import RuntimeConfig

cfg = RuntimeConfig("config/audio_runtime.json")

import queue
audio_queue = queue.Queue(maxsize=8)


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
danceability = 0.5 # default, if no AUX classifier

# mood color, read JSON for genre labels
mood_mapper = MoodColorMapper("models/genre_discogs400-discogs-effnet-1.json",
                              "config/mood_valence.json")


def on_audio(audio, rms_rt):
    if audio.size % 2 == 0:
        audio = audio.reshape(-1, 2).mean(axis=1)

    beat = aubio_beat_detector.process(audio)
    if beat:
        print('beat detected')

    audio = resample(audio, AUDIO_DEVICE_RATE, MODEL_SAMPLE_RATE)

    try:
        audio_queue.put_nowait((audio, rms_rt, time.time()))
    except queue.Full:
        try:
            audio_queue.get_nowait()  # drop old
            audio_queue.put_nowait((audio, rms_rt, time.time()))
        except queue.Empty:
            pass

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

    # beat detector, samplerate need to be checked
    aubio_beat_detector = AubioBeatDetector(
        samplerate=cfg.AUDIO_DEVICE_RATE,  # IMPORTANT
        hop_size=512,
        win_size=1024,
    )

    analysis = AnalysisWorker(
        audio_queue=audio_queue,
        cfg=cfg,
        clf=clf,
        smooth=smooth,
        adaptive=adaptive,
        osc=osc,
        mood_mapper=mood_mapper,
        aux_classifiers=aux_classifiers,
        visual=visual if VISUAL_DEBUG else None,
        use_macro=USE_MACRO_GENRES,
        macro_agg=MACRO_AGG,
        color1=COLOR1,
        debug=DEBUG_DATA
    )

    threading.Thread(
        target=analysis.run,
        daemon=True
    ).start()

    # display visual debug overlay if enabled
    if VISUAL_DEBUG:
        try:
            while True:
                visual.render()
                time.sleep(0.16)  # 60 FPS
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