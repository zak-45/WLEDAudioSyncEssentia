import sys
from multiprocessing import Process, Queue, freeze_support
import queue
import argparse
import time
import threading

import pyaudio

from configmanager import *

from src.analysis_process import run_analysis_process
from src.beat_printer import BeatPrinter
from src.utils import resample
from src.audio_stream import AudioStream
from src.osc_sender import OSCSender
from src.aubio_beat_detector import AubioBeatDetector
from src.runtime_config import RuntimeConfig

cfg = RuntimeConfig(root_path("config/audio_runtime.json"))

# import queue
audio_queue = Queue(maxsize=8)

# Audio
DEVICE_INDEX = cfg.DEVICE_INDEX
AUDIO_DEVICE_RATE = cfg.AUDIO_DEVICE_RATE
MODEL_SAMPLE_RATE = cfg.MODEL_SAMPLE_RATE
CHANNELS = cfg.CHANNELS

spinner = BeatPrinter()

def list_devices(p: pyaudio.PyAudio):
    """Lists all available audio input devices."""
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            print(f"  [{info['index']}] {info['name']}")
    print("\nUse the index with the --device-index flag to select a device.")


def on_audio(audio, rms_rt):
    # stereo to mono
    if audio.size % 2 == 0:
        audio = audio.reshape(-1, 2).mean(axis=1)

    # beat detector, dB
    beat, level = aubio_beat_detector.process(audio)

    if beat:
        spinner_char = spinner.get_char()
        sys.stdout.write(f"Beat detected {spinner_char} dB: {level:.2f} \r")
        osc.send('/WASEssentia/audio/beat', spinner_char)
        osc.send('/WASEssentia/audio/dB', level)

    # resample to model rate
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
    freeze_support()

    print('Start WLEDAudioSyncEssentia')

    if "NUITKA_ONEFILE_PARENT" in os.environ:
        """
        When this env var exist, this mean run from the one-file compressed executable.
        This env not exist when run from the extracted program.
        Expected way to work.
        """
        # Nuitka compressed version extract binaries to "WLEDAudioSyncEssentia" folder (as set in the GitHub action)
        # show message

        from src.message import msg
        msg.message()
        input('enter to continue...')
        sys.exit()

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
        help="Choose color type Genre centric for final hue"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command to list devices
    list_parser = subparsers.add_parser("list", help="List available audio input devices.")

    args, unknown = parser.parse_known_args()

    if args.command == "list":
        list_devices(pyaudio.PyAudio())

    else:

        OSC_IP = args.osc_ip
        OSC_PORT = args.osc_port
        OSC_PATH = args.osc_path

        COLOR1 = args.color1

        VISUAL_DEBUG = args.visual
        DEBUG_DATA = args.debug

        USE_MACRO_GENRES = args.macro
        MACRO_AGG = args.macro_agg

        AUX = args.aux

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

        # --- Device Selection ---
        p_temp = pyaudio.PyAudio()
        try:
            if args.device_index is not None:
                device_info = p_temp.get_device_info_by_index(args.device_index)
                print(f"Attempting to use specified device: [{device_info['index']}] {device_info['name']}")
                DEVICE_INDEX = args.device_index
            elif DEVICE_INDEX is not None:
                device_info = p_temp.get_device_info_by_index(DEVICE_INDEX)
                print(f"Attempting to use specified device from config: [{device_info['index']}] {device_info['name']}")
            else:
                device_info = p_temp.get_default_input_device_info()
                args.device = device_info['index']
                print(f"No device specified, using default input: [{device_info['index']}] {device_info['name']}")
        except (IOError, IndexError):
            print(f"Error: Device index {DEVICE_INDEX} is invalid. Use 'list' command to see available devices.")
            sys.exit(1)
        finally:
            p_temp.terminate()

        if AUDIO_DEVICE_RATE is None:
            AUDIO_DEVICE_RATE = int(device_info['defaultSampleRate'])

        print(f"Using device sample rate: {AUDIO_DEVICE_RATE} Hz")

        if args.channels is not None:
            CHANNELS = args.channels

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

        # beat detector, samplerate provide from config or from audio (need to be accurate to avoid false value)
        aubio_beat_detector = AubioBeatDetector(
            samplerate=AUDIO_DEVICE_RATE,  # IMPORTANT
            hop_size=512,
            win_size=1024,
        )

        analysis_proc = Process(
            target=run_analysis_process,
            args=(
                audio_queue,
                root_path("config/audio_runtime.json"),
                OSC_IP,
                OSC_PORT,
                OSC_PATH,
                USE_MACRO_GENRES,
                MACRO_AGG,
                COLOR1,
                DEBUG_DATA,
                VISUAL_DEBUG,
                AUX,
            ),
            daemon=True
        )

        analysis_proc.start()

        try:
            # blocking call
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            analysis_proc.terminate()
            main_audio.stop()
            print("Stopping…")

    print('End WLEDAudioSyncEssentia')