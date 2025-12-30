# WLEDAudioSyncEssentia

Real-time audio analysis and genre classification using the [Essentia](https://essentia.upf.edu/) library. This tool captures audio, detects musical genres and moods, and sends the data via **OSC (Open Sound Control)**, allowing for synchronization with lighting systems like [WLED](https://kno.wled.ge/) (via an OSC bridge), visualizers like TouchDesigner, or other creative coding platforms.

## Overview

This project captures real-time audio input, analyzes it using Essentia's machine learning models (EffNet) to determine the musical genre, and broadcasts the results over the network.

## Features

*   **Real-time Audio Capture**: Listens to system audio or microphone input via PyAudio.
*   **Deep Learning Analysis**: Uses Essentia's TensorFlow models for high-accuracy genre detection.
*   **Macro Genre Support**: Option to collapse specific subgenres (e.g., "Deep House") into broad categories (e.g., "Electronic").
*   **Auxiliary Classifiers**: Optional analysis for:
    *   Danceability
    *   Mood (Happy, Sad, Relaxed)
    *   Instrumentation
    *   Musical Themes
*   **OSC Output**: Broadcasts classification results over UDP to any OSC-compatible receiver.
*   **Adaptive Smoothing**: Implements buffer logic to prevent rapid flickering between predictions.

## Prerequisites

*   **Python 3.8+**
*   **Essentia**: The audio analysis library with TensorFlow support.
*   **OSC Receiver**: A target application (e.g., TouchDesigner, Resolume, or a custom WLED bridge) to receive the `/genre` messages.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/WLEDAudioSyncEssentia.git
    cd WLEDAudioSyncEssentia
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install numpy pyaudio python-osc
    # Install Essentia with TensorFlow support
    pip install essentia-tensorflow
    ```

3.  **Download Models:**
    Create a `models/` directory in the project root. Download the following models (and their `.json` metadata) from the Essentia Models repository:

    **Required:**
    *   `discogs-effnet-bs64-1.pb`
    *   `genre_discogs400-discogs-effnet-1.pb`
    *   `genre_discogs400-discogs-effnet-1.json`

    **Optional (for --aux):**
    *   `danceability-discogs-effnet-1.pb` (+ .json)
    *   `mood_happy-discogs-effnet-1.pb` (+ .json)
    *   `mood_relaxed-discogs-effnet-1.pb` (+ .json)
    *   `mood_sad-discogs-effnet-1.pb` (+ .json)
    *   `nsynth_instrument-discogs-effnet-1.pb` (+ .json)
    *   `mtg_jamendo_top50tags-discogs-effnet-1.pb` (+ .json)
    *   `mtg_jamendo_moodtheme-discogs-effnet-1.pb` (+ .json)

## Usage

Run the main script to start listening and analyzing.

### Basic Usage
```bash
python WLEDAudioSyncEssentia.py
```
By default, this sends OSC messages to `127.0.0.1:12000` at path `/genre`.

### Advanced Usage

**Enable Macro Genres (e.g., "Rock" instead of "Punk Rock"):**
```bash
python WLEDAudioSyncEssentia.py --macro
```

**Enable Auxiliary Classifiers (Mood, Danceability, etc.):**
```bash
python WLEDAudioSyncEssentia.py --aux
```

**Custom OSC Target:**
```bash
python WLEDAudioSyncEssentia.py --osc-ip 192.168.1.50 --osc-port 8000
```

## Configuration

You can fine-tune audio buffering and smoothing in `config.py`:

```python
MODEL_SAMPLE_RATE = 16000
BUFFER_SECONDS = 2.5      # Minimum buffer for analysis
HOP_SECONDS = 0.5         # Analysis interval
SMOOTHING_ALPHA = 0.4     # Smoothing factor for predictions
```

## Usage

1.  Ensure your WLED device is powered on and connected to the network.
2.  Run the application:
    ```bash
    python WLEDAudioSyncEssentia.py
    ```
3.  Play music on your system. The script will analyze the audio stream and switch the WLED presets based on the dominant genre detected.

## License

Distributed under the MIT License. See `LICENSE` for more information.
