<div align="center">

# WLEDAudioSyncEssentia
 ### Real-time audio analysis and genre classification.
 **A Cross-Platform (Windows / Linux / macOS) Portable Application.**

</div>

---
## Introduction

Real-time audio analysis and genre classification using the Essentia library. This tool captures audio, detects musical genres, moods, and other characteristics, and broadcasts the data via **OSC (Open Sound Control)**.

This allows for synchronization with lighting systems (like WLED via an OSC bridge), visualizers (TouchDesigner, Resolume), or other creative coding platforms.

## Overview

This project captures real-time audio input, analyzes it using Essentia's machine learning models (EffNet) to determine the musical genre, and broadcasts the results over the network. It includes smoothing logic to prevent jitter and supports auxiliary models for mood and instrumentation detection.

## Features

*   **Real-time Audio Capture**: Listens to system audio or microphone input via PyAudio.
*   **Deep Learning Analysis**: Uses Essentia's TensorFlow models for high-accuracy genre detection (Discogs 400).
*   **Macro Genre Support**: Option to collapse specific subgenres (e.g., "Deep House") into broad categories (e.g., "Electronic").
*   **Auxiliary Classifiers**: Optional analysis for:
    *   Danceability
    *   Mood (Happy, Sad, Relaxed)
    *   Instrumentation
    *   Musical Themes
*   **OSC Output**: Broadcasts classification results, color mappings, and mood data over UDP.
*   **Visual Debug Overlay**: Optional graphical window showing detected genre, mood colors, and energy.
*   **Adaptive Smoothing**: Implements buffer logic to prevent rapid flickering between predictions.

## Prerequisites

*   **Python 3.8+**
*   **Essentia**: The audio analysis library with TensorFlow support.
*   **OSC Receiver**: A target application to receive the OSC messages.

## Installation

 ### Portable (Recommended)
 1.  **Download**: Grab the latest release for your OS from the **Releases Page**.
 2.  **Extract**:
     -  Run the downloaded executable. It will extract application into a `WLEDAudioSyncEssentia` folder.

 3.  **Run**:
     -   Open a terminal and navigate to the extracted `WLEDAudioSyncEssentia` folder.
     -   Run the `WLEDAudioSyncEssentia-{OS}` executable.

 ### From Source (Manual)

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
    Create a `models/` directory in the project root. Download the following models (and their `.json` metadata) from the Essentia Models repository and place them there.

    **Required (Genre):**
    *   `discogs-effnet-bs64-1.pb` (Embedding model)
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
python WLEDAudioSyncEssentia.py or WLEDAudioSyncEssentia-{OS}
```
By default, this sends OSC messages to `127.0.0.1:12000` at path `/WASEssentia/genre/`.

### Command Line Arguments

| Argument | Description |
| :--- | :--- |
| `--macro` | Collapse specific subgenres into macro genres (e.g., Rock, Electronic). |
| `--macro-agg` | Aggregation method for macro genres (`mean` or `max`). |
| `--aux` | Enable auxiliary classifiers (Mood, Danceability, etc.). |
| `--visual` | Enable the visual debug overlay window. |
| `--debug` | Print detailed debug information to the console. |
| `--show_raw` | Print raw probability values (requires `--debug`). |
| `--osc-ip <ip>` | Set OSC destination IP (default: 127.0.0.1). |
| `--osc-port <port>` | Set OSC destination port (default: 12000). |
| `--osc-path <path>` | Set OSC base path (default: /genre). |
| `--device-index <id>` | Select specific PyAudio input device index. |
| `--channels <n>` | Set number of input channels (default: 2). |

### Examples

**Enable Macro Genres and Visual Debug:**
```bash
python WLEDAudioSyncEssentia.py --macro --visual
```

**Enable All Classifiers and Send to External IP:**
```bash
python WLEDAudioSyncEssentia.py --aux --osc-ip 192.168.1.50
```

## OSC API

The application sends the following OSC messages:

| Path | Type | Description |
| :--- | :--- | :--- |
| `/WASEssentia/genre/top{0-4}` | String | Top 5 detected genres (e.g., "Rock---Punk"). |
| `/WASEssentia/genre/macro_top{0-4}/` | String | Top 5 macro genres (if `--macro` is enabled). |
| `/WASEssentia/aux/{model}/{label}` | Float | Probability (0.0-1.0) for aux labels (e.g., `/WASEssentia/aux/mood_happy/happy`). |
| `/WASEssentia/genre/color/{r,g,b}` | Float | RGB color derived from the genre hue. |
| `/WASEssentia/mood/color` | JSON | JSON string containing valence, energy, and mood RGB. |
| `/WASEssentia/final/color/{r,g,b}` | Float | Final blended color (Genre + Mood). |

## Configuration

You can fine-tune audio buffering and smoothing in `config.py` (or `config/audio_runtime.json` if available).

```python
MODEL_SAMPLE_RATE = 16000
BUFFER_SECONDS = 2.5      # Minimum buffer for analysis
HOP_SECONDS = 0.5         # Analysis interval
SMOOTHING_ALPHA = 0.4     # Smoothing factor for predictions
```

## Credits

This project makes extensive use of the Essentia open-source library for audio analysis and music information retrieval, 
which is developed by the Music Technology Group (MTG) at Universitat Pompeu Fabra. 
Special thanks to the researchers and developers behind the pre-trained TensorFlow models used in this project. 
## Model Licenses
While the source code in this repository is licensed under MIT, the Essentia models (files ending in .pb) 
downloaded separately are subject to their own licenses, typically Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). 
If you intend to use this project for commercial purposes, please ensure you comply with the licensing terms of the specific models you download from the Essentia Models repository. 


## License

Distributed under the MIT License. See `LICENSE` for more information.
