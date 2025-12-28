# WLEDAudioSyncEssentia

Real-time audio analysis and genre classification using the [Essentia](https://essentia.upf.edu/) library to synchronize and control [WLED](https://kno.wled.ge/) lighting effects.

## Overview

This project captures real-time audio input, analyzes it using Essentia's machine learning models to determine the musical genre, and automatically switches WLED presets or effects to match the mood of the music.

## Features

*   **Real-time Audio Capture**: Listens to system audio or microphone input.
*   **Genre Classification**: Uses Essentia's pre-trained TensorFlow models to detect genres (e.g., Rock, Electronic, Hip Hop, Jazz) on the fly.
*   **WLED Integration**: Sends JSON/UDP commands to a WLED controller to change presets based on the detected genre.
*   **Smoothing**: Implements buffer logic to prevent rapid flickering between genres.

## Prerequisites

*   **Python 3.8+**
*   **WLED Controller**: An ESP32 or ESP8266 running WLED, connected to the same network.
*   **Essentia**: The audio analysis library.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/WLEDAudioSyncEssentia.git
    cd WLEDAudioSyncEssentia
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install numpy requests sounddevice
    # Install Essentia with TensorFlow support
    pip install essentia-tensorflow
    ```

3.  **Download Models:**
    Download the required genre classification models (e.g., `genre_discogs400-discogs-effnet-1.pb` and its metadata) from the Essentia Models repository and place them in a `models/` directory within the project.

## Configuration

Update the configuration variables in your main script or `config.py`:

```python
# WLED Configuration
WLED_IP = "192.168.1.100"  # Replace with your WLED IP

# Audio Configuration
SAMPLE_RATE = 16000
BUFFER_SIZE = 512

# Genre to WLED Preset Mapping
# Key: Genre Label, Value: WLED Preset ID
GENRE_MAPPING = {
    "Electronic": 1,
    "Rock": 2,
    "Hip Hop": 3,
    "Jazz": 4,
    "Classical": 5,
    "Pop": 6
}
```

## Usage

1.  Ensure your WLED device is powered on and connected to the network.
2.  Run the application:
    ```bash
    python main.py
    ```
3.  Play music on your system. The script will analyze the audio stream and switch the WLED presets based on the dominant genre detected.

## License

Distributed under the MIT License. See `LICENSE` for more information.
