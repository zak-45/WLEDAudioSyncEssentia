import pyaudio
import numpy as np
import time

class AudioStream:
    def __init__(self, callback, device_index=None, channels=2):
        self.callback = callback
        self.device_index = device_index
        self.channels = channels

        self.pa = pyaudio.PyAudio()
        self.rate = 44100
        self.frames_per_buffer = 2048

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio = np.frombuffer(in_data, dtype=np.float32)
        self.callback(audio)
        return (None, pyaudio.paContinue)

    def start(self):
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self.audio_callback
        )

        print("🎧 Listening (PyAudio)... Ctrl+C to stop")
        self.stream.start_stream()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping audio stream...")
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()
