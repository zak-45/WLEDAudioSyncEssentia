import aubio
import numpy as np


class AubioBeatDetector:
    def __init__(
        self,
        samplerate=44100,
        hop_size=512,
        win_size=1024,
        method="default"
    ):
        if win_size < hop_size:
            raise ValueError("win_size must be >= hop_size")

        self.samplerate = samplerate
        self.hop_size = hop_size

        self.tempo = aubio.tempo(
            method,
            win_size,
            hop_size,
            samplerate
        )

        # internal rolling buffer
        self._buffer = np.zeros(0, dtype=np.float32)

        self.last_beat_time = 0.0

    def process(self, audio_block: np.ndarray) -> tuple[float, float]:
        """
        Feed arbitrary-sized audio blocks.
        Internally processes fixed hop_size frames.
        Returns max beat strength detected in this block.
        """

        if audio_block.ndim > 1:
            audio_block = audio_block.mean(axis=1)

        audio_block = audio_block.astype(np.float32)
        db_level = aubio.db_spl(audio_block)

        # append to internal buffer
        self._buffer = np.concatenate([self._buffer, audio_block])

        beat_strength = 0.0

        # process as many hops as possible
        while len(self._buffer) >= self.hop_size:
            frame = self._buffer[:self.hop_size]
            self._buffer = self._buffer[self.hop_size:]

            is_beat = self.tempo(frame)

            if is_beat:
                beat_strength = 1.0
                self.last_beat_time = self.tempo.get_last_s()

        return beat_strength, db_level

    def get_bpm(self) -> float:
        return float(self.tempo.get_bpm())
