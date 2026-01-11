"""Real-time beat detection wrapper around aubio.

This module provides a small helper class that accepts arbitrary audio chunks,
feeds them into aubio's tempo detector using a rolling buffer, and exposes
beats, loudness, and tempo estimates to the rest of the system. It hides the
fixed hop-size requirements of aubio so the caller can stream audio in any
block size while still getting consistent beat events and BPM readings.
"""

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
        self._buffer_start = None
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
        self.detected_bpm = 0.0

    def process(self, audio_block: np.ndarray) -> tuple[float, float, float]:
        """
        Feed arbitrary-sized audio blocks.
        Internally processes fixed hop_size frames.
        Returns max beat strength detected in this block.
        """

        if audio_block.ndim > 1:
            audio_block = audio_block.mean(axis=1)

        audio_block = audio_block.astype(np.float32)
        db_level = aubio.db_spl(audio_block)

        # initialize ring buffer start index if not present
        if not hasattr(self, "_buffer_start"):
            self._buffer_start = 0

        # append to internal buffer without discarding existing data
        if self._buffer.size == 0:
            # if buffer is empty, just take the new block
            self._buffer = audio_block
            self._buffer_start = 0
        else:
            # append new samples to the end
            self._buffer = np.concatenate([self._buffer, audio_block])

        beat_strength = 0.0

        # process as many hops as possible using a sliding window
        # use effective length from current start index
        while (self._buffer.size - self._buffer_start) >= self.hop_size:
            end = self._buffer_start + self.hop_size
            frame = self._buffer[self._buffer_start:end]
            self._buffer_start += self.hop_size

            is_beat = self.tempo(frame)

            if is_beat:
                beat_strength = 1.0
                self.last_beat_time = self.tempo.get_last_s()
                self.detected_bpm = self.tempo.get_bpm()

        # compact buffer occasionally to avoid unbounded growth of _buffer_start
        if self._buffer_start > 0:
            remaining = self._buffer.size - self._buffer_start
            if remaining == 0:
                # no unread samples left
                self._buffer = np.zeros(0, dtype=np.float32)
                self._buffer_start = 0
            else:
                # if we've skipped a significant portion, copy remaining to front
                # threshold factor (e.g., > hop_size or > half the buffer) can be tuned
                if self._buffer_start >= self.hop_size:
                    self._buffer = self._buffer[self._buffer_start:]
                    self._buffer_start = 0

        return beat_strength, db_level, self.detected_bpm

    def get_bpm(self) -> float:
        return float(self.tempo.get_bpm())
