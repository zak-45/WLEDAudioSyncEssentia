# src/ring_buffer.py
"""Ring buffer for efficient sliding window audio processing.

This module provides a circular buffer that maintains a fixed-size audio window
and allows efficient updates and extractions for overlapping analysis windows.
Designed specifically for Effnet's minimum 2.1s requirement while enabling
analysis at higher frequencies (e.g., every 1s).
"""

import numpy as np


class RingBuffer:
    """Circular buffer for audio samples with efficient sliding window extraction.

    Args:
        capacity_seconds: Total buffer capacity in seconds
        sample_rate: Audio sample rate in Hz
        min_analysis_seconds: Minimum segment size for analysis (e.g., 2.1s for Effnet)
        hop_seconds: Time between consecutive analyses (e.g., 1.0s)
    """

    def __init__(self, capacity_seconds, sample_rate, min_analysis_seconds, hop_seconds):
        self.capacity_samples = int(capacity_seconds * sample_rate)
        self.min_analysis_samples = int(min_analysis_seconds * sample_rate)
        self.hop_samples = int(hop_seconds * sample_rate)
        self.sample_rate = sample_rate

        # Ring buffer storage
        self.buffer = np.zeros(self.capacity_samples, dtype=np.float32)

        # Write position (next position to write)
        self.write_pos = 0

        # Total samples written (for knowing when we have enough data)
        self.total_written = 0

        # Last extraction position (to track when we need a new analysis)
        self.last_extract_pos = 0

    def append(self, audio_chunk):
        """Add new audio samples to the ring buffer.

        Args:
            audio_chunk: NumPy array of audio samples to append
        """
        chunk_size = len(audio_chunk)

        # Handle wrap-around
        if self.write_pos + chunk_size <= self.capacity_samples:
            # Simple case: no wrap-around
            self.buffer[self.write_pos:self.write_pos + chunk_size] = audio_chunk
        else:
            # Wrap-around case
            first_part_size = self.capacity_samples - self.write_pos
            self.buffer[self.write_pos:] = audio_chunk[:first_part_size]
            self.buffer[:chunk_size - first_part_size] = audio_chunk[first_part_size:]

        # Update write position (circular)
        self.write_pos = (self.write_pos + chunk_size) % self.capacity_samples
        self.total_written += chunk_size

    def has_minimum_data(self):
        """Check if buffer contains enough samples for minimum analysis.

        Returns:
            bool: True if buffer has at least min_analysis_samples
        """
        return self.total_written >= self.min_analysis_samples

    def should_analyze(self):
        """Check if enough new data has arrived since last extraction.

        Returns:
            bool: True if hop_samples have been written since last extraction
        """
        if not self.has_minimum_data():
            return False

        # Calculate samples written since last extraction
        samples_since_extract = self.total_written - self.last_extract_pos

        return samples_since_extract >= self.hop_samples

    def get_analysis_segment(self, segment_seconds=None):
        """Extract the most recent segment for analysis.

        Args:
            segment_seconds: Length of segment to extract (defaults to min_analysis_seconds)

        Returns:
            NumPy array containing the requested segment, or None if insufficient data
        """
        if not self.has_minimum_data():
            return None

        # Use minimum analysis size by default
        if segment_seconds is None:
            segment_size = self.min_analysis_samples
        else:
            segment_size = int(segment_seconds * self.sample_rate)
            segment_size = min(segment_size, self.capacity_samples)

        # Calculate start position (most recent segment_size samples)
        available_samples = min(self.total_written, self.capacity_samples)

        if segment_size > available_samples:
            segment_size = available_samples

        # Start position in circular buffer
        start_pos = (self.write_pos - segment_size) % self.capacity_samples

        # Extract segment
        if start_pos + segment_size <= self.capacity_samples:
            # Simple case: no wrap-around
            segment = self.buffer[start_pos:start_pos + segment_size].copy()
        else:
            # Wrap-around case
            first_part = self.buffer[start_pos:]
            second_part = self.buffer[:(start_pos + segment_size) % self.capacity_samples]
            segment = np.concatenate([first_part, second_part])

        # Update last extraction position
        self.last_extract_pos = self.total_written

        return segment

    def reset(self):
        """Reset the ring buffer to initial state."""
        self.buffer[:] = 0.0
        self.write_pos = 0
        self.total_written = 0
        self.last_extract_pos = 0

    def get_fill_ratio(self):
        """Get the current fill ratio of the buffer.

        Returns:
            float: Ratio of filled samples (0.0 to 1.0)
        """
        return min(self.total_written, self.capacity_samples) / self.capacity_samples

    def get_stats(self):
        """Get buffer statistics for debugging.

        Returns:
            dict: Statistics including fill ratio, samples written, etc.
        """
        return {
            "capacity_samples": self.capacity_samples,
            "total_written": self.total_written,
            "write_pos": self.write_pos,
            "fill_ratio": self.get_fill_ratio(),
            "has_minimum": self.has_minimum_data(),
            "should_analyze": self.should_analyze(),
            "samples_since_extract": self.total_written - self.last_extract_pos
        }