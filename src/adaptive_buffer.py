
class AdaptiveBuffer:
    """Dynamically adjusts a buffer size based on label stability and confidence.
    Maintains a bounded buffer length that grows when input is stable and shrinks otherwise.

    Args:
        min_s: Minimum allowed buffer size.
        max_s: Maximum allowed buffer size.
        start_s: Initial buffer size.
        confidence_threshold: Minimum confidence required to consider a label stable.
        stability_frames: Number of consecutive stable frames required before growing the buffer.
        buffer_growth_rate: Amount to increase the buffer when growing.
        buffer_shrink_rate: Amount to decrease the buffer when shrinking.
    """
    def __init__(self, min_s,
                 max_s,
                 start_s,
                 confidence_threshold,
                 stability_frames,
                 buffer_growth_rate,
                 buffer_shrink_rate):

        self.min_s = min_s
        self.max_s = max_s
        self.current = start_s
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.buffer_growth_rate = buffer_growth_rate
        self.buffer_shrink_rate = buffer_shrink_rate

        self.last_label = None
        self.stable_count = 0

    def update(self, top_label, confidence, silent=False):
        """Update the buffer size based on the latest label and confidence.
        Grows the buffer when the label is consistently stable and shrinks it on instability or silence.

        Args:
            top_label: The current dominant label or category.
            confidence: Confidence score associated with the current label.
            silent: Whether the current frame is considered silent and should force shrinking.

        Returns:
            The updated buffer size.
        """
        if silent:
            self.shrink()
            self.reset()
            return self.current

        if top_label == self.last_label and confidence >= self.confidence_threshold:
            self.stable_count += 1
            if self.stable_count >= self.stability_frames:
                self.grow()
        else:
            self.shrink()
            self.reset()

        self.last_label = top_label
        return self.current

    def grow(self):
        """Increase the buffer size while respecting the configured maximum.
        Uses the configured growth rate to expand the buffer when conditions are stable.
        """
        self.current = min(self.current + self.buffer_growth_rate, self.max_s)

    def shrink(self):
        """Decrease the buffer size while respecting the configured minimum.
        Uses the configured shrink rate to contract the buffer when conditions are unstable or silent.
        """
        self.current = max(self.current - self.buffer_shrink_rate, self.min_s)

    def reset(self):
        """Reset the internal stability tracking state.
        Clears the last seen label and the count of consecutive stable frames.
        """
        self.last_label = None
        self.stable_count = 0
