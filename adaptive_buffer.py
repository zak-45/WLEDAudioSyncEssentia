from config import *

class AdaptiveBuffer:
    def __init__(self, min_s, max_s, start_s):
        self.min_s = min_s
        self.max_s = max_s
        self.current = start_s

        self.last_label = None
        self.stable_count = 0

    def update(self, top_label, confidence, silent=False):
        if silent:
            self.shrink()
            self.reset()
            return self.current

        if top_label == self.last_label and confidence >= CONFIDENCE_THRESHOLD:
            self.stable_count += 1
            if self.stable_count >= STABILITY_FRAMES:
                self.grow()
        else:
            self.shrink()
            self.reset()

        self.last_label = top_label
        return self.current

    def grow(self):
        self.current = min(self.current + BUFFER_GROWTH_RATE, self.max_s)

    def shrink(self):
        self.current = max(self.current - BUFFER_SHRINK_RATE, self.min_s)

    def reset(self):
        self.last_label = None
        self.stable_count = 0
