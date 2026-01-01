
class AdaptiveBuffer:
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
        self.current = min(self.current + self.buffer_growth_rate, self.max_s)

    def shrink(self):
        self.current = max(self.current - self.buffer_shrink_rate, self.min_s)

    def reset(self):
        self.last_label = None
        self.stable_count = 0
