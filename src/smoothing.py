"""Temporal smoothing utilities for noisy genre predictions.

This module provides a lightweight exponential moving average over model
probabilities so genre outputs become more stable and readable over time.
It is used to dampen frame-to-frame noise from the classifier while still
allowing the system to follow gradual changes in the music.
"""

import numpy as np

class GenreSmoother:
    def __init__(self, labels, alpha):
        self.labels = labels
        self.alpha = alpha
        self.state = np.zeros(len(labels), dtype=np.float32)

    def update(self, probs):
        # flatten array and check size
        probs = np.ravel(probs)
        if probs.size != len(self.labels):
            # ignore invalid or incomplete predictions
            return
        self.state = self.alpha * self.state + (1 - self.alpha) * probs

    def top_n(self, n=5):
        # skip if state is empty
        if len(self.state) == 0:
            return []
        idx = np.argsort(self.state)[::-1][:n]
        return [(self.labels[i], float(self.state[i])) for i in idx]

    def reset(self):
        self.state[:] = 0.0
