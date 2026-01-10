from collections import deque

import numpy as np

class RealTimeProcess:
    def __init__(self, cfg, debug=False, show_activity=False):
        self.motion_history = deque(maxlen=120)
        self._activity_energy = 0.0
        self.cfg = cfg
        self.fast_buffer = np.zeros(0, dtype=np.float32)
        self.debug = debug
        self.show_activity = show_activity

    def run(self,audio):
        # --------------------------------------------------
        # FAST BUFFER (activity_energy)
        # --------------------------------------------------

        self.fast_buffer = np.concatenate([self.fast_buffer, audio])

        fast_needed = int(self.cfg.MODEL_SAMPLE_RATE * self.cfg.MIN_BUFFER_SECONDS)

        if len(self.fast_buffer) < fast_needed:
            return 0

        fast_segment = self.fast_buffer[-fast_needed:]
        activity_energy = float(self.compute_activity_energy(fast_segment))

        # keep fast buffer tight
        self.fast_buffer = self.fast_buffer[-fast_needed:]

        return activity_energy

    def compute_activity_energy(self, segment: np.ndarray) -> float:
        """
        Activity energy from FAST buffer.
        Depends on BOTH loudness and envelope motion.
        Returns [0.0, 1.0]
        """

        sr = self.cfg.MODEL_SAMPLE_RATE

        # --------------------------------------------------
        # 1. Short-term RMS envelope
        # --------------------------------------------------
        frame = int(0.02 * sr)  # 20 ms
        hop = int(0.01 * sr)  # 10 ms

        if len(segment) < frame * 3:
            return 0.0

        frames = np.lib.stride_tricks.sliding_window_view(
            segment, frame
        )[::hop]

        env = np.sqrt(np.mean(frames ** 2, axis=1))

        rms = float(np.mean(env))
        if rms < self.cfg.MIN_RMS:
            self.motion_history.clear()
            self._activity_energy = 0.0
            return 0.0

        # --------------------------------------------------
        # 2. Envelope motion (FAST)
        # --------------------------------------------------
        motion = np.abs(np.diff(env))
        env_motion = float(np.mean(motion))

        # --------------------------------------------------
        # 3. Normalize loudness (FIXED SCALE)
        # --------------------------------------------------
        RMS_FLOOR = self.cfg.MIN_RMS  # e.g. 0.002
        RMS_REF = self.cfg.REF_RMS  # e.g. 0.05

        rms_norm = (rms - RMS_FLOOR) / (RMS_REF - RMS_FLOOR)
        rms_norm = np.clip(rms_norm, 0.0, 1.0)

        # --------------------------------------------------
        # 4. Normalize motion (FIXED SCALE)
        # --------------------------------------------------
        MOTION_REF = self.cfg.MOTION_REF
        # MOTION_FLOOR = 0.0010
        MOTION_FLOOR = self.compute_adaptive_motion_floor(env_motion)

        motion_norm = max(0.0, env_motion - MOTION_FLOOR)
        motion_norm = motion_norm / MOTION_REF
        motion_norm = np.clip(motion_norm, 0.0, 1.0)

        # --------------------------------------------------
        # 5. Activity = loud AND moving
        # --------------------------------------------------
        raw_activity = rms_norm * motion_norm

        # --------------------------------------------------
        # 6. Temporal smoothing (no accumulation!)
        # --------------------------------------------------
        if not hasattr(self, "_activity_energy"):
            self._activity_energy = raw_activity
        else:
            alpha = self.cfg.SMOOTHING_ALPHA
            self._activity_energy = (
                    alpha * raw_activity +
                    (1.0 - alpha) * self._activity_energy
            )

        # --------------------------------------------------
        # 7. Debug
        # --------------------------------------------------
        if self.debug and self.show_activity:
            print(
                f"activity | rms={rms:.4f} "
                f"rms_n={rms_norm:.3f} "
                f"motion={env_motion:.6f} "
                f"motion_n={motion_norm:.3f} "
                f"activity={self._activity_energy:.3f}"
            )

        return float(self._activity_energy)

    def compute_adaptive_motion_floor(self, env_motion):
        self.motion_history.append(env_motion)

        if len(self.motion_history) < 20:
            return 0.08 * self.cfg.MOTION_REF  # bootstrap tied to ref

        hist = np.array(self.motion_history)

        noise_motion = np.percentile(hist, 20)

        adaptive_floor = noise_motion * 1.8

        # anchor to MOTION_REF (your idea)
        ref_floor = 0.08 * self.cfg.MOTION_REF

        # combine: adaptive, but sane
        floor = np.clip(
            adaptive_floor,
            0.5 * ref_floor,
            1.5 * ref_floor
        )

        return floor
