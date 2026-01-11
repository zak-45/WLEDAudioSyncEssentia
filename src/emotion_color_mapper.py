"""Color mapping for continuous emotion coordinates in a valence–arousal space.

This module turns high-level emotional descriptors (valence and arousal in
[-1, 1]) into smooth, visually pleasing RGB colors using a precomputed 2D
emotion surface. It is designed to provide stable, gradual color changes that
track the emotional character of music without abrupt jumps.
"""

from  src.color_map_2d import *


class EmotionColorMapper:
    """
    Maps continuous (valence, arousal) values in [-1, 1]
    to an RGB color using a 2D emotion color surface
    (ported from WLEDAudioSyncRTMood).
    """

    def __init__(
        self,
        mood_image_path: str,
        smoothing: float = 0.0
    ):
        """
        :param mood_image_path: Path to music_color_mood.png
        :param smoothing: EMA smoothing factor [0..1]
                          0 = no smoothing
        """
        self.smoothing = np.clip(smoothing, 0.0, 1.0)
        self._prev_color = None

        # Load background image (only for dimensions)
        img = cv2.cvtColor(
            cv2.imread(mood_image_path),
            cv2.COLOR_BGR2RGB
        )

        if img is None:
            raise FileNotFoundError(f"Cannot load {mood_image_path}")

        self.height, self.width, _ = img.shape

        # Emotion anchor colors
        colors = {
            "orange": [255, 165, 0],
            "blue": [0, 0, 255],
            "bluegreen": [0, 165, 255],
            "green": [0, 205, 0],
            "red": [255, 0, 0],
            "yellow": [255, 255, 0],
            "purple": [128, 0, 128],
            "neutral": [255, 241, 224],
        }

        # Valence / Energy anchor points (RTMood originals)
        points = [
            [-0.9,  0.0],   # disgust
            [-0.5,  0.5],   # angry
            [ 0.0,  0.6],   # alert
            [ 0.5,  0.5],   # happy
            [ 0.4, -0.4],   # calm
            [ 0.0, -0.6],   # relaxed
            [-0.5, -0.5],   # sad
            [ 0.0,  0.0],   # neutral
        ]

        point_colors = [
            colors["purple"],
            colors["red"],
            colors["orange"],
            colors["yellow"],
            colors["green"],
            colors["bluegreen"],
            colors["blue"],
            colors["neutral"],
        ]

        # Precompute emotion color surface
        self.emo_map = create_2d_color_map(
            points,
            point_colors,
            self.height,
            self.width
        )

    @staticmethod
    def _normalize(valence: float, arousal: float):
        """
        Clamp values to [-1, 1]
        """
        return (
            float(np.clip(valence, -1.0, 1.0)),
            float(np.clip(arousal, -1.0, 1.0))
        )

    def get_rgb(self, valence: float, arousal: float):
        """
        Convert valence/arousal to RGB color

        :param valence: [-1..1]
        :param arousal: [-1..1]
        :return: (R, G, B)
        """
        valence, arousal = self._normalize(valence, arousal)

        # Convert to pixel coordinates
        x = int(self.width  / 2 + (self.width  / 2) * valence)
        y = int(self.height / 2 - (self.height / 2) * arousal)

        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)

        # Sample local region (noise-robust)
        color = np.median(
            self.emo_map[max(0, y-2):y+2, max(0, x-2):x+2],
            axis=0
        ).mean(axis=0)

        # Convert BGR → RGB
        rgb = np.array(
            [int(color[2]), int(color[1]), int(color[0])],
            dtype=np.float32
        )

        # Optional smoothing (EMA)
        if self.smoothing > 0.0:
            if self._prev_color is None:
                self._prev_color = rgb
            else:
                rgb = (
                    self.smoothing * self._prev_color +
                    (1.0 - self.smoothing) * rgb
                )
                self._prev_color = rgb

        return tuple(rgb.astype(int))
