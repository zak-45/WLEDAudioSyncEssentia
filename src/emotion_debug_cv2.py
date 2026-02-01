import cv2
import numpy as np
import math
from src.emotion_color import emotion_to_rgb, WHITE_RADIUS
from src.emotion_mapper_aux import apply_genre_profile_rgb

WINDOW = "WASEssentia Emotion Visual"

class EmotionDebugCV2:
    def __init__(self, size=540, use_profile=False):
        self.size = size
        self.center = size // 2
        self.radius = int(size * 0.42)
        self.use_profile = use_profile

        self.background = self._build_wheel()
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    # --------------------------------------------------

    def _build_wheel(self):
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        for y in range(self.size):
            for x in range(self.size):
                dx = (x - self.center) / self.radius
                dy = (self.center - y) / self.radius  # UP positive

                r = math.sqrt(dx * dx + dy * dy)
                if r > 1.0:
                    continue

                # Use correct emotion → RGB
                rgb = emotion_to_rgb(dx, dy, 1.0)

                # Radial falloff for nicer wheel look
                fade = min(1.0, r * 1.1)
                col = (
                    int(rgb[2] * fade),
                    int(rgb[1] * fade),
                    int(rgb[0] * fade),
                )

                img[y, x] = col

        # axes
        cv2.line(img, (0, self.center), (self.size, self.center), (80, 80, 80), 1)
        cv2.line(img, (self.center, 0), (self.center, self.size), (80, 80, 80), 1)

        # region labels
        self._label(img, "ANGER",   -0.83, +0.50)
        self._label(img, "JOY",     +0.63, +0.50)
        self._label(img, "CALM",    +0.60, -0.50)
        self._label(img, "SADNESS", -0.85, -0.50)

        self._draw_white_radius(img)

        return img

    def _label(self, img, text, vx, vy):
        x = int(self.center + vx * self.radius)
        y = int(self.center - vy * self.radius)

        cv2.putText(img, text, (x+2, y+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)
        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (240, 240, 240), 1)

    # --------------------------------------------------

    def draw(self, valence, arousal, intensity,
             emotion_label="", effect="",
             profile=None, genre=""):


        img = self.background.copy()

        # Position
        px = int(self.center + valence * self.radius)
        py = int(self.center - arousal * self.radius)

        if self.use_profile:
            base_rgb = emotion_to_rgb(valence, arousal, intensity)
            final_rgb = apply_genre_profile_rgb(base_rgb, intensity, profile)

            rgb = final_rgb

            # Base emotion point (ghost)
            cv2.circle(img, (px, py), 10, (40, 40, 40), 1)
            cv2.circle(img, (px, py), 6, base_rgb[::-1], 1)

            # Final emotion point (solid)
            cv2.circle(img, (px, py), 4, final_rgb[::-1], -1)

            # Base emotion swatch
            cv2.rectangle(img,
                          (20, self.size - 120),
                          (140, self.size - 80),
                          base_rgb[::-1], -1)
            cv2.putText(img, "Emotion",
                        (20, self.size - 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (220, 220, 220), 1)

            # Final (genre-shaped) swatch
            cv2.rectangle(img,
                          (20, self.size - 70),
                          (140, self.size - 30),
                          final_rgb[::-1], -1)
            cv2.putText(img, "Final",
                        (20, self.size - 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (220, 220, 220), 1)
        else:

            rgb = emotion_to_rgb(valence, arousal, intensity)

            # Draw point
            cv2.circle(img, (px, py), 9, (0, 0, 0), -1)
            cv2.circle(img, (px, py), 6, rgb[::-1], -1)

            # Color swatch
            cv2.rectangle(img,
                          (20, self.size - 80),
                          (140, self.size - 30),
                          rgb[::-1], -1)

        # Debug text
        def text(y, s):
            cv2.putText(img, s, (160, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (230, 230, 230), 1)

        text(30, f"Genre : {genre}")
        text(60, f"Emotion : {emotion_label}")
        text(85, f"Effect  : {effect}")
        text(110, f"Profile : {profile}")
        text(135, f"Valence : {valence:+.2f}")
        text(160, f"Arousal : {arousal:+.2f}")
        text(185, f"Intensity: {intensity:.2f}")

        if profile and self.use_profile:
            text(215, f"GenreSatGain : {profile.emotion_sat_gain:.2f}")
            text(235, f"GenreBright : {profile.emotion_bright_gain:.2f}")
            text(255, f"WhiteGain  : {profile.emotion_white_gain:.2f}")
            text(275, f"Curve      : {profile.emotion_intensity_curve}")

        text(self.size - 30, f"RGB : {rgb}")

        ring_r = int(10 + intensity * 25)
        cv2.circle(img, (px, py), ring_r, (200, 200, 200), 1)

        cv2.imshow(WINDOW, img)
        cv2.waitKey(1)

    def _draw_white_radius(self, img):
        r = int(self.radius * WHITE_RADIUS)

        # dashed circle
        for i in range(0, 360, 6):
            a1 = np.deg2rad(i)
            a2 = np.deg2rad(i + 3)

            x1 = int(self.center + r * np.cos(a1))
            y1 = int(self.center - r * np.sin(a1))
            x2 = int(self.center + r * np.cos(a2))
            y2 = int(self.center - r * np.sin(a2))

            cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # label
        cv2.putText(
            img,
            "Neutral zone",
            (self.center - 50, self.center + r + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1
        )

    # --------------------------------------------------
    @staticmethod
    def close():
        cv2.destroyWindow(WINDOW)
