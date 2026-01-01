import cv2
import numpy as np
import threading

class VisualDebugOverlay:
    def __init__(self, width=420, height=260):
        self.width = width
        self.height = height
        self.window_name = "WLEDAudioSyncEssentia Color Visual"

        self.lock = threading.Lock()
        self.last_frame = None
        self.active = True

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def update(
        self,
        genre,
        genre_hue,
        mood_hue,
        final_hue,
        valence,
        energy,
        rgb
    ):
        """Called from audio thread"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30
        dy = 26

        def text(label):
            nonlocal y
            cv2.putText(img, label, (12, y), font, 0.4, (220, 220, 220), 1)
            y += dy

        text(f"GENRE       : {genre}")
        text(f"GENRE HUE   : {genre_hue:.1f} deg.")
        text(f"MOOD HUE    : {mood_hue:.1f} deg.")
        text(f"FINAL HUE   : {final_hue:.1f} deg.")
        y += 10
        text(f"VALENCE     : {valence:+.3f}")
        text(f"ENERGY      : {energy:.3f}")

        # --- Hue preview rectangles ---
        rect_y = y + 10
        rect_h = 22
        rect_w = 90
        gap = 10
        x0 = 12

        genre_col = self._hsv_to_bgr(genre_hue)
        mood_col = self._hsv_to_bgr(mood_hue)
        final_col = self._hsv_to_bgr(final_hue)

        cv2.rectangle(img, (x0, rect_y),
                      (x0 + rect_w, rect_y + rect_h), genre_col, -1)
        cv2.rectangle(img, (x0 + rect_w + gap, rect_y),
                      (x0 + 2 * rect_w + gap, rect_y + rect_h), mood_col, -1)
        cv2.rectangle(img, (x0 + 2 * (rect_w + gap), rect_y),
                      (x0 + 3 * rect_w + 2 * gap, rect_y + rect_h), final_col, -1)

        cv2.putText(img, "GENRE", (x0, rect_y - 4), font, 0.35, (200, 200, 200), 1)
        cv2.putText(img, "MOOD", (x0 + rect_w + gap, rect_y - 4),
                    font, 0.35, (200, 200, 200), 1)
        cv2.putText(img, "FINAL", (x0 + 2 * (rect_w + gap), rect_y - 4),
                    font, 0.35, (200, 200, 200), 1)

        # --- RGB preview bar ---

        bar_top = self.height - 30
        cv2.rectangle(
            img,
            (10, bar_top),
            (self.width - 10, self.height - 10),
            (int(rgb[2]), int(rgb[1]), int(rgb[0])),
            -1
        )

        with self.lock:
            self.last_frame = img

    def render(self):
        """Called from main thread"""
        with self.lock:
            if self.last_frame is None:
                return
            frame = self.last_frame.copy()

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def close(self):
        self.active = False
        cv2.destroyWindow(self.window_name)

    @staticmethod
    def _hsv_to_bgr(hue):
        """
        hue: 0–360
        returns BGR tuple for OpenCV
        """
        hsv = np.uint8([[[int(hue / 2), 255, 255]]])  # OpenCV uses 0–180
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in bgr)
