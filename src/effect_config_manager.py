import os

from configmanager import root_path
from src.emotion_mapper_aux import EffectConfig


class EffectConfigManager:
    def __init__(self, config_dir="config"):
        self.config_dir = root_path(config_dir)
        self.cache = {}

    def load(self, name):
        if name not in self.cache:
            path = os.path.join(self.config_dir, f"{name}.json")
            self.cache[name] = EffectConfig(path)
        return self.cache[name]

    def select(self, genre=None, bpm=None, hour=None):
        """
        Priority:
        1. Genre
        2. Time of day
        3. BPM range
        4. Default
        """

        # ---- Genre ----
        if genre:
            name = genre.lower()
            if self._exists(name):
                return self.load(name)

        # ---- Time of day ----
        if hour is not None:
            if 22 <= hour or hour < 6:
                if self._exists("night"):
                    return self.load("night")
            elif 6 <= hour < 12:
                if self._exists("morning"):
                    return self.load("morning")
            elif 12 <= hour < 18:
                if self._exists("day"):
                    return self.load("day")
            else:
                if self._exists("evening"):
                    return self.load("evening")

        # ---- BPM ----
        if bpm is not None:
            if bpm >= 120 and self._exists("club"):
                return self.load("club")
            if bpm <= 80 and self._exists("ambient"):
                return self.load("ambient")

        # ---- Fallback ----
        return self.load("default")

    def _exists(self, name):
        return os.path.exists(os.path.join(self.config_dir, f"{name}.json"))
