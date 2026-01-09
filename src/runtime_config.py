import json

class RuntimeConfig:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

    # --- Audio ---
    @property
    def AUDIO_DEVICE_RATE(self):
        return self.cfg["audio"]["device_rate"]

    @property
    def DEVICE_INDEX(self):
        return self.cfg["audio"]["device_index"]

    @property
    def CHANNELS(self):
        return self.cfg["audio"]["channels_default"]

    @property
    def MODEL_SAMPLE_RATE(self):
        return self.cfg["audio"]["model_rate"]

    @property
    def MIN_RMS(self):
        return self.cfg["audio"]["min_rms"]

    @property
    def REF_RMS(self):
        return self.cfg["audio"]["ref_rms"]

    @property
    def SILENCE_TIMEOUT(self):
        return self.cfg["audio"]["silence_timeout"]


    # --- Buffer ---
    @property
    def BUFFER_SECONDS(self):
        return self.cfg["buffer"]["initial_seconds"]

    @property
    def MIN_BUFFER_SECONDS(self):
        return self.cfg["buffer"]["min_seconds"]

    @property
    def MAX_BUFFER_SECONDS(self):
        return self.cfg["buffer"]["max_seconds"]

    @property
    def HOP_SECONDS(self):
        return self.cfg["buffer"]["hop_seconds"]

    @property
    def BUFFER_GROWTH_RATE(self):
        return self.cfg["buffer"]["growth_rate"]

    @property
    def BUFFER_SHRINK_RATE(self):
        return self.cfg["buffer"]["shrink_rate"]

    @property
    def CONFIDENCE_THRESHOLD(self):
        return self.cfg["buffer"]["confidence_threshold"]

    @property
    def STABILITY_FRAMES(self):
        return self.cfg["buffer"]["stability_frames"]

    @property
    def ACTIVATE_BUFFER(self):
        return self.cfg["buffer"]["activate"]


    # --- Smoothing ---
    @property
    def SMOOTHING_ALPHA(self):
        return self.cfg["smoothing"]["alpha"]


    # --- Activity Energy ---
    @property
    def MOTION_REF(self):
        return self.cfg["activity_energy"]["motion_ref"]
