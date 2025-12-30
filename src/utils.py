import soxr
import numpy as np

def resample(audio, sr_in, sr_out):
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)

    return soxr.resample(
        audio.astype(np.float32, copy=False),
        sr_in,
        sr_out,
        quality="HQ"
    )
