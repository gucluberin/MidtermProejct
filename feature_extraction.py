import numpy as np
import librosa

def is_voiced(frame, energy_threshold=0.01, zcr_threshold=0.1):
    if len(frame) == 0:
        return False

    energy = np.sum(frame ** 2) / len(frame)
    zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2

    return energy > energy_threshold and zcr < zcr_threshold


def compute_f0(frame, sr):
    if len(frame) == 0:
        return 0

    frame = frame - np.mean(frame)

    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2:]

    d = np.diff(corr)
    positive_slope = np.where(d > 0)[0]

    if len(positive_slope) == 0:
        return 0

    start = positive_slope[0]
    peak = np.argmax(corr[start:]) + start

    if peak == 0:
        return 0

    f0 = sr / peak
    return f0