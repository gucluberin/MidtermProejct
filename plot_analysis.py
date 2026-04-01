import numpy as np
import librosa
import matplotlib.pyplot as plt


def plot_autocorrelation_fft(file_path):
    signal, sr = librosa.load(file_path, sr=None)

    # Otokorelasyon
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]

    d = np.diff(autocorr)
    positive_slope = np.where(d > 0)[0]

    if len(positive_slope) == 0:
        f0_autocorr = 0.0
    else:
        start = positive_slope[0]
        peak = np.argmax(autocorr[start:]) + start
        f0_autocorr = sr / peak if peak != 0 else 0.0

    # FFT
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(fft), d=1 / sr)
    magnitude = np.abs(fft)

    freqs = freqs[:len(freqs) // 2]
    magnitude = magnitude[:len(magnitude) // 2]

    peak_idx = np.argmax(magnitude)
    f0_fft = freqs[peak_idx]

    # Grafik
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(autocorr)
    plt.title(f"Otokorelasyon\nF0 ≈ {f0_autocorr:.2f} Hz")
    plt.xlabel("Lag")
    plt.ylabel("Genlik")

    plt.subplot(1, 2, 2)
    plt.plot(freqs, magnitude)
    plt.title(f"FFT Spektrum\nF0 ≈ {f0_fft:.2f} Hz")
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Genlik")

    plt.tight_layout()
    plt.show()

    return f0_autocorr, f0_fft


# Test amaçlı çalıştırmak istersen:
if __name__ == "__main__":
    file_path = input("Analiz edilecek wav dosyası yolunu gir: ")
    f0_autocorr, f0_fft = plot_autocorrelation_fft(file_path)
    print(f"Otokorelasyon F0: {f0_autocorr:.2f} Hz")
    print(f"FFT F0: {f0_fft:.2f} Hz")