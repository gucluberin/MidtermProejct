import streamlit as st
import librosa
import numpy as np
import tempfile

from feature_extraction import compute_f0, is_voiced
from classifier import classify


def predict_audio(signal, sr):
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    f0_values = []

    for i in range(0, len(signal) - frame_length, hop_length):
        frame = signal[i:i + frame_length]

        if is_voiced(frame):
            f0 = compute_f0(frame, sr)
            if 50 < f0 < 400:
                f0_values.append(f0)

    if len(f0_values) == 0:
        avg_f0 = 0.0
    else:
        avg_f0 = float(np.mean(f0_values))

    prediction = classify(avg_f0)
    return avg_f0, prediction


st.title("Ses Analizi ve Cinsiyet Sınıflandırma")

uploaded_file = st.file_uploader("Bir WAV dosyası yükleyin", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    signal, sr = librosa.load(temp_path, sr=None)

    avg_f0, prediction = predict_audio(signal, sr)

    st.subheader("Sonuç")
    st.write(f"Ortalama F0: {avg_f0:.2f} Hz")
    st.write(f"Tahmin: **{prediction}**")