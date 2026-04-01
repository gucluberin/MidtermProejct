import os
import pandas as pd
import librosa
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from feature_extraction import compute_f0, is_voiced
from classifier import classify

METADATA_PATH = "Dataset/master_metadata.xlsx"
BASE_PATH = "Dataset"

df = pd.read_excel(METADATA_PATH)

print("Excel sütunları:", df.columns.tolist())

results = []

print("Analiz başlıyor...\n")

for index, row in df.iterrows():

    file_name = row["Dosya_Adi"]

    # ❗ boş satır kontrolü
    if pd.isna(file_name):
        continue

    file_name = str(file_name)

    true_label = row["Cinsiyet"]

    # 🔥 TÜM KLASÖRLERDE ARA
    found_path = None

    for root, dirs, files in os.walk(BASE_PATH):
        if file_name in files:
            found_path = os.path.join(root, file_name)
            break

    if found_path is None:
        print("Dosya bulunamadı:", file_name)
        continue

    signal, sr = librosa.load(found_path, sr=None)

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    f0_values = []

    for i in range(0, len(signal) - frame_length, hop_length):
        frame = signal[i:i + frame_length]

        if is_voiced(frame):
            f0 = compute_f0(frame, sr)

            if 50 < f0 < 400:
                f0_values.append(f0)

    avg_f0 = np.mean(f0_values) if len(f0_values) > 0 else 0.0

    predicted = classify(avg_f0)

    results.append({
        "Dosya": found_path,
        "Gerçek": true_label,
        "Tahmin": predicted,
        "F0": avg_f0
    })

    print(f"{found_path} -> {predicted} (F0: {avg_f0:.2f})")

# -----------------------------
# SONUÇ
# -----------------------------
result_df = pd.DataFrame(results)

if len(result_df) == 0:
    print("Hiç sonuç üretilemedi.")
else:
    accuracy = (result_df["Gerçek"] == result_df["Tahmin"]).mean()

    print("\nAccuracy:", accuracy)

    labels = sorted(result_df["Gerçek"].unique())

    cm = confusion_matrix(result_df["Gerçek"], result_df["Tahmin"], labels=labels)
    print("\nConfusion Matrix:\n", cm)

    plt.figure(figsize=(6,5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()

    result_df.to_excel("sonuclar.xlsx", index=False)
    print("\nSonuçlar kaydedildi: sonuclar.xlsx")