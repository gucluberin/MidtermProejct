import os
import glob
import pandas as pd
import numpy as np
import librosa

print("BAŞLADI")

def compute_f0(y, sr):
    autocorr = librosa.autocorrelate(y)
    d_min, d_max = int(sr/400), int(sr/80)
    
    if len(autocorr) > d_max:
        region = autocorr[d_min:d_max]
        if len(region) > 0:
            return sr / (np.argmax(region) + d_min)
    return 0

dataset_path = r"C:\Users\emir\Desktop\sesislemeproje3\DATASET"

excel_files = glob.glob(os.path.join(dataset_path, "**", "*.xlsx"), recursive=True)

print("Excel sayısı:", len(excel_files))

results = []

for f in excel_files:
    print("Excel:", f)
    try:
        df = pd.read_excel(f)
        base = os.path.dirname(f)

        print("Sütunlar:", df.columns)

        file_col = df.columns[0]  
        gender_col = df.columns[1] if len(df.columns) > 1 else None

        for _, row in df.iterrows():
            fname = str(row[file_col]).strip()
            if not fname.endswith(".wav"):
                fname += ".wav"

            path = os.path.join(base, fname)

            if os.path.exists(path):
                print("OK:", fname)
                y, sr = librosa.load(path, sr=None)
                f0 = compute_f0(y, sr)

                real = str(row[gender_col]).strip().upper()[0] if gender_col else "?"

                results.append({
                    "Dosya": fname,
                    "Gerçek": real,
                    "F0": round(f0, 2)
                })
            else:
                print("YOK", path)

    except Exception as e:
        print("HATA:", e)

df_out = pd.DataFrame(results)

df_out.to_csv("F0_sonuclar.csv", index=False)

print("Bitti.")