import pandas as pd
import glob
import os
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# 1. F0 

def compute_f0(y, sr):
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    f0_list = []

    for i in range(frames.shape[1]):
        frame = frames[:, i]
        autocorr = librosa.autocorrelate(frame)

        d_min = int(sr / 400)
        d_max = int(sr / 80)

        if len(autocorr) > d_max:
            region = autocorr[d_min:d_max]
            peak = np.argmax(region)

            if peak > 0:
                f0 = sr / (peak + d_min)
                f0_list.append(f0)

    return np.mean(f0_list) if f0_list else 0


# 2. FEATURE EXTRACTION

def extract_features(y, sr):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    energy = np.mean(librosa.feature.rms(y=y))

    energy_frame = librosa.feature.rms(y=y)[0]
    zcr_frame = librosa.feature.zero_crossing_rate(y)[0]

    e_th = np.mean(energy_frame) * 0.6
    z_th = np.mean(zcr_frame) * 1.3

    voiced_idx = np.where((energy_frame > e_th) & (zcr_frame < z_th))[0]

    if len(voiced_idx) > 5:
        y = y[voiced_idx[0]*512 : voiced_idx[-1]*512]

    f0 = compute_f0(y, sr)

    return f0, zcr, energy, y


# 3. CLASSIFIER 

def classify_gender(f0, zcr, energy):
    if f0 < 165:
        return "E"
    elif f0 > 290 or (f0 > 260 and zcr > 0.12):
        return "C"
    else:
        return "K"


# 4. FFT ve AUTOCORR 

def compare_methods(y, sr):

    # AUTOCORRELATION
    autocorr = librosa.autocorrelate(y)
    d_min = int(sr / 400)
    d_max = int(sr / 80)
    region = autocorr[d_min:d_max]
    f0_auto = sr / (np.argmax(region) + d_min)

    # FFT
    D = np.abs(librosa.stft(y))
    spectrum = np.mean(D, axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    peak_idx = np.argmax(spectrum)
    f0_fft = freqs[peak_idx]

    plt.style.use("dark_background")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Autocorr
    ax[0].plot(autocorr[:1000])
    ax[0].set_title(f"Otokorelasyon (F0 ≈ {f0_auto:.2f} Hz)")

    # FFT
    ax[1].plot(freqs[:1000], spectrum[:1000])
    ax[1].set_title(f"FFT Spektrumu (F0 ≈ {f0_fft:.2f} Hz)")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------
# EKLEME: ETİKET DÜZENLEME FONKSİYONLARI
# ---------------------------------------------------
def normalize_gender_label(label):
    text = str(label).strip().lower()

    if text.startswith("e"):
        return "E"
    elif text.startswith("k"):
        return "K"
    elif text.startswith("c") or text.startswith("ç"):
        return "C"
    else:
        return str(label).strip().upper()[:1]


def label_to_full(label):
    if label == "E":
        return "Erkek"
    elif label == "K":
        return "Kadın"
    elif label == "C":
        return "Çocuk"
    else:
        return str(label)


# 5. ACCURACY

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

excel_files = glob.glob("**/Dataset/**/*.xlsx", recursive=True)

all_data = []
for f in excel_files:
    try:
        df = pd.read_excel(f)
        all_data.append(df)
    except:
        continue

if len(all_data) > 0:
    master_df = pd.concat(all_data, ignore_index=True)
    master_df.columns = master_df.columns.str.strip()
    tahminler = []
    gercekler = []
    
    # ---------------------------------------------------
    # EKLEME: F0 ve detay listeleri
    # ---------------------------------------------------
    f0_listesi = []
    detay_kayitlari = []
    
    print("\nSes dosyaları analiz ediliyor. Bekleyiniz..")

    for _, row in master_df.iterrows():
        fname = str(row['Dosya_Adi']).strip()
        path = glob.glob(f"**/Dataset/**/{fname}", recursive=True)

        if path:
            try:
                y, sr = librosa.load(path[0])

                f0, zcr, energy, _ = extract_features(y, sr)
                pred = classify_gender(f0, zcr, energy)

                gercek = normalize_gender_label(row['Cinsiyet'])

                tahminler.append(pred)
                gercekler.append(gercek)
                f0_listesi.append(round(float(f0), 2))

                detay_kayitlari.append({
                    "Dosya_Adi": fname,
                    "Gercek_Cinsiyet": label_to_full(gercek),
                    "Tahmin_Cinsiyet": label_to_full(pred),
                    "F0_Hz": round(float(f0), 2)
                })

            except:
                tahminler.append("H")
                gercekler.append("H")
                f0_listesi.append(np.nan)

                detay_kayitlari.append({
                    "Dosya_Adi": fname,
                    "Gercek_Cinsiyet": "Hata",
                    "Tahmin_Cinsiyet": "Hata",
                    "F0_Hz": np.nan
                })
        else:
            tahminler.append("B")
            gercekler.append("B")
            f0_listesi.append(np.nan)

            detay_kayitlari.append({
                "Dosya_Adi": fname,
                "Gercek_Cinsiyet": "Bulunamadı",
                "Tahmin_Cinsiyet": "Bulunamadı",
                "F0_Hz": np.nan
            })

    master_df["Tahmin"] = tahminler
    
    # ---------------------------------------------------
    # EKLEME: Gerçek ve F0 sütunları
    # ---------------------------------------------------
    master_df["Gercek_Normalize"] = gercekler
    master_df["F0_Hz"] = f0_listesi

    valid = master_df[~master_df["Tahmin"].isin(["H", "B"])]

    if len(valid) > 0:
        dogru = (valid["Gercek_Normalize"] == valid["Tahmin"]).sum()
        acc = dogru / len(valid) * 100

        print("\n------------------------------------")
        print(f"ACCURACY= %{acc:.2f}")
        print(f"Toplam Veri= {len(valid)}")
        print("------------------------------------\n")

        # ---------------------------------------------------
        # EKLEME: DETAY EXCEL TABLOSU
        # ---------------------------------------------------
        detay_df = pd.DataFrame(detay_kayitlari)

        # Sadece geçerli sınıfları al
        valid_stats = valid[valid["Gercek_Normalize"].isin(["E", "K", "C"])].copy()

        # ---------------------------------------------------
        # EKLEME: SINIF ÖZETLERİ
        # ---------------------------------------------------
        ozet_listesi = []

        for sinif_kodu, sinif_adi in [("K", "Kadın"), ("E", "Erkek"), ("C", "Çocuk")]:
            sinif_df = valid_stats[valid_stats["Gercek_Normalize"] == sinif_kodu]

            gercek_sayi = len(sinif_df)

            if gercek_sayi > 0:
                ort_f0 = sinif_df["F0_Hz"].mean()
                std_f0 = sinif_df["F0_Hz"].std()
                dogru_sayi = (sinif_df["Tahmin"] == sinif_kodu).sum()
                sinif_basarisi = (dogru_sayi / gercek_sayi) * 100
            else:
                ort_f0 = np.nan
                std_f0 = np.nan
                dogru_sayi = 0
                sinif_basarisi = np.nan

            ozet_listesi.append({
                "Sinif": sinif_adi,
                "Gercek_Sayi": gercek_sayi,
                "Ortalama_F0_Hz": round(ort_f0, 2) if pd.notna(ort_f0) else np.nan,
                "Standart_Sapma_Hz": round(std_f0, 2) if pd.notna(std_f0) else np.nan,
                "Dogru_Tahmin_Sayisi": dogru_sayi,
                "Sinif_Basarisi_%": round(sinif_basarisi, 2) if pd.notna(sinif_basarisi) else np.nan
            })

        ozet_df = pd.DataFrame(ozet_listesi)

        # ---------------------------------------------------
        # EKLEME: GENEL BAŞARI TABLOSU
        # ---------------------------------------------------
        genel_df = pd.DataFrame([{
            "Toplam_Gecerli_Ornek": len(valid),
            "Genel_Basari_%": round(acc, 2)
        }])

        # ---------------------------------------------------
        # EKLEME: EXCEL DOSYASINA YAZ
        # ---------------------------------------------------
        cikti_dosyasi = "cinsiyet_analiz_raporu.xlsx"

        with pd.ExcelWriter(cikti_dosyasi, engine="openpyxl") as writer:
            detay_df.to_excel(writer, sheet_name="Detaylar", index=False)
            ozet_df.to_excel(writer, sheet_name="Ozet", index=False)
            genel_df.to_excel(writer, sheet_name="Genel_Basari", index=False)

        print(f"Excel oluşturuldu: {cikti_dosyasi}")

else:
    print("Excel dosyası bulunamadı!")


# 6. UI

def tahmin_et():
    file_path = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])

    if file_path:
        y, sr = librosa.load(file_path)

        f0, zcr, energy, y_voiced = extract_features(y, sr)
        pred = classify_gender(f0, zcr, energy)

        label_result.config(text=f"Tahmin: {pred}")
        label_f0.config(text=f"F0: {f0:.2f} Hz")

        # waveform + autocorr + fft
        plt.style.use("dark_background")

        fig, ax = plt.subplots(3, 1, figsize=(8, 7))

        ax[0].plot(y[:2000])
        ax[0].set_title("Waveform")

        autocorr = librosa.autocorrelate(y)
        ax[1].plot(autocorr[:1000])
        ax[1].set_title("Autocorrelation")

        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        ax[2].plot(freqs[:1000], D.mean(axis=1)[:1000])
        ax[2].set_title("FFT Spectrum")

        plt.tight_layout()
        plt.show()

        compare_methods(y, sr)


# UI
root = tk.Tk()
root.title(" 🎀 Voice Classifier")
root.geometry("800x600")
root.configure(bg="#ffe6f0")

tk.Label(root, text=" 🎀Voice Classifier",
         font=("Times New Roman", 30, "bold"),
         fg="#cc0066", 
         bg="#ffe6f0").pack(pady=30)

btn = tk.Button(root, text="Ses Dosyası Seç",
                command=tahmin_et,
                font=("Times New Roman", 18, "bold"),
                bg="#ff66b2", fg="white",
                width=30, height=4)
btn.pack(pady=20)

label_result = tk.Label(root, text="Tahmin:   ",
                        font=("Times New Roman", 20),
                        fg="#99004d", bg="#ffe6f0")
label_result.pack(pady=20)

label_f0 = tk.Label(root, text="F0:    Hz",
                    font=("Times New Roman", 12),
                    fg="#99004d", bg="#ffe6f0")
label_f0.pack()

root.mainloop()