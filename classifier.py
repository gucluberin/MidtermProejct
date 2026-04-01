def classify(f0):
    """
    Kural tabanlı cinsiyet sınıflandırması.
    """
    if f0 == 0:
        return "Unknown"
    elif f0 < 150:
        return "Erkek"
    elif 150 <= f0 < 250:
        return "Kadın"
    else:
        return "Çocuk"