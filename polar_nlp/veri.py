"""Veri yükleme, doğrulama, bölme ve PyTorch veri kümesi."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from .config import MAKS_UZUNLUK, N_SINIF, SEED, VERI_YOLU
from .stilometri import Olcekleyici, stilometri_matrisi

ZORUNLU_SUTUNLAR = ["Metin", "Etiket"]
VARSAYILANLAR = {"Konu": "bilinmiyor", "Kaynak": "v1", "Zorluk": "normal"}


def veri_yukle(yol: str | Path = VERI_YOLU) -> pd.DataFrame:
    """CSV'yi okur ve doğrular. Sorunlu veride açık bir ValueError verir."""
    df = pd.read_csv(yol)
    eksik = [s for s in ZORUNLU_SUTUNLAR if s not in df.columns]
    if eksik:
        raise ValueError(f"{yol}: eksik sütun(lar): {eksik}")
    for sutun, varsayilan in VARSAYILANLAR.items():
        if sutun not in df.columns:
            df[sutun] = varsayilan
        df[sutun] = df[sutun].fillna(varsayilan).astype(str)
    if df["Metin"].isna().any() or (df["Metin"].astype(str).str.strip() == "").any():
        raise ValueError(f"{yol}: boş metin içeren satır var")
    df["Metin"] = df["Metin"].astype(str).str.strip()
    if not pd.api.types.is_integer_dtype(df["Etiket"]):
        raise ValueError(f"{yol}: Etiket sütunu tamsayı olmalı")
    if not df["Etiket"].between(0, N_SINIF - 1).all():
        raise ValueError(f"{yol}: Etiket 0..{N_SINIF - 1} aralığında olmalı")
    if df["Metin"].duplicated().any():
        raise ValueError(f"{yol}: yinelenen metin var ({int(df['Metin'].duplicated().sum())} adet)")
    return df.reset_index(drop=True)


def katlar(df: pd.DataFrame, n_kat: int = 5, seed: int = SEED):
    """(eğitim_idx, test_idx) çiftleri üretir.

    `Konu` bilgisi varsa aynı konu hem eğitimde hem testte bulunmaz (grup bazlı bölme);
    böylece model konuya özgü kalıpları ezberleyerek puan alamaz.
    """
    y = df["Etiket"].to_numpy()
    gruplar = df["Konu"].to_numpy()
    if len(set(gruplar)) >= n_kat:
        bolucu = StratifiedGroupKFold(n_splits=n_kat, shuffle=True, random_state=seed)
        yield from bolucu.split(np.zeros(len(df)), y, gruplar)
    else:
        bolucu = StratifiedKFold(n_splits=n_kat, shuffle=True, random_state=seed)
        yield from bolucu.split(np.zeros(len(df)), y)


def dogrulama_ayir(idx: np.ndarray, y: np.ndarray, oran: float = 0.15, seed: int = SEED):
    """Eğitim indekslerinden katmanlı bir doğrulama parçası ayırır (model seçimi için)."""
    idx = np.asarray(idx)
    n_sinif = len(np.unique(y[idx]))
    n_dog = min(max(int(round(len(idx) * oran)), n_sinif), len(idx) - n_sinif)   # her sınıftan en az bir örnek
    egitim, dogrulama = train_test_split(idx, test_size=n_dog, random_state=seed, stratify=y[idx])
    return np.sort(egitim), np.sort(dogrulama)


class PolarDataset(Dataset):
    """Metin + etiket + ölçeklenmiş stilometri. Tokenizasyon harmanlayıcıda yapılır."""

    def __init__(self, metinler, etiketler, olcekleyici: Olcekleyici):
        self.metinler = [str(m) for m in metinler]
        self.etiketler = None if etiketler is None else [int(e) for e in etiketler]
        self.stilometri = olcekleyici.donustur(stilometri_matrisi(self.metinler))

    def __len__(self):
        return len(self.metinler)

    def __getitem__(self, i):
        etiket = -1 if self.etiketler is None else self.etiketler[i]
        return self.metinler[i], self.stilometri[i], etiket


def harmanlayici(tokenizer, maks_uzunluk: int = MAKS_UZUNLUK):
    """Partiyi en uzun örneğe kadar doldurur (sabit 128 yerine) — CPU'da belirgin hız kazancı."""

    def _harmanla(parti):
        metinler, stilo, etiketler = zip(*parti)
        kod = tokenizer(list(metinler), padding=True, truncation=True, max_length=maks_uzunluk, return_tensors="pt")
        return {
            "input_ids": kod["input_ids"],
            "attention_mask": kod["attention_mask"],
            "stilometri": torch.from_numpy(np.stack(stilo)),
            "etiket": torch.tensor(etiketler, dtype=torch.long),
        }

    return _harmanla
