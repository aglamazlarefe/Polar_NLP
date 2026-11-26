"""Çıkarım: eğitilmiş modeli bir kez yükler, tek ya da toplu metin analiz eder."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .config import BELIRSIZLIK_ESIGI, MODEL_DIZINI, RISK_AGIRLIKLARI, SINIF_ADLARI
from .egitim import cihaz_sec, tahmin_et
from .model import AGIRLIK_DOSYASI, BILGI_DOSYASI, HybridPolarModel
from .stilometri import OZELLIK_ADLARI, Olcekleyici, stilometri_matrisi
from .veri import PolarDataset

OLCEKLEYICI_DOSYASI = "olcekleyici.json"
TOKENIZER_DIZINI = "tokenizer"
GEREKLI_DOSYALAR = [AGIRLIK_DOSYASI, BILGI_DOSYASI, "config.json", OLCEKLEYICI_DOSYASI]


def model_var_mi(dizin: str | Path = MODEL_DIZINI) -> bool:
    dizin = Path(dizin)
    return all((dizin / d).exists() for d in GEREKLI_DOSYALAR) and (dizin / TOKENIZER_DIZINI).is_dir()


class PolarAnalizci:
    """Modeli bellekte tutar. Dashboard'da `st.cache_resource` ile tek örnek kullanılır."""

    def __init__(self, model_dizini: str | Path = MODEL_DIZINI, cihaz=None):
        self.dizin = Path(model_dizini)
        if not model_var_mi(self.dizin):
            raise FileNotFoundError(
                f"Eğitilmiş model bulunamadı: {self.dizin}. Önce `python scripts/egit_final.py` çalıştırın."
            )
        self.cihaz = cihaz or cihaz_sec()
        self.model, self.bilgi = HybridPolarModel.kayittan(self.dizin)
        self.model.to(self.cihaz)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.dizin / TOKENIZER_DIZINI))
        self.olcekleyici = Olcekleyici.sozlukten(
            json.loads((self.dizin / OLCEKLEYICI_DOSYASI).read_text(encoding="utf-8"))
        )
        self.sicaklik = float(self.bilgi.get("sicaklik", 1.0))

    def olasiliklar(self, metinler: list[str], batch: int = 32) -> np.ndarray:
        veri = PolarDataset(metinler, None, self.olcekleyici)
        logit = tahmin_et(self.model, self.tokenizer, veri, self.cihaz, batch)
        return torch.softmax(torch.from_numpy(logit) / self.sicaklik, dim=1).numpy()

    def analiz_et(self, metinler: list[str], batch: int = 32) -> list[dict]:
        metinler = [str(m) for m in metinler]
        if not metinler:
            return []
        olasilik = self.olasiliklar(metinler, batch)
        ham = stilometri_matrisi(metinler)
        z = self.olcekleyici.objektife_gore_z(ham)
        sonuclar = []
        for i in range(len(metinler)):
            p = olasilik[i]
            sinif = int(p.argmax())
            sonuclar.append({
                "sinif": sinif,
                "sinif_adi": SINIF_ADLARI[sinif],
                "guven_skoru": round(float(p[sinif]), 4),
                "olasiliklar": {SINIF_ADLARI[j]: round(float(p[j]), 4) for j in range(len(SINIF_ADLARI))},
                "risk_skoru": round(float(np.dot(p, RISK_AGIRLIKLARI)), 4),
                "belirsiz": bool(p[sinif] < BELIRSIZLIK_ESIGI),
                "stilometri": {
                    ad: {"deger": round(float(ham[i, j]), 4), "z": round(float(z[i, j]), 2)}
                    for j, ad in enumerate(OZELLIK_ADLARI)
                },
            })
        return sonuclar

    def kelime_katkilari(self, metin: str, maks_kelime: int = 80) -> list[tuple[str, float]]:
        """Kelime silme (occlusion): her kelime çıkarıldığında tahmin edilen sınıfın olasılığı
        ne kadar düşüyor? Pozitif değer = kelime bu kararı destekliyor."""
        kelimeler = str(metin).split()
        if len(kelimeler) < 2 or len(kelimeler) > maks_kelime:
            return [(k, 0.0) for k in kelimeler]
        eksikler = [" ".join(kelimeler[:i] + kelimeler[i + 1:]) for i in range(len(kelimeler))]
        p = self.olasiliklar([metin] + eksikler)
        sinif = int(p[0].argmax())
        return [(k, round(float(p[0, sinif] - p[i + 1, sinif]), 4)) for i, k in enumerate(kelimeler)]


@lru_cache(maxsize=2)
def _analizci(dizin: str) -> PolarAnalizci:
    return PolarAnalizci(dizin)


def polar_analiz_et(metin: str, model_dir: str | Path = MODEL_DIZINI) -> dict:
    """Tek metin için kısayol. Model ilk çağrıda yüklenir, sonrakilerde bellekten gelir."""
    return _analizci(str(Path(model_dir).resolve())).analiz_et([metin])[0]
