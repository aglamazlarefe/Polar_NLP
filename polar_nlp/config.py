"""Projenin tek ayar kaynağı. Sabitleri başka dosyada yeniden tanımlamayın."""

from dataclasses import asdict, dataclass
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent

VERI_DIZINI = KOK / "data"
V1_YOLU = VERI_DIZINI / "polar_v1.csv"            # özgün 300 örnek (Metin, Etiket)
VERI_YOLU = VERI_DIZINI / "polar_v2.csv"          # v1 + genişletme (Metin, Etiket, Konu, Kaynak, Zorluk)
DEMO_YOLU = VERI_DIZINI / "demo_gunlukler.csv"    # kurgusal 14 günlük günlükler, eğitimde kullanılmaz

MODEL_DIZINI = KOK / "egitilmis_hybrid_model"
SONUC_DIZINI = KOK / "results"
METRIK_YOLU = SONUC_DIZINI / "metrics.json"
DEMO_TREND_YOLU = SONUC_DIZINI / "demo_trend.json"
CIKTI_DIZINI = KOK / "outputs"                    # önbellek, tahmin dökümleri (git'e girmez)

MODEL_ADI = "dbmdz/bert-base-turkish-cased"
MAKS_UZUNLUK = 128
SEED = 42

SINIF_ADLARI = ["Objektif Rapor", "Bilişsel Yorgunluk", "Kış Sendromu"]
SINIF_KISA = ["Objektif", "Bil. Yorgunluk", "Kış Sendromu"]
SINIF_RENKLERI = ["#66bb6a", "#ffb300", "#ef5350"]
N_SINIF = len(SINIF_ADLARI)

# risk_skoru = Σ olasılık × ağırlık  (0 = objektif, 1 = kış sendromu)
RISK_AGIRLIKLARI = [0.0, 0.5, 1.0]
# En yüksek olasılık bunun altındaysa sonuç "belirsiz" işaretlenir.
BELIRSIZLIK_ESIGI = 0.60


@dataclass
class EgitimAyari:
    epoch: int = 6
    batch: int = 16
    lr_bert: float = 3e-5
    lr_bas: float = 1e-3          # rastgele başlatılan sınıflandırıcı ve stilometri dalı
    agirlik_curumesi: float = 0.01
    isinma_orani: float = 0.1
    dropout: float = 0.2
    grad_kirpma: float = 1.0
    sabir: int | None = 2         # erken durdurma; None = kapalı
    sinif_agirligi: bool = False  # True ise ağırlıklar eğitim verisinden hesaplanır
    stilo_kullan: bool = True
    seed: int = SEED

    def sozluk(self) -> dict:
        return asdict(self)
