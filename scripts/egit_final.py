"""Dashboard'un kullandığı nihai hibrit modeli eğitir ve kaydeder.

    python scripts/egit_final.py            # -> egitilmis_hybrid_model/
    python scripts/egit_final.py --hizli    # duman testi -> outputs/model_hizli/

Verinin %90'ında eğitir; %10 doğrulama parçası en iyi epoch'u ve olasılık kalibrasyonu
için sıcaklığı seçer. Bu parçadaki skor bir TEST skoru değildir — raporlanacak performans
için scripts/deney.py çıktısına (çapraz doğrulama) bakın.

Ayrıca kurgusal demo günlüklerini puanlayıp results/demo_trend.json'a yazar.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polar_nlp.cikarim import OLCEKLEYICI_DOSYASI, TOKENIZER_DIZINI, PolarAnalizci  # noqa: E402
from polar_nlp.config import (CIKTI_DIZINI, DEMO_TREND_YOLU, DEMO_YOLU, METRIK_YOLU, MODEL_ADI,  # noqa: E402
                              MODEL_DIZINI, SEED, SINIF_ADLARI, VERI_YOLU, EgitimAyari)
from polar_nlp.egitim import egit, metrikler, sicaklik_bul, tahmin_et  # noqa: E402
from polar_nlp.model import HybridPolarModel  # noqa: E402
from polar_nlp.stilometri import N_OZELLIK, Olcekleyici, stilometri_matrisi  # noqa: E402
from polar_nlp.veri import PolarDataset, dogrulama_ayir, veri_yukle  # noqa: E402


def demo_puanla(analizci: PolarAnalizci, demo_yolu: Path) -> dict | None:
    if not demo_yolu.exists():
        return None
    demo = pd.read_csv(demo_yolu)
    sonuclar = analizci.analiz_et(demo["Metin"].tolist())
    kayitlar = [{
        "Personel": s["Personel"], "Tarih": s["Tarih"],
        "Beklenen": int(s["Beklenen"]) if "Beklenen" in demo.columns else None,
        "sinif": r["sinif"], "risk_skoru": r["risk_skoru"], "guven_skoru": r["guven_skoru"],
    } for (_, s), r in zip(demo.iterrows(), sonuclar)]
    ozet = {"aciklama": "Kurgusal 14 günlük günlükler; eğitimde kullanılmadı. Değerler nihai modelin gerçek çıktısıdır.",
            "kayitlar": kayitlar}
    if "Beklenen" in demo.columns:
        ozet["tasarimla_uyusma"] = float(np.mean([k["sinif"] == k["Beklenen"] for k in kayitlar]))
    return ozet


def main():
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--veri", default=str(VERI_YOLU))
    ap.add_argument("--epoch", type=int, default=EgitimAyari().epoch)
    ap.add_argument("--hizli", action="store_true")
    a = ap.parse_args()

    df = veri_yukle(a.veri)
    ayar = EgitimAyari(epoch=a.epoch)
    cikti = MODEL_DIZINI
    if a.hizli:
        df = df.groupby("Etiket", group_keys=False).sample(n=12, random_state=SEED).reset_index(drop=True)
        ayar, cikti = EgitimAyari(epoch=1, sabir=None), CIKTI_DIZINI / "model_hizli"

    y = df["Etiket"].to_numpy()
    e_idx, d_idx = dogrulama_ayir(np.arange(len(df)), y, oran=0.10, seed=ayar.seed)
    olcek = Olcekleyici().uydur(stilometri_matrisi(df["Metin"].iloc[e_idx]), y[e_idx])
    egitim = PolarDataset(df["Metin"].iloc[e_idx], y[e_idx], olcek)
    dogrulama = PolarDataset(df["Metin"].iloc[d_idx], y[d_idx], olcek)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
    model = HybridPolarModel.onceden_egitilmis(MODEL_ADI, N_OZELLIK, stilo_kullan=True, dropout=ayar.dropout)
    print(f"Eğitim: {len(e_idx)} | Doğrulama: {len(d_idx)}", flush=True)
    sonuc = egit(model, tokenizer, egitim, dogrulama, ayar, gunluk=lambda s: print(s, flush=True))

    dog_logit = tahmin_et(model, tokenizer, dogrulama)
    dog = metrikler(y[d_idx], dog_logit.argmax(1))
    sicaklik = sicaklik_bul(dog_logit, y[d_idx].tolist())

    bilgi = {
        "taban_model": MODEL_ADI,
        "siniflar": SINIF_ADLARI,
        "tarih": datetime.now().isoformat(timespec="seconds"),
        "veri": Path(a.veri).name,
        "n_egitim": int(len(e_idx)),
        "n_dogrulama": int(len(d_idx)),
        "en_iyi_epoch": sonuc["en_iyi_epoch"],
        "dogrulama_macro_f1": dog["macro_f1"],
        "dogrulama_accuracy": dog["accuracy"],
        "sicaklik": sicaklik,
        "ayar": ayar.sozluk(),
        "sure_dk": round(sonuc["sure_sn"] / 60, 1),
    }
    model.kaydet(cikti, sonuc["en_iyi_durum"], bilgi)
    tokenizer.save_pretrained(str(cikti / TOKENIZER_DIZINI))
    (cikti / OLCEKLEYICI_DOSYASI).write_text(json.dumps(olcek.sozluk(), indent=2), encoding="utf-8")
    print(f"Model kaydedildi -> {cikti} (doğrulama macro-F1 {dog['macro_f1']:.4f}, sıcaklık {sicaklik:.2f})")

    analizci = PolarAnalizci(cikti)       # kaydı diskten geri yükleyerek doğrular
    ornek = "Rüzgar türbini jeneratör yatakları yağlandı. Voltaj dalgalanması analizi loglara eklendi."
    print("Örnek çıkarım:", json.dumps(analizci.analiz_et([ornek])[0]["olasiliklar"], ensure_ascii=False))

    if a.hizli:
        return
    METRIK_YOLU.parent.mkdir(exist_ok=True)
    metrik = json.loads(METRIK_YOLU.read_text(encoding="utf-8")) if METRIK_YOLU.exists() else {}
    metrik["final"] = {**bilgi, "gecmis": sonuc["gecmis"], "dogrulama": dog,
                       "not": "Doğrulama parçası model seçiminde kullanıldı; test skoru değildir."}
    METRIK_YOLU.write_text(json.dumps(metrik, indent=2, ensure_ascii=False), encoding="utf-8")

    demo = demo_puanla(analizci, DEMO_YOLU)
    if demo:
        DEMO_TREND_YOLU.write_text(json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Demo günlükleri puanlandı -> {DEMO_TREND_YOLU} (tasarımla uyuşma: {demo.get('tasarimla_uyusma')})")


if __name__ == "__main__":
    main()
