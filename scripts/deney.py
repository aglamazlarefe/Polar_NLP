"""Tüm karşılaştırma deneylerini çalıştırır ve results/metrics.json dosyasını üretir.

Modeller : tfidf_lr · stilometri_lr · yalniz_bert · hibrit
Deneyler : A  v1 üzerinde çapraz doğrulama (konu bazlı gruplarla)
           B  v1+v2 üzerinde çapraz doğrulama
           C  v1'de eğit → v2'de test   (dağılım dışı genelleme)
           D  v2'de eğit → v1'de test

    python scripts/deney.py              # tam çalıştırma (CPU'da ~2 saat)
    python scripts/deney.py --hizli      # duman testi: küçük alt küme, 1 epoch, sonuç outputs/ altına

Her (deney, model, kat) sonucu outputs/deney_onbellek/ altında saklanır; yarıda kesilen
çalıştırma kaldığı yerden devam eder. Projedeki TÜM raporlanan sayılar bu dosyadan gelir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polar_nlp.config import (CIKTI_DIZINI, KOK, METRIK_YOLU, MODEL_ADI, SEED, SINIF_ADLARI,  # noqa: E402
                              VERI_DIZINI, VERI_YOLU, EgitimAyari)
from polar_nlp.egitim import egit, metrikler, tahmin_et  # noqa: E402
from polar_nlp.model import HybridPolarModel  # noqa: E402
from polar_nlp.stilometri import N_OZELLIK, OZELLIK_ADLARI, Olcekleyici, stilometri_matrisi, tr_kucult  # noqa: E402
from polar_nlp.veri import PolarDataset, dogrulama_ayir, katlar, veri_yukle  # noqa: E402

MODELLER = ["tfidf_lr", "stilometri_lr", "yalniz_bert", "hibrit"]
BERT_MODELLERI = {"yalniz_bert": False, "hibrit": True}   # ad -> stilo_kullan
MAKS_SAAT = 4.0


def yaz(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- tek koşu


def klasik_kosu(model: str, df, egitim_idx, test_idx) -> dict:
    """Klasik modeller doğrulama parçası kullanmaz; tüm eğitim katında eğitilir."""
    x_e, y_e, x_t = df["Metin"].iloc[egitim_idx], df["Etiket"].iloc[egitim_idx], df["Metin"].iloc[test_idx]
    if model == "tfidf_lr":
        boru = make_pipeline(TfidfVectorizer(preprocessor=tr_kucult, ngram_range=(1, 2), sublinear_tf=True),
                             LogisticRegression(C=10, max_iter=3000))
        boru.fit(x_e, y_e)
        tahmin = boru.predict(x_t)
    else:
        boru = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        boru.fit(stilometri_matrisi(x_e), y_e)
        tahmin = boru.predict(stilometri_matrisi(x_t))
    return {"tahmin": [int(t) for t in tahmin]}


def bert_kosusu(stilo_kullan: bool, df, egitim_idx, test_idx, ayar: EgitimAyari, tokenizer) -> dict:
    y = df["Etiket"].to_numpy()
    e_idx, d_idx = dogrulama_ayir(np.asarray(egitim_idx), y, seed=ayar.seed)
    metin = df["Metin"]
    olcek = Olcekleyici().uydur(stilometri_matrisi(metin.iloc[e_idx]), y[e_idx])
    egitim = PolarDataset(metin.iloc[e_idx], y[e_idx], olcek)
    dogrulama = PolarDataset(metin.iloc[d_idx], y[d_idx], olcek)
    test = PolarDataset(metin.iloc[test_idx], y[test_idx], olcek)
    ayar = EgitimAyari(**{**ayar.sozluk(), "stilo_kullan": stilo_kullan})
    model = HybridPolarModel.onceden_egitilmis(MODEL_ADI, N_OZELLIK, stilo_kullan=stilo_kullan, dropout=ayar.dropout)
    sonuc = egit(model, tokenizer, egitim, dogrulama, ayar, gunluk=yaz)
    tahmin = tahmin_et(model, tokenizer, test).argmax(1)
    return {"tahmin": [int(t) for t in tahmin], "en_iyi_epoch": sonuc["en_iyi_epoch"],
            "epoch_sayisi": len(sonuc["gecmis"]), "sure_sn": round(sonuc["sure_sn"], 1), "n_egitim": len(e_idx)}


# ---------------------------------------------------------------- deney planı


def deney_plani(df: pd.DataFrame, n_kat: int) -> dict:
    """deney adı -> {aciklama, bolmeler: [(eğitim_idx, test_idx)]}. İndeksler df konumlarıdır."""
    tum = np.arange(len(df))
    v1, v2 = tum[(df["Kaynak"] == "v1").to_numpy()], tum[(df["Kaynak"] == "v2").to_numpy()]
    plan = {}
    if len(v1):
        alt = df.iloc[v1].reset_index(drop=True)
        plan["A_v1_cv"] = {"aciklama": f"Özgün veri (v1) üzerinde {n_kat}-kat konu-gruplu çapraz doğrulama",
                           "bolmeler": [(v1[e], v1[t]) for e, t in katlar(alt, n_kat)]}
    if len(v1) and len(v2):
        plan["C_v1_v2"] = {"aciklama": "v1'de eğit, v2'de test (dağılım dışı genelleme)", "bolmeler": [(v1, v2)]}
        plan["D_v2_v1"] = {"aciklama": "v2'de eğit, v1'de test", "bolmeler": [(v2, v1)]}
        plan["B_tum_cv"] = {"aciklama": f"Birleşik veri (v1+v2) üzerinde {n_kat}-kat konu-gruplu çapraz doğrulama",
                            "bolmeler": list(katlar(df, n_kat))}
    return plan


def _alt_kume(df, test_idx, tahmin, maske) -> dict | None:
    if maske.sum() == 0:
        return None
    m = metrikler(df["Etiket"].to_numpy()[test_idx][maske], np.asarray(tahmin)[maske])
    return {"n": int(maske.sum()), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}


def model_ozeti(df, bolmeler, kosular) -> dict:
    y = df["Etiket"].to_numpy()
    kat_metrikleri = [metrikler(y[t], k["tahmin"]) for (_, t), k in zip(bolmeler, kosular)]
    test_idx = np.concatenate([t for _, t in bolmeler])
    tahmin = np.concatenate([k["tahmin"] for k in kosular])
    havuz = metrikler(y[test_idx], tahmin)
    f1 = [m["macro_f1"] for m in kat_metrikleri]
    acc = [m["accuracy"] for m in kat_metrikleri]
    zorluk, kaynak = df["Zorluk"].to_numpy()[test_idx], df["Kaynak"].to_numpy()[test_idx]
    ozet = {
        "katlar": [{"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "n_test": int(len(t))}
                   for m, (_, t) in zip(kat_metrikleri, bolmeler)],
        "macro_f1_ort": float(np.mean(f1)), "macro_f1_std": float(np.std(f1)),
        "accuracy_ort": float(np.mean(acc)), "accuracy_std": float(np.std(acc)),
        "havuz": havuz,   # tüm katların test tahminleri birleştirilerek
        "alt_kumeler": {ad: _alt_kume(df, test_idx, tahmin, maske) for ad, maske in {
            "zor": zorluk == "zor", "normal": zorluk == "normal", "v1": kaynak == "v1", "v2": kaynak == "v2"}.items()},
    }
    if "sure_sn" in kosular[0]:
        ozet["sure_sn"] = float(sum(k["sure_sn"] for k in kosular))
        ozet["en_iyi_epochlar"] = [k["en_iyi_epoch"] for k in kosular]
    return ozet


def mcnemar(y, a, b) -> dict:
    """İki modelin aynı örneklerdeki doğru/yanlışlarını karşılaştırır (kesin iki yönlü test)."""
    y, a, b = map(np.asarray, (y, a, b))
    yalniz_a = int(((a == y) & (b != y)).sum())
    yalniz_b = int(((a != y) & (b == y)).sum())
    p = 1.0 if yalniz_a + yalniz_b == 0 else float(binomtest(yalniz_a, yalniz_a + yalniz_b, 0.5).pvalue)
    return {"yalniz_ilk_dogru": yalniz_a, "yalniz_ikinci_dogru": yalniz_b, "p": p}


# ---------------------------------------------------------------- ana akış


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=KOK, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "bilinmiyor"


def veri_ozeti(df) -> dict:
    ham = stilometri_matrisi(df["Metin"])
    y = df["Etiket"].to_numpy()
    genel_ort, genel_std = ham.mean(0), np.where(ham.std(0) < 1e-6, 1.0, ham.std(0))
    ozet = {
        "n": int(len(df)),
        "sinif_dagilimi": {SINIF_ADLARI[i]: int((y == i).sum()) for i in range(len(SINIF_ADLARI))},
        "kaynak_dagilimi": {k: int(v) for k, v in df["Kaynak"].value_counts().items()},
        "zorluk_dagilimi": {k: int(v) for k, v in df["Zorluk"].value_counts().items()},
        "konu_sayisi": int(df["Konu"].nunique()),
        "stilometri": {
            "ozellikler": OZELLIK_ADLARI,
            # satır = özellik, sütun = sınıf
            "sinif_ortalamalari": [[float(ham[y == c, j].mean()) for c in range(len(SINIF_ADLARI))]
                                   for j in range(N_OZELLIK)],
            "sinif_z": [[float((ham[y == c, j].mean() - genel_ort[j]) / genel_std[j]) for c in range(len(SINIF_ADLARI))]
                        for j in range(N_OZELLIK)],
        },
    }
    denetim = VERI_DIZINI / "ham" / "denetim_ozeti.json"
    if denetim.exists():
        ozet["etiket_denetimi"] = json.loads(denetim.read_text(encoding="utf-8"))
    return ozet


def calistir(df, n_kat: int, ayar: EgitimAyari, onbellek: Path, sure_denetimi: bool) -> tuple[dict, dict] | None:
    """Tüm koşuları yapar. Süre projeksiyonu sınırı aşarsa None döner (çağıran kat sayısını düşürür)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
    plan = deney_plani(df, n_kat)
    onbellek.mkdir(parents=True, exist_ok=True)
    bert_isleri = [(len(e) * 0.85) for d in plan.values() for e, _ in d["bolmeler"]] * len(BERT_MODELLERI)
    toplam_is = sum(bert_isleri) * ayar.epoch          # örnek × epoch (üst sınır)
    hiz, kosular = None, {}
    for deney, d in plan.items():
        for model in MODELLER:
            for kat, (e_idx, t_idx) in enumerate(d["bolmeler"]):
                dosya = onbellek / f"{deney}__{model}__kat{kat}.json"
                if dosya.exists():
                    kosu = json.loads(dosya.read_text(encoding="utf-8"))
                else:
                    yaz(f"[{datetime.now():%H:%M:%S}] {deney} | {model} | kat {kat + 1}/{len(d['bolmeler'])} "
                        f"(eğitim {len(e_idx)}, test {len(t_idx)})")
                    if model in BERT_MODELLERI:
                        kosu = bert_kosusu(BERT_MODELLERI[model], df, e_idx, t_idx, ayar, tokenizer)
                    else:
                        kosu = klasik_kosu(model, df, e_idx, t_idx)
                    dosya.write_text(json.dumps(kosu), encoding="utf-8")
                kosular[(deney, model, kat)] = kosu
                if sure_denetimi and hiz is None and "sure_sn" in kosu:
                    hiz = kosu["sure_sn"] / (kosu["n_egitim"] * kosu["epoch_sayisi"])
                    saat = hiz * toplam_is / 3600
                    yaz(f"  süre projeksiyonu (üst sınır): {saat:.1f} saat")
                    if saat > MAKS_SAAT and n_kat > 3:
                        return None
    sonuc = {}
    for deney, d in plan.items():
        modeller = {m: model_ozeti(df, d["bolmeler"], [kosular[(deney, m, k)] for k in range(len(d["bolmeler"]))])
                    for m in MODELLER}
        test_idx = np.concatenate([t for _, t in d["bolmeler"]])
        birlesik = {m: np.concatenate([kosular[(deney, m, k)]["tahmin"] for k in range(len(d["bolmeler"]))])
                    for m in MODELLER}
        y = df["Etiket"].to_numpy()[test_idx]
        sonuc[deney] = {
            "aciklama": d["aciklama"],
            "n_test": int(len(test_idx)),
            "modeller": modeller,
            "mcnemar": {"hibrit_vs_yalniz_bert": mcnemar(y, birlesik["hibrit"], birlesik["yalniz_bert"]),
                        "hibrit_vs_tfidf_lr": mcnemar(y, birlesik["hibrit"], birlesik["tfidf_lr"])},
        }
        pd.DataFrame({"satir": test_idx, "gercek": y, **{f"tahmin_{m}": birlesik[m] for m in MODELLER}}).to_csv(
            onbellek / f"{deney}__tahminler.csv", index=False)
    return sonuc, {d: {"aciklama": p["aciklama"]} for d, p in plan.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--veri", default=str(VERI_YOLU))
    ap.add_argument("--kat", type=int, default=5)
    ap.add_argument("--epoch", type=int, default=EgitimAyari().epoch)
    ap.add_argument("--hizli", action="store_true", help="duman testi; results/metrics.json'a DOKUNMAZ")
    ap.add_argument("--cikti", default=None)
    a = ap.parse_args()

    df = veri_yukle(a.veri)
    ayar = EgitimAyari(epoch=a.epoch)
    n_kat = a.kat
    if a.hizli:
        df = df.groupby(["Kaynak", "Etiket"], group_keys=False).sample(n=8, random_state=SEED).reset_index(drop=True)
        ayar, n_kat = EgitimAyari(epoch=1, sabir=None), 2
    cikti = Path(a.cikti) if a.cikti else (CIKTI_DIZINI / "metrics_hizli.json" if a.hizli else METRIK_YOLU)
    parmak_izi = hashlib.sha1(pd.util.hash_pandas_object(df[["Metin", "Etiket", "Konu", "Kaynak"]]).values.tobytes()
                              + json.dumps(ayar.sozluk(), sort_keys=True).encode()).hexdigest()[:10]

    baslangic = time.time()
    not_ = None
    while True:
        onbellek = CIKTI_DIZINI / "deney_onbellek" / f"{parmak_izi}_{n_kat}kat"
        cikan = calistir(df, n_kat, ayar, onbellek, sure_denetimi=not a.hizli)
        if cikan is not None:
            break
        not_ = f"Süre projeksiyonu {MAKS_SAAT:.0f} saati aştığı için kat sayısı {n_kat}'ten 3'e düşürüldü."
        yaz(not_)
        n_kat = 3
    deneyler, _ = cikan

    import sklearn
    import torch
    import transformers
    metrik = json.loads(cikti.read_text(encoding="utf-8")) if cikti.exists() else {}
    metrik.update({
        "meta": {
            "tarih": datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(),
            "taban_model": MODEL_ADI,
            "n_kat": n_kat,
            "ayar": ayar.sozluk(),
            "cihaz": "cuda" if torch.cuda.is_available() else "cpu",
            "sure_dk": round((time.time() - baslangic) / 60, 1),
            "surumler": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "sklearn": sklearn.__version__},
            "hizli_mod": a.hizli,
            "not": not_,
        },
        "siniflar": SINIF_ADLARI,
        "veri": veri_ozeti(df),
        "deneyler": deneyler,
    })
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps(metrik, indent=2, ensure_ascii=False), encoding="utf-8")

    yaz(f"\n{'deney':<10} {'model':<15} {'macro-F1':>16} {'accuracy':>16}")
    for deney, d in deneyler.items():
        for m, o in d["modeller"].items():
            yaz(f"{deney:<10} {m:<15} {o['macro_f1_ort']:>8.4f} ±{o['macro_f1_std']:.4f} "
                f"{o['accuracy_ort']:>8.4f} ±{o['accuracy_std']:.4f}")
    yaz(f"\n-> {cikti}")


if __name__ == "__main__":
    main()
