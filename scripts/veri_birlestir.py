"""Veri genişletme hattı: üretilen metinleri kör etiket denetiminden geçirip v1 ile birleştirir.

Adımlar (ayrıntı: docs/VERI_KARTI.md):

    python scripts/veri_birlestir.py hazirla     # data/ham/uretim_*.jsonl -> kör denetim girdisi
    #   -> bir denetçi (etiketleri GÖRMEDEN) data/ham/denetim_1.csv yazar  (kimlik,etiket)
    python scripts/veri_birlestir.py uyusmazlik  # uyuşmayanları ikinci hakem için ayırır
    #   -> ikinci kör hakem data/ham/denetim_2.csv yazar
    python scripts/veri_birlestir.py birlestir   # -> data/polar_v2.csv + data/ham/denetim_ozeti.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polar_nlp.config import SEED, V1_YOLU, VERI_DIZINI, VERI_YOLU  # noqa: E402
from polar_nlp.stilometri import kelimelere_ayir, tr_kucult  # noqa: E402

HAM = VERI_DIZINI / "ham"
KONULAR = {
    "jenerator_enerji", "yakit", "arac_kar_motoru", "isitma_yalitim", "su_atik", "telsiz_anten",
    "uydu_veri", "bilisim", "navigasyon", "meteoroloji", "buz_karotu", "laboratuvar",
    "soguk_depolama", "erzak_mutfak", "saglik_guvenlik", "yasam_alani", "saha_lojistik",
}
MIN_KELIME, MAKS_KELIME = 12, 60       # üretim hedefi 15-50; tokenizasyon farkı için pay
YAKIN_KOPYA_ESIGI = 0.6


def uretilenleri_yukle() -> pd.DataFrame:
    dosyalar = sorted(HAM.glob("uretim_*.jsonl"))
    if not dosyalar:
        sys.exit(f"{HAM} içinde uretim_*.jsonl yok")
    df = pd.concat([pd.read_json(d, lines=True, dtype={"id": str}) for d in dosyalar], ignore_index=True)
    df["metin"] = df["metin"].astype(str).str.strip()
    assert df["id"].is_unique, "yinelenen id"
    assert df["etiket"].isin([0, 1, 2]).all(), "geçersiz etiket"
    assert df["konu"].isin(KONULAR).all(), f"listede olmayan konu: {set(df['konu']) - KONULAR}"
    assert df["zorluk"].isin(["normal", "zor"]).all(), "geçersiz zorluk"
    return df


def _ngramlar(metin: str, n: int = 5) -> set[str]:
    s = " ".join(kelimelere_ayir(metin))
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


def kalite_suzgeci(df: pd.DataFrame, mevcut_metinler: list[str]) -> tuple[pd.DataFrame, dict]:
    """Uzunluk sınırı dışındakileri ve (v1 dahil) yakın kopyaları eler."""
    atilan = {"uzunluk": [], "yakin_kopya": []}
    gorulen = [_ngramlar(m) for m in mevcut_metinler]
    tut = []
    for _, satir in df.iterrows():
        n = len(kelimelere_ayir(satir["metin"]))
        if not MIN_KELIME <= n <= MAKS_KELIME:
            atilan["uzunluk"].append(satir["id"])
            continue
        g = _ngramlar(satir["metin"])
        if any(len(g & o) / len(g | o) > YAKIN_KOPYA_ESIGI for o in gorulen):
            atilan["yakin_kopya"].append(satir["id"])
            continue
        gorulen.append(g)
        tut.append(satir["id"])
    return df[df["id"].isin(tut)].reset_index(drop=True), atilan


def hazirla() -> None:
    df = uretilenleri_yukle()
    rng = random.Random(SEED)
    sira = list(range(len(df)))
    rng.shuffle(sira)
    # Özgün id'ler etiketi içerir (A01_0); denetçiye maskeli kimlik verilir.
    anahtar = pd.DataFrame({"kimlik": [f"k{i:03d}" for i in range(len(df))], "id": df["id"].iloc[sira].to_numpy()})
    anahtar.to_csv(HAM / "denetim_anahtar.csv", index=False)
    girdi = anahtar.merge(df[["id", "metin"]], on="id")[["kimlik", "metin"]]
    girdi.to_json(HAM / "denetim_girdi.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"{len(girdi)} metin -> {HAM / 'denetim_girdi.jsonl'} (etiketsiz, karıştırılmış)")


def _denetimi_oku(ad: str) -> pd.DataFrame:
    d = pd.read_csv(HAM / ad, dtype={"kimlik": str})
    assert d["etiket"].isin([0, 1, 2]).all(), f"{ad}: geçersiz etiket"
    return d.merge(pd.read_csv(HAM / "denetim_anahtar.csv"), on="kimlik", how="left")[["id", "kimlik", "etiket"]]


def uyusmazlik() -> None:
    df = uretilenleri_yukle().merge(_denetimi_oku("denetim_1.csv").rename(columns={"etiket": "denetci_1"}), on="id")
    assert len(df) == len(uretilenleri_yukle()), "denetim_1.csv tüm metinleri kapsamıyor"
    farkli = df[df["etiket"] != df["denetci_1"]]
    # İkinci hakem hangi kayıtların tartışmalı olduğunu bilmesin diye aynı sayıda uyuşan kayıt karıştırılır.
    uyusan = df[df["etiket"] == df["denetci_1"]].sample(n=min(len(farkli), (df["etiket"] == df["denetci_1"]).sum()),
                                                         random_state=SEED)
    karisik = pd.concat([farkli, uyusan]).sample(frac=1, random_state=SEED)
    karisik[["kimlik", "metin"]].to_json(HAM / "denetim_2_girdi.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"uyuşmazlık: {len(farkli)}/{len(df)}; ikinci hakem girdisi: {len(karisik)} metin")


def birlestir() -> None:
    uretilen = uretilenleri_yukle()
    d1 = _denetimi_oku("denetim_1.csv").rename(columns={"etiket": "denetci_1"})
    df = uretilen.merge(d1[["id", "denetci_1"]], on="id", how="left")
    assert df["denetci_1"].notna().all(), "denetim_1.csv eksik"
    df["denetci_1"] = df["denetci_1"].astype(int)

    kappa = float(cohen_kappa_score(df["etiket"], df["denetci_1"]))
    uyusma = float((df["etiket"] == df["denetci_1"]).mean())
    farkli = df[df["etiket"] != df["denetci_1"]]

    atilan_etiket, hakem_onayi = [], 0
    kontrol_uyusma = None
    if len(farkli):
        d2 = _denetimi_oku("denetim_2.csv").rename(columns={"etiket": "denetci_2"})
        df = df.merge(d2[["id", "denetci_2"]], on="id", how="left")
        eksik = df.loc[df["id"].isin(farkli["id"]) & df["denetci_2"].isna(), "id"].tolist()
        assert not eksik, f"denetim_2.csv şu uyuşmazlıkları kapsamıyor: {eksik}"
        for _, s in df[df["id"].isin(farkli["id"])].iterrows():
            if int(s["denetci_2"]) == s["etiket"]:
                hakem_onayi += 1          # iki kaynak (üretici + ikinci hakem) aynı etikette
            else:
                atilan_etiket.append(s["id"])
        kontrol = df[~df["id"].isin(farkli["id"]) & df["denetci_2"].notna()]
        if len(kontrol):
            kontrol_uyusma = float((kontrol["denetci_2"].astype(int) == kontrol["etiket"]).mean())
            # Kontrol kayıtlarında ikinci hakem de karşı çıkıyorsa (1 lehte, 1 aleyhte denetçi) kayıt tutulur
            # ama sayısı raporlanır.
    df = df[~df["id"].isin(atilan_etiket)]

    # İsteğe bağlı: proje dışında yalıtılmış ortamda çalışan ikinci denetçinin örneklem etiketleri.
    # Yalnız raporlanır; kayıt atmaz (üretici + 1. denetçi zaten uyuştuysa 2'ye 1 çoğunluk korunur).
    orneklem = None
    if (HAM / "denetim_orneklem.csv").exists():
        o = (_denetimi_oku("denetim_orneklem.csv").rename(columns={"etiket": "denetci"})
             .merge(uretilen[["id", "etiket", "zorluk"]], on="id"))
        orneklem = {
            "n": int(len(o)),
            "uyusma_orani": round(float((o["denetci"] == o["etiket"]).mean()), 4),
            "cohen_kappa": round(float(cohen_kappa_score(o["etiket"], o["denetci"])), 4),
            "zor_uyusma_orani": round(float((o.loc[o.zorluk == "zor", "denetci"] == o.loc[o.zorluk == "zor", "etiket"]).mean()), 4),
            "uyusmayan_idler": o.loc[o["denetci"] != o["etiket"], "id"].tolist(),
        }

    v1 = pd.read_csv(V1_YOLU)
    v1["Metin"] = v1["Metin"].astype(str).str.strip()
    konu = pd.read_csv(HAM / "v1_konu.csv")
    assert len(konu) == len(v1) and konu["konu"].isin(KONULAR).all(), "v1_konu.csv uyumsuz"
    v1 = v1.assign(Konu=konu["konu"].to_numpy(), Kaynak="v1", Zorluk="normal", Senaryo="")

    df, atilan_kalite = kalite_suzgeci(df, v1["Metin"].tolist())
    v2 = pd.DataFrame({"Metin": df["metin"], "Etiket": df["etiket"], "Konu": df["konu"], "Kaynak": "v2",
                       "Zorluk": df["zorluk"], "Senaryo": df["senaryo"]})
    tum = pd.concat([v1, v2], ignore_index=True)
    assert not tum["Metin"].duplicated().any()
    tum.to_csv(VERI_YOLU, index=False, encoding="utf-8")

    ozet = {
        "uretilen": int(len(uretilen)),
        "denetci_1_uyusma_orani": round(uyusma, 4),
        "cohen_kappa": round(kappa, 4),
        "uyusmazlik_sayisi": int(len(farkli)),
        "ikinci_hakem_uretici_etiketini_onayladi": int(hakem_onayi),
        "etiket_uyusmazligindan_atilan": atilan_etiket,
        "ikinci_hakem_kontrol_uyusma_orani": None if kontrol_uyusma is None else round(kontrol_uyusma, 4),
        "yalitilmis_orneklem_denetimi": orneklem,
        "kalite_suzgecinden_atilan": atilan_kalite,
        "v2_kalan": int(len(v2)),
        "v2_sinif_dagilimi": {str(k): int(v) for k, v in v2["Etiket"].value_counts().sort_index().items()},
        "v2_zor_sayisi": int((v2["Zorluk"] == "zor").sum()),
        "toplam": int(len(tum)),
    }
    (HAM / "denetim_ozeti.json").write_text(json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ozet, indent=2, ensure_ascii=False))
    print(f"-> {VERI_YOLU} ({len(tum)} satır)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adim", choices=["hazirla", "uyusmazlik", "birlestir"])
    {"hazirla": hazirla, "uyusmazlik": uyusmazlik, "birlestir": birlestir}[ap.parse_args().adim]()
