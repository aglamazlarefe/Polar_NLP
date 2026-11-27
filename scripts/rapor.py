"""results/metrics.json'dan docs/SONUCLAR.md üretir (elle sayı kopyalamamak için).

    python scripts/rapor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polar_nlp.config import DEMO_TREND_YOLU, KOK, METRIK_YOLU, SINIF_ADLARI  # noqa: E402
from polar_nlp.stilometri import OZELLIK_ETIKETLERI  # noqa: E402

CIKTI = KOK / "docs" / "SONUCLAR.md"
MODELLER = {"tfidf_lr": "TF-IDF + LR", "stilometri_lr": "Yalnız stilometri", "yalniz_bert": "Yalnız BERTurk",
            "hibrit": "Hibrit"}


def _pm(ort: float, std: float, cok_kat: bool) -> str:
    return f"{ort:.3f} ± {std:.3f}" if cok_kat else f"{ort:.3f}"


def _alt(o: dict, ad: str) -> str:
    a = (o.get("alt_kumeler") or {}).get(ad)
    return "—" if not a else f"{a['accuracy']:.3f} (n={a['n']})"


def uret(metrik: dict) -> str:
    meta, veri = metrik["meta"], metrik["veri"]
    s = ["# Sonuçlar", "",
         "> Bu dosya `python scripts/rapor.py` ile `results/metrics.json`'dan **otomatik üretilir. Elle düzenlemeyin.**", "",
         f"- Tarih: {meta['tarih'][:10]} · commit `{meta['commit']}` · cihaz: {meta['cihaz']} · süre: {meta['sure_dk']} dk",
         f"- Taban model: `{meta['taban_model']}` · {meta['n_kat']}-kat konu-gruplu çapraz doğrulama · tohum {meta['ayar']['seed']}",
         f"- Veri: {veri['n']} metin — " + ", ".join(f"{k}: {v}" for k, v in veri["sinif_dagilimi"].items())
         + " · kaynak " + ", ".join(f"{k}: {v}" for k, v in veri["kaynak_dagilimi"].items())
         + " · zorluk " + ", ".join(f"{k}: {v}" for k, v in veri["zorluk_dagilimi"].items())]
    if meta.get("bert_yontemi_aciklama"):
        s.append(f"- BERT modelleri: {meta['bert_yontemi_aciklama']}")
    if meta.get("not"):
        s.append(f"- Not: {meta['not']}")

    for ad, deney in metrik["deneyler"].items():
        s += ["", f"## {ad} — {deney['aciklama']}", "", f"Test örneği: {deney['n_test']}", "",
              "| Model | Macro-F1 | Doğruluk | Zor örnekler | Normal örnekler | v1 kısmı | v2 kısmı |",
              "|---|---|---|---|---|---|---|"]
        for m, etiket in MODELLER.items():
            o = deney["modeller"][m]
            cok = len(o["katlar"]) > 1
            s.append(f"| {etiket} | {_pm(o['macro_f1_ort'], o['macro_f1_std'], cok)} | "
                     f"{_pm(o['accuracy_ort'], o['accuracy_std'], cok)} | {_alt(o, 'zor')} | {_alt(o, 'normal')} | "
                     f"{_alt(o, 'v1')} | {_alt(o, 'v2')} |")
        mc = deney.get("mcnemar", {})
        for anahtar, baslik in [("hibrit_vs_yalniz_bert", "Hibrit ↔ Yalnız BERTurk"), ("hibrit_vs_tfidf_lr", "Hibrit ↔ TF-IDF + LR")]:
            if anahtar in mc:
                t = mc[anahtar]
                s.append("")
                s.append(f"McNemar ({baslik}): yalnız hibritin doğru bildiği {t['yalniz_ilk_dogru']}, yalnız diğerinin "
                         f"doğru bildiği {t['yalniz_ikinci_dogru']} örnek; p = {t['p']:.4f}")
        h = deney["modeller"]["hibrit"]["havuz"]
        s += ["", "Hibrit model — sınıf bazında (tüm test tahminleri birleştirilerek):", "",
              "| Sınıf | Kesinlik | Duyarlılık | F1 | n |", "|---|---|---|---|---|"]
        for i, sinif in enumerate(SINIF_ADLARI):
            c = h["sinif"][str(i)]
            s.append(f"| {sinif} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['destek']} |")
        s += ["", "Karışıklık matrisi (satır = gerçek, sütun = tahmin): `" + json.dumps(h["karisiklik_matrisi"]) + "`"]

    st = veri["stilometri"]
    s += ["", "## Stilometrik özelliklerin sınıf ortalamaları (ham değer)", "",
          "| Özellik | " + " | ".join(SINIF_ADLARI) + " |", "|---|" + "---|" * len(SINIF_ADLARI)]
    for ad, satir in zip(st["ozellikler"], st["sinif_ortalamalari"]):
        s.append(f"| {OZELLIK_ETIKETLERI.get(ad, ad)} | " + " | ".join(f"{v:.4f}" for v in satir) + " |")

    d = veri.get("etiket_denetimi")
    if d:
        s += ["", "## Etiket denetimi (v2 üretimi)", "",
              f"- Üretilen metin: {d['uretilen']} · veri setine giren: {d['v2_kalan']}",
              f"- Kör denetçi ile üretici etiketi uyumu: %{d['denetci_1_uyusma_orani'] * 100:.1f} (Cohen κ = {d['cohen_kappa']:.3f})",
              f"- Uyuşmazlık: {d['uyusmazlik_sayisi']} · ikinci hakemin üretici etiketini onayladığı: "
              f"{d['ikinci_hakem_uretici_etiketini_onayladi']} · atılan: {len(d['etiket_uyusmazligindan_atilan'])}",
              f"- Kalite süzgecinden atılan: uzunluk {len(d['kalite_suzgecinden_atilan']['uzunluk'])}, "
              f"yakın kopya {len(d['kalite_suzgecinden_atilan']['yakin_kopya'])}"]
        o = d.get("yalitilmis_orneklem_denetimi")
        if o:
            s.append(f"- Yalıtılmış ikinci denetçi ({o['n']} metinlik örneklem, zor örnekler ağırlıklı): uyum "
                     f"%{o['uyusma_orani'] * 100:.1f} (κ = {o['cohen_kappa']:.3f}); zor örneklerde "
                     f"%{o['zor_uyusma_orani'] * 100:.1f}")
        s.append("- Denetçiler de dil modelidir; üretici modelle benzer yanlılıklar taşıyabilirler. İnsan denetimi yapılmadı.")

    f = metrik.get("final")
    if f:
        s += ["", "## Nihai model (panelde kullanılan)", "",
              f"- Eğitim {f['n_egitim']} / doğrulama {f['n_dogrulama']} örnek · seçilen epoch {f['en_iyi_epoch']} · "
              f"sıcaklık {f['sicaklik']:.2f} · süre {f['sure_dk']} dk",
              f"- Doğrulama macro-F1 {f['dogrulama_macro_f1']:.3f} — model seçiminde kullanıldığı için **test skoru değildir**."]
    if DEMO_TREND_YOLU.exists():
        demo = json.loads(DEMO_TREND_YOLU.read_text(encoding="utf-8"))
        if demo.get("tasarimla_uyusma") is not None:
            s.append(f"- Kurgusal demo günlükleri (42 kayıt, eğitimde yok): senaryo tasarımıyla uyuşma "
                     f"%{demo['tasarimla_uyusma'] * 100:.1f}")
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    if not METRIK_YOLU.exists():
        sys.exit(f"{METRIK_YOLU} yok. Önce `python scripts/deney.py` çalıştırın.")
    CIKTI.write_text(uret(json.loads(METRIK_YOLU.read_text(encoding="utf-8"))), encoding="utf-8")
    print(f"-> {CIKTI}")
