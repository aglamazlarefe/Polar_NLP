"""results/metrics.json ve results/demo_trend.json dosyalarından grafik üretir.

KURAL: Bu dosyada elle yazılmış sonuç sayısı bulunmaz. Her değer JSON'dan okunur;
JSON'lar scripts/deney.py ve scripts/egit_final.py tarafından gerçek eğitimle üretilir.

    python scripts/gorseller.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polar_nlp.config import (DEMO_TREND_YOLU, METRIK_YOLU, SINIF_KISA, SINIF_RENKLERI,  # noqa: E402
                              SONUC_DIZINI)
from polar_nlp.stilometri import OZELLIK_ETIKETLERI  # noqa: E402

ARKA, YAZI, IZGARA, KUTU = "#0a1628", "#cfe8f5", "#607d8b", "#1a2a45"
MODEL_ETIKETLERI = {"tfidf_lr": "TF-IDF + LR", "stilometri_lr": "Yalnız stilometri",
                    "yalniz_bert": "Yalnız BERTurk", "hibrit": "Hibrit"}
MODEL_RENKLERI = {"tfidf_lr": "#90a4ae", "stilometri_lr": "#ba68c8", "yalniz_bert": "#4fc3f7", "hibrit": "#00e5ff"}
DENEY_ETIKETLERI = {"A_v1_cv": "A · v1 çapraz doğ.", "B_tum_cv": "B · v1+v2 çapraz doğ.",
                    "C_v1_v2": "C · v1 → v2", "D_v2_v1": "D · v2 → v1"}


def _tema(ax):
    ax.set_facecolor(ARKA)
    ax.tick_params(colors=YAZI)
    for nesne in (ax.xaxis.label, ax.yaxis.label, ax.title):
        nesne.set_color(YAZI)
    for kenar in ax.spines.values():
        kenar.set_edgecolor(IZGARA)


def _lejant(ax, **kw):
    ax.legend(facecolor=KUTU, edgecolor=IZGARA, labelcolor=YAZI, fontsize=8.5, **kw)


def _kaydet(fig, ad: str) -> Path:
    yol = SONUC_DIZINI / ad
    fig.savefig(yol, dpi=150, facecolor=ARKA, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {yol.relative_to(SONUC_DIZINI.parent)}")
    return yol


def _ana_deney(metrik: dict) -> tuple[str, dict]:
    for ad in ("B_tum_cv", "A_v1_cv"):
        if ad in metrik["deneyler"]:
            return ad, metrik["deneyler"][ad]
    return next(iter(metrik["deneyler"].items()))


def karisiklik_matrisi(metrik):
    ad, deney = _ana_deney(metrik)
    cm = np.array(deney["modeller"]["hibrit"]["havuz"]["karisiklik_matrisi"])
    oran = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    yazi = np.array([[f"{cm[i, j]}\n%{oran[i, j] * 100:.0f}" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=ARKA)
    _tema(ax)
    sns.heatmap(oran, annot=yazi, fmt="", cmap="Blues", vmin=0, vmax=1, xticklabels=SINIF_KISA,
                yticklabels=SINIF_KISA, linewidths=0.5, linecolor=IZGARA, cbar_kws={"shrink": 0.8}, ax=ax)
    renk_cubugu = ax.collections[0].colorbar
    plt.setp(renk_cubugu.ax.yaxis.get_ticklabels(), color=YAZI)
    ax.set_xticklabels(SINIF_KISA, color=YAZI)
    ax.set_yticklabels(SINIF_KISA, color=YAZI, rotation=0)
    ax.set_xlabel("Tahmin edilen sınıf")
    ax.set_ylabel("Gerçek sınıf")
    ax.set_title(f"Karışıklık matrisi — hibrit model\n{DENEY_ETIKETLERI.get(ad, ad)}, n = {int(cm.sum())}",
                 fontsize=12, fontweight="bold", pad=10)
    return _kaydet(fig, "confusion_matrix.png")


def model_karsilastirma(metrik):
    deneyler = [d for d in DENEY_ETIKETLERI if d in metrik["deneyler"]]
    modeller = list(MODEL_ETIKETLERI)
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=ARKA)
    _tema(ax)
    gen = 0.8 / len(modeller)
    for i, m in enumerate(modeller):
        ozet = [metrik["deneyler"][d]["modeller"][m] for d in deneyler]
        ort = [o["macro_f1_ort"] for o in ozet]
        std = [o["macro_f1_std"] if len(o["katlar"]) > 1 else 0 for o in ozet]
        x = np.arange(len(deneyler)) + (i - (len(modeller) - 1) / 2) * gen
        ax.bar(x, ort, gen * 0.92, yerr=std, color=MODEL_RENKLERI[m], label=MODEL_ETIKETLERI[m],
               error_kw={"ecolor": YAZI, "capsize": 3, "lw": 1})
        for xi, o in zip(x, ort):
            ax.text(xi, min(o + 0.04, 1.06), f"{o:.2f}", ha="center", va="bottom", color=YAZI, fontsize=8)
    ax.axhline(1 / 3, color=IZGARA, ls=":", lw=1.2)
    ax.text(len(deneyler) - 0.52, 1 / 3 + 0.01, "şans düzeyi", color=IZGARA, fontsize=8, ha="right")
    ax.set_xticks(np.arange(len(deneyler)))
    ax.set_xticklabels([DENEY_ETIKETLERI[d] for d in deneyler], color=YAZI)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Macro-F1")
    ax.grid(axis="y", color=IZGARA, alpha=0.15, ls="--")
    ax.set_title("Model karşılaştırması (hata çubukları: katlar arası standart sapma)",
                 fontsize=12, fontweight="bold", pad=10)
    _lejant(ax, loc="lower right", ncol=4)
    return _kaydet(fig, "model_karsilastirma.png")


def egitim_egrisi(metrik):
    final = metrik.get("final")
    if not final:
        print("  [atlandı] egitim_egrisi.png — metrics.json içinde 'final' yok (scripts/egit_final.py çalıştırın)")
        return None
    g = final["gecmis"]
    epoch = [e["epoch"] for e in g]
    fig, ax1 = plt.subplots(figsize=(9, 5), facecolor=ARKA)
    _tema(ax1)
    c1 = ax1.plot(epoch, [e["egitim_kaybi"] for e in g], color="#4fc3f7", marker="o", lw=2.3, label="Eğitim kaybı")
    c2 = ax1.plot(epoch, [e["dogrulama_kaybi"] for e in g], color="#ffb300", marker="^", lw=2.3, label="Doğrulama kaybı")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Kayıp (cross-entropy)")
    ax1.set_xticks(epoch)
    ax1.grid(True, color=IZGARA, alpha=0.15, ls="--")
    ax2 = ax1.twinx()
    _tema(ax2)
    c3 = ax2.plot(epoch, [e["dogrulama_f1"] for e in g], color="#66bb6a", marker="s", ls="--", lw=2.3,
                  label="Doğrulama macro-F1")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Macro-F1")
    ax1.axvline(final["en_iyi_epoch"], color=YAZI, ls=":", lw=1.2)
    ax1.text(final["en_iyi_epoch"], ax1.get_ylim()[1], " seçilen epoch", color=YAZI, fontsize=8.5, va="top")
    cizgiler = c1 + c2 + c3
    ax1.legend(cizgiler, [c.get_label() for c in cizgiler], loc="center right", facecolor=KUTU,
               edgecolor=IZGARA, labelcolor=YAZI, fontsize=9)
    ax1.set_title(f"Nihai modelin eğitim süreci (eğitim n = {final['n_egitim']}, doğrulama n = {final['n_dogrulama']})",
                  fontsize=12, fontweight="bold", pad=10)
    return _kaydet(fig, "egitim_egrisi.png")


def stilometri_karsilastirma(metrik):
    s = metrik["veri"]["stilometri"]
    z = np.array(s["sinif_z"])
    etiketler = [OZELLIK_ETIKETLERI.get(o, o) for o in s["ozellikler"]]
    fig, ax = plt.subplots(figsize=(10.5, 6), facecolor=ARKA)
    _tema(ax)
    y = np.arange(len(etiketler))
    yuk = 0.26
    for c, (ad, renk) in enumerate(zip(SINIF_KISA, SINIF_RENKLERI)):
        ax.barh(y + (c - 1) * yuk, z[:, c], yuk * 0.92, color=renk, label=ad)
    ax.axvline(0, color=YAZI, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(etiketler, color=YAZI)
    ax.invert_yaxis()
    ax.set_xlabel("Sınıf ortalamasının genel ortalamadan farkı (standart sapma birimi)")
    ax.grid(axis="x", color=IZGARA, alpha=0.15, ls="--")
    ax.set_title(f"Sınıflara göre stilometrik özellikler (n = {metrik['veri']['n']})",
                 fontsize=12, fontweight="bold", pad=10)
    _lejant(ax, loc="lower right")
    return _kaydet(fig, "stilometri_karsilastirma.png")


def ozellik_radar(metrik):
    s = metrik["veri"]["stilometri"]
    z = np.array(s["sinif_z"])
    etiketler = [OZELLIK_ETIKETLERI.get(o, o).split(" (")[0].replace(" oranı", "") for o in s["ozellikler"]]
    aci = np.linspace(0, 2 * np.pi, len(etiketler), endpoint=False).tolist()
    aci += aci[:1]
    sinir = max(1.0, float(np.abs(z).max()) * 1.1)
    fig, eksenler = plt.subplots(1, 3, figsize=(15, 5.4), facecolor=ARKA, subplot_kw={"projection": "polar"})
    for c, ax in enumerate(eksenler):
        deger = z[:, c].tolist()
        deger += deger[:1]
        ax.set_facecolor(ARKA)
        ax.plot(aci, [0] * len(aci), color=IZGARA, lw=1, ls=":")
        ax.plot(aci, deger, color=SINIF_RENKLERI[c], lw=2.2)
        ax.fill(aci, deger, color=SINIF_RENKLERI[c], alpha=0.25)
        ax.set_ylim(-sinir, sinir)
        ax.set_xticks(aci[:-1])
        ax.set_xticklabels(etiketler, color=YAZI, fontsize=7.5)
        ax.tick_params(axis="y", colors=YAZI, labelsize=6.5)
        ax.grid(color=IZGARA, alpha=0.3, ls="--")
        ax.spines["polar"].set_color(IZGARA)
        ax.set_title(SINIF_KISA[c], color=SINIF_RENKLERI[c], fontsize=12, fontweight="bold", pad=16)
    fig.suptitle("Sınıfların stilometrik profili (genel ortalamadan sapma, std birimi; noktalı çember = ortalama)",
                 color=YAZI, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _kaydet(fig, "ozellik_radar.png")


def demo_trend():
    if not DEMO_TREND_YOLU.exists():
        print("  [atlandı] demo_trend.png — results/demo_trend.json yok (scripts/egit_final.py çalıştırın)")
        return None
    demo = json.loads(DEMO_TREND_YOLU.read_text(encoding="utf-8"))
    personeller = list(dict.fromkeys(k["Personel"] for k in demo["kayitlar"]))
    fig, eksenler = plt.subplots(len(personeller), 1, figsize=(11, 2.6 * len(personeller)), facecolor=ARKA, sharex=True)
    for ax, p in zip(np.atleast_1d(eksenler), personeller):
        _tema(ax)
        kayit = sorted((k for k in demo["kayitlar"] if k["Personel"] == p), key=lambda k: k["Tarih"])
        gun = np.arange(1, len(kayit) + 1)
        risk = [k["risk_skoru"] for k in kayit]
        ax.plot(gun, risk, color=YAZI, lw=1.6, alpha=0.7, zorder=2)
        ax.scatter(gun, risk, c=[SINIF_RENKLERI[k["sinif"]] for k in kayit], s=70, zorder=3, edgecolors=ARKA)
        if kayit[0].get("Beklenen") is not None:
            for g, k in zip(gun, kayit):   # senaryo tasarımındaki sınıf, alt şerit olarak
                ax.axvspan(g - 0.5, g + 0.5, ymin=0, ymax=0.06, color=SINIF_RENKLERI[k["Beklenen"]], alpha=0.9)
        ax.set_ylim(-0.08, 1.05)
        ax.set_xlim(0.5, len(kayit) + 0.5)
        ax.set_xticks(gun)
        ax.set_ylabel("Risk skoru")
        ax.grid(True, color=IZGARA, alpha=0.15, ls="--")
        ax.set_title(p, fontsize=10.5, loc="left", fontweight="bold")
    np.atleast_1d(eksenler)[-1].set_xlabel("Gün")
    tutamaclar = [plt.Line2D([], [], marker="o", ls="", color=r, label=a) for a, r in zip(SINIF_KISA, SINIF_RENKLERI)]
    np.atleast_1d(eksenler)[0].legend(handles=tutamaclar, facecolor=KUTU, edgecolor=IZGARA, labelcolor=YAZI,
                                      fontsize=8.5, loc="upper left", ncol=3, title="Modelin tahmini (nokta) · senaryo tasarımı (alt şerit)",
                                      title_fontsize=8)
    np.atleast_1d(eksenler)[0].get_legend().get_title().set_color(YAZI)
    uyum = demo.get("tasarimla_uyusma")
    alt = "" if uyum is None else f" — tasarımla uyuşma %{uyum * 100:.0f}"
    fig.suptitle(f"KURGUSAL SENARYO: 14 günlük günlüklerde modelin risk skoru{alt}",
                 color=YAZI, fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    return _kaydet(fig, "demo_trend.png")


def main():
    if not METRIK_YOLU.exists():
        sys.exit(f"{METRIK_YOLU} yok. Önce `python scripts/deney.py` çalıştırın.")
    metrik = json.loads(METRIK_YOLU.read_text(encoding="utf-8"))
    print("Grafikler üretiliyor (kaynak: results/metrics.json)")
    karisiklik_matrisi(metrik)
    model_karsilastirma(metrik)
    egitim_egrisi(metrik)
    stilometri_karsilastirma(metrik)
    ozellik_radar(metrik)
    demo_trend()


if __name__ == "__main__":
    main()
