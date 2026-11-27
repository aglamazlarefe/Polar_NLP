"""Veri seti sağlık raporu: sınıf/uzunluk dengesi ve "kalıp kelime" sızıntısı.

Kalıp kelime = neredeyse yalnız tek sınıfta geçen kelime. Çok sayıda kalıp kelime varsa model
üslubu değil o kelimeleri ezberler ve skorlar yapay biçimde yüksek çıkar.

    python scripts/veri_analiz.py                 # data/polar_v2.csv
    python scripts/veri_analiz.py data/polar_v1.csv
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polar_nlp.config import N_SINIF, VERI_YOLU  # noqa: E402
from polar_nlp.stilometri import cumlelere_ayir, kelimelere_ayir  # noqa: E402
from polar_nlp.veri import veri_yukle  # noqa: E402


def kalip_kelimeler(metinler, etiketler, min_belge: int = 8, min_pay: float = 0.9) -> list[tuple[str, int, int, float]]:
    """(kelime, baskın sınıf, belge sayısı, baskın sınıf payı) — paya göre sıralı."""
    sayac = [Counter() for _ in range(N_SINIF)]
    for metin, e in zip(metinler, etiketler):
        sayac[e].update(set(kelimelere_ayir(metin)))
    sonuc = []
    for k in set().union(*sayac):
        adet = [s[k] for s in sayac]
        toplam = sum(adet)
        if toplam >= min_belge and max(adet) / toplam >= min_pay:
            sonuc.append((k, adet.index(max(adet)), toplam, max(adet) / toplam))
    return sorted(sonuc, key=lambda x: (-x[3], -x[2]))


def rapor(df, baslik: str) -> dict:
    print(f"\n## {baslik} — {len(df)} metin")
    df = df.assign(_kelime=df["Metin"].map(lambda m: len(kelimelere_ayir(m))),
                   _cumle=df["Metin"].map(lambda m: len(cumlelere_ayir(m))))
    ozet = df.groupby("Etiket").agg(adet=("Metin", "size"), ort_kelime=("_kelime", "mean"),
                                    ort_cumle=("_cumle", "mean")).round(2)
    print(ozet.to_string())
    kalip = kalip_kelimeler(df["Metin"], df["Etiket"])
    kapsama = [sum(any(k in set(kelimelere_ayir(m)) for k, s, _, _ in kalip if s == e) for m in df[df.Etiket == e]["Metin"])
               / max((df.Etiket == e).sum(), 1) for e in range(N_SINIF)]
    print(f"kalıp kelime sayısı: {len(kalip)} | en az bir kalıp kelime içeren metin oranı (sınıf 0/1/2): "
          + " / ".join(f"{k:.0%}" for k in kapsama))
    for k, s, n, pay in kalip[:15]:
        print(f"  {k:<18} sınıf {s}  {n:>3} metin  %{pay * 100:.0f}")
    return {"kalip_kelime_sayisi": len(kalip), "kalip_kapsama": [round(k, 4) for k in kapsama]}


def main():
    df = veri_yukle(sys.argv[1] if len(sys.argv) > 1 else VERI_YOLU)
    rapor(df, "Tüm veri")
    if df["Kaynak"].nunique() > 1:
        for kaynak, parca in df.groupby("Kaynak"):
            rapor(parca, f"Kaynak = {kaynak}")
        print("\n## Zorluk dağılımı\n" + df.groupby(["Kaynak", "Zorluk"]).size().to_string())
    print("\n## Konu dağılımı\n" + df.groupby(["Konu", "Kaynak"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
