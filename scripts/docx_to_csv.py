"""Özgün Word dosyasını ("Metin | Etiket" satırları) CSV'ye çevirir. v1'in nasıl üretildiğinin kaydıdır.

    python scripts/docx_to_csv.py [girdi.docx] [cikti.csv]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import docx

KOK = Path(__file__).resolve().parent.parent
VARSAYILAN_GIRDI = KOK / "data" / "kaynak" / "seyir_defteri_veri_seti.docx"
VARSAYILAN_CIKTI = KOK / "data" / "polar_v1.csv"


def cevir(girdi: Path, cikti: Path) -> int:
    satirlar = []
    for paragraf in docx.Document(str(girdi)).paragraphs:
        metin, ayrac, etiket = paragraf.text.strip().rpartition("|")
        metin, etiket = metin.strip(), etiket.strip()
        if not ayrac or not etiket.isdigit() or not metin:
            continue   # başlık, ayraç çizgileri, boş satırlar
        satirlar.append((metin, int(etiket)))
    with open(cikti, "w", encoding="utf-8", newline="") as f:
        yazar = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        yazar.writerow(["Metin", "Etiket"])
        yazar.writerows(satirlar)
    return len(satirlar)


if __name__ == "__main__":
    girdi = Path(sys.argv[1]) if len(sys.argv) > 1 else VARSAYILAN_GIRDI
    cikti = Path(sys.argv[2]) if len(sys.argv) > 2 else VARSAYILAN_CIKTI
    print(f"{cevir(girdi, cikti)} satır -> {cikti}")
