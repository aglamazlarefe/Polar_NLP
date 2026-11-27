"""Türkçe'ye uygun stilometrik özellikler.

Türkçe sondan eklemeli ve özne düşüren bir dildir: "ben" çoğu zaman yazılmaz, kişi ve
olumsuzluk fiil ekleriyle verilir ("gitmiyorum"). Bu yüzden özellikler kelime listesi
yerine büyük ölçüde EK örüntülerine bakar.

Örüntüler biçimbilimsel çözümleyici değil, sezgisel düzenli ifadelerdir; yanlış pozitif
ve negatifleri vardır (bkz. docs/MIMARI.md). Özellik kategorileri literatürden gelir
(LIWC: birinci tekil kişi, olumsuzluk, nedensellik; Al-Mosaiwi & Johnstone 2018:
mutlakçı dil) — veri setinden sınıf ayıran kelime devşirilmez.

Yeni özellik eklemek için: bir `_ozellik` fonksiyonu yazıp `_OZELLIKLER` listesine ekleyin.
`OZELLIK_ADLARI` ve model giriş boyutu kendiliğinden güncellenir; modeli yeniden eğitin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Türkçe metin yardımcıları
# --------------------------------------------------------------------------


def tr_kucult(metin: str) -> str:
    """Türkçe kurallarıyla küçük harfe çevirir (I→ı, İ→i). `str.lower()` bunu yapmaz."""
    return metin.replace("I", "ı").replace("İ", "i").lower()


_KELIME = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
_CUMLE_SONU = re.compile(r"(?<=[.!?…])\s+")


def kelimelere_ayir(metin: str) -> list[str]:
    """Küçük harfli kelime listesi. Sayılar atılır; kesme işaretli ekler kelimeye yapışık kalır."""
    return [k.replace("’", "'") for k in _KELIME.findall(tr_kucult(metin))]


def cumlelere_ayir(metin: str) -> list[str]:
    """Nokta/ünlem/soru işaretinden sonra boşluk gelen yerlerden böler ("3.5" bölünmez)."""
    return [c for c in _CUMLE_SONU.split(metin.strip()) if _KELIME.search(c)]


# --------------------------------------------------------------------------
# Ek ve sözlük örüntüleri
# --------------------------------------------------------------------------

_BIRINCI_TEKIL_SOZCUK = {
    "ben", "beni", "bana", "benim", "bende", "benden", "bence", "benimle",
    "kendim", "kendimi", "kendime", "kendimde", "kendimden",
    "tarafımca", "tarafımdan",
    "değilim", "değildim", "yokum", "yoktum",
}
# -DIm ile biten ama fiil olmayan yaygın adlar (yanlış pozitifleri azaltır).
_BIRINCI_TEKIL_HARIC = {
    "yardım", "adım", "eğitim", "iletim", "üretim", "yönetim", "öğretim", "gözetim",
    "denetim", "işletim", "anlatım", "yalıtım", "dağıtım", "satım", "tüm", "tutum",
    "bitim", "yetim", "tüketim", "iletişim", "kadim", "takdim", "hortum", "kostüm",
    "oturum", "tutum", "bütüm",
}
_BIRINCI_TEKIL_EK = re.compile(
    r"(?:"
    r"[ıiuü]yorum"                                   # ediyorum
    r"|[dt](?:ım|im|um|üm)"                          # ettim, aldım
    r"|(?:acağım|eceğim)"                            # yapacağım
    r"|(?:malıyım|meliyim)"                          # yapmalıyım
    r"|(?:mışım|mişim|muşum|müşüm)"                  # unutmuşum
    r"|[dt](?:ığım|iğim|uğum|üğüm)(?:[ıiuü]|[ae]|d[ae]n?|[ıiuü]n)?"   # ilettiğim, yaptığımı
    r"|l[ae]r[ıi]m(?:[ıi]|[ae]|d[ae]n?|[ıi]n)?"      # işlerim, saatlerimi
    r")$"
)

_OLUMSUZ_SOZCUK = {"değil", "değildir", "değildi", "yok", "yoktur", "yoktu", "hayır",
                   "değilim", "değiliz", "değilsin", "değildim", "yokum", "yoktum"}
# Olumsuzluk eki gibi görünen ama öyle olmayan sözcük kökleri ("telsiz" = radyo).
_OLUMSUZ_HARIC_KOK = ("telsiz", "mütemadiyen", "hırsız", "komedi", "pekmez", "yılmaz", "kurmay")
_OLUMSUZ_EK = re.compile(
    r"\w{2,}(?:"
    r"m[ıiuü]yor"                 # gelmiyor
    r"|m[ae]d[ıi]"                # yapılmadı, edilmediğinden
    r"|m[ae]m[ıi]ş"               # yapılmamış
    r"|m[ae]y[ae]c[ae][kğ]"       # yapılmayacak
    r"|m[ae]m[ae](?:s[ıi]|k|y[ıi])"  # yapılmaması, planlanmamakta
    r"|m[ae]y[ae]n$"              # gelmeyen
    r"|m[ae]d[ae]n$"              # beklemeden
    r"|m[ae]z(?:$|s[ae]|l[ae]r|d[ıi])"   # olmaz, çalışmazsa
    r"|s[ıiuü]z(?:$|l[ıiuü][kğ])"        # yetersiz, imkansızlık
    r")"
)

_RESMI_HARIC = {"bildir", "kaldır", "saldır", "doldur", "durdur", "yıldır", "aldır", "prosedür", "kültür",
                "takdir", "kadir"}
# Ettirgen/emir gövdeleri: değiştir, geliştir, dönüştür (ama "-mIştIr" bildirmedir), güçlendir, değerlendir.
_RESMI_ETTIRGEN = re.compile(r"(?:(?<!m[ıiuü])ş[dt][ıiuü]r|l[ae]nd[ıi]r)$")
_RESMI_EK = re.compile(r"\w{3,}[dt][ıiuü]r$")   # tamamlanmıştır, yapılmaktadır, anlamsızdır

_EDILGEN_SONRASI = (
    r"(?:d[ıiuü](?!r)|m[ıiuü]ş|[ıiuü]yor|m[ae]kt|[ae]c[ae][kğ]|[ae]r[ae]k|[ae]n|m[ae]s[ıi]"
    r"|[ıiuü]p|m[ae]d|m[ae]y|m[ae]m|d[ıiuü]ğ|m[ıiuü]yor)"
)
_EDILGEN_EK = re.compile(
    r"(?:\w{2,}(?:ıl|il|ul|ül)" + _EDILGEN_SONRASI + r")"       # yapıldı, edilmiş, çözülüyor
    r"|(?:\w{2,}l[ae]n" + _EDILGEN_SONRASI + r")"               # tamamlandı, temizlendi
    r"|(?:\w+l(?:ın|in|un|ün)" + _EDILGEN_SONRASI + r")"        # alındı, bilinen
)
# Edilgen olmayan -lAn-/-Un- kökleri: etken "kullandı", dönüşlü "odaklandı", "bulunuyor" = mevcut,
# oluş fiilleri "evlendi", "yaşlandı". Bu köklerden sonra -Il- gelirse ("kullanıldı") yine edilgendir.
_EDILGEN_HARIC_KOK = ("kullan", "bulun", "dalgalan", "odaklan", "havalan", "yararlan", "faydalan",
                      "evlen", "yaşlan", "hastalan", "sevin", "sinirlen", "heyecanlan", "endişelen")
_EDILGEN_IL = re.compile(r"[ıiuü]l")

_MUTLAKCI_SOZCUK = {
    "asla", "hiç", "hiçbir", "hiçbiri", "kesinlikle", "daima", "hep", "tamamen",
    "tümüyle", "bütünüyle", "tamamıyla", "mutlaka", "tüm", "bütün", "herkes",
    "kimse", "sürekli", "katiyen", "kesin", "tam",
}
_MUTLAKCI_OBEK = re.compile(r"\bher (?:zaman|şey|gün|defasında|seferinde)\w*")

_NEDENSEL_SOZCUK = {
    "çünkü", "yüzünden", "yüzden", "nedeniyle", "nedenle", "sebebiyle", "sebeple",
    "dolayı", "dolayısıyla", "ötürü", "neticesinde", "sonucunda", "sonucu", "zira",
    "sayesinde", "rağmen", "halde", "gerekçesiyle", "kaynaklı",
}
_NEDENSEL_EK = re.compile(r"\w{2,}[dt][ıiuü]ğ[ıiuü]nd[ae]n$")   # edilmediğinden


# --------------------------------------------------------------------------
# Özellikler
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Metin:
    ham: str
    kucuk: str
    kelimeler: list[str]
    cumleler: list[str]

    @property
    def n(self) -> int:
        return max(len(self.kelimeler), 1)


def _oran(m: _Metin, kosul) -> float:
    return sum(1 for k in m.kelimeler if kosul(k)) / m.n


def _birinci_tekil(k: str) -> bool:
    if k in _BIRINCI_TEKIL_SOZCUK:
        return True
    return k not in _BIRINCI_TEKIL_HARIC and len(k) >= 5 and bool(_BIRINCI_TEKIL_EK.search(k))


def _olumsuz(k: str) -> bool:
    if k in _OLUMSUZ_SOZCUK:
        return True
    return not k.startswith(_OLUMSUZ_HARIC_KOK) and bool(_OLUMSUZ_EK.match(k))


def _resmi(k: str) -> bool:
    return k not in _RESMI_HARIC and bool(_RESMI_EK.match(k)) and not _RESMI_ETTIRGEN.search(k)


def _edilgen(k: str) -> bool:
    for kok in _EDILGEN_HARIC_KOK:
        if k.startswith(kok):
            return bool(_EDILGEN_IL.match(k[len(kok):]))
    return bool(_EDILGEN_EK.match(k))


def _nedensel(k: str) -> bool:
    return k in _NEDENSEL_SOZCUK or bool(_NEDENSEL_EK.match(k))


def _mutlakci(m: _Metin) -> float:
    sayi = sum(1 for k in m.kelimeler if k in _MUTLAKCI_SOZCUK)
    return (sayi + len(_MUTLAKCI_OBEK.findall(m.kucuk))) / m.n


# (ad, panelde görünen etiket, fonksiyon)
_OZELLIKLER = [
    ("birinci_tekil", "1. tekil kişi oranı", lambda m: _oran(m, _birinci_tekil)),
    ("olumsuzluk", "Olumsuzluk oranı", lambda m: _oran(m, _olumsuz)),
    ("resmi_bildirme", "Resmî bildirme (-DIr) oranı", lambda m: _oran(m, _resmi)),
    ("edilgen", "Edilgen çatı oranı", lambda m: _oran(m, _edilgen)),
    ("mutlakci", "Mutlakçı ifade oranı", _mutlakci),
    ("nedensellik", "Nedensellik bağlacı oranı", lambda m: _oran(m, _nedensel)),
    ("ort_cumle_uzunlugu", "Ortalama cümle uzunluğu (kelime)", lambda m: m.n / max(len(m.cumleler), 1)),
    ("ort_kelime_uzunlugu", "Ortalama kelime uzunluğu (harf)", lambda m: sum(map(len, m.kelimeler)) / m.n),
    ("ttr", "Kelime çeşitliliği (TTR)", lambda m: len(set(m.kelimeler)) / m.n),
    ("cumle_sayisi", "Cümle sayısı", lambda m: float(len(m.cumleler))),
]

OZELLIK_ADLARI = [ad for ad, _, _ in _OZELLIKLER]
OZELLIK_ETIKETLERI = {ad: etiket for ad, etiket, _ in _OZELLIKLER}
N_OZELLIK = len(OZELLIK_ADLARI)


def stilometri_cikar(metin: str) -> np.ndarray:
    """Tek metin için ham (ölçeklenmemiş) özellik vektörü, `OZELLIK_ADLARI` sırasıyla."""
    metin = metin if isinstance(metin, str) else ""
    m = _Metin(metin, tr_kucult(metin), kelimelere_ayir(metin), cumlelere_ayir(metin))
    return np.array([f(m) for _, _, f in _OZELLIKLER], dtype=np.float32)


def stilometri_matrisi(metinler) -> np.ndarray:
    metinler = list(metinler)
    if not metinler:
        return np.zeros((0, N_OZELLIK), dtype=np.float32)
    return np.stack([stilometri_cikar(t) for t in metinler])


# --------------------------------------------------------------------------
# Ölçekleme (pickle yerine JSON'a yazılabilir)
# --------------------------------------------------------------------------


class Olcekleyici:
    """Standart ölçekleyici. Ayrıca "objektif" sınıfın dağılımını saklar; çıkarımda
    her özelliğin objektif raporlara göre z-skoru bununla hesaplanır."""

    KIRPMA = 5.0

    def __init__(self, ort=None, std=None, objektif_ort=None, objektif_std=None):
        self.ort = None if ort is None else np.asarray(ort, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.objektif_ort = None if objektif_ort is None else np.asarray(objektif_ort, dtype=np.float32)
        self.objektif_std = None if objektif_std is None else np.asarray(objektif_std, dtype=np.float32)

    @staticmethod
    def _guvenli_std(x: np.ndarray) -> np.ndarray:
        s = x.std(axis=0)
        return np.where(s < 1e-6, 1.0, s).astype(np.float32)

    def uydur(self, ham: np.ndarray, etiketler=None) -> "Olcekleyici":
        ham = np.asarray(ham, dtype=np.float32)
        self.ort = ham.mean(axis=0)
        self.std = self._guvenli_std(ham)
        objektif = ham[np.asarray(etiketler) == 0] if etiketler is not None else ham
        if len(objektif) == 0:
            objektif = ham
        self.objektif_ort = objektif.mean(axis=0)
        self.objektif_std = self._guvenli_std(objektif)
        return self

    def donustur(self, ham: np.ndarray) -> np.ndarray:
        z = (np.asarray(ham, dtype=np.float32) - self.ort) / self.std
        return np.clip(z, -self.KIRPMA, self.KIRPMA).astype(np.float32)

    def objektife_gore_z(self, ham: np.ndarray) -> np.ndarray:
        z = (np.asarray(ham, dtype=np.float32) - self.objektif_ort) / self.objektif_std
        return np.clip(z, -self.KIRPMA, self.KIRPMA)

    def sozluk(self) -> dict:
        return {
            "ozellikler": OZELLIK_ADLARI,
            "ort": self.ort.tolist(),
            "std": self.std.tolist(),
            "objektif_ort": self.objektif_ort.tolist(),
            "objektif_std": self.objektif_std.tolist(),
        }

    @classmethod
    def sozlukten(cls, d: dict) -> "Olcekleyici":
        if d.get("ozellikler") != OZELLIK_ADLARI:
            raise ValueError(
                "Kayıtlı ölçekleyicinin özellik listesi koddakiyle uyuşmuyor; "
                "modeli yeniden eğitin (python scripts/egit_final.py)."
            )
        return cls(d["ort"], d["std"], d["objektif_ort"], d["objektif_std"])
