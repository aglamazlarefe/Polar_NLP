"""Ortak test araçları.

Testler internet ve gerçek BERTurk ağırlıkları GEREKTİRMEZ: rastgele başlatılmış küçücük
bir BERT ve harf tabanlı bir tokenizer kullanılır. Böylece tüm paket saniyeler içinde koşar.
"""

from __future__ import annotations

import json

import pytest
from transformers import BertConfig, BertModel, BertTokenizer

from polar_nlp.cikarim import OLCEKLEYICI_DOSYASI, TOKENIZER_DIZINI
from polar_nlp.config import EgitimAyari
from polar_nlp.egitim import egit
from polar_nlp.model import HybridPolarModel
from polar_nlp.stilometri import N_OZELLIK, Olcekleyici, stilometri_matrisi
from polar_nlp.veri import PolarDataset

ORNEK_METINLER = [
    "Jeneratör bakımı tamamlandı. Voltaj değerleri kaydedildi ve sistem çalışıyor.",
    "Yakıt hattındaki sızıntı giderildi. Vanalar kontrol edildi, basınç normal.",
    "Radyosonde balonu saat sekizde fırlatıldı. Veriler merkeze aktarılıyor.",
    "Kalibrasyon yine saatlerimi aldı. Yedek parça gelmediği için asıl işlerime vakit kalmıyor.",
    "Aynı sayımı üçüncü kez yaptım ve yine tutmadı. Uykusuzluk yüzünden dikkatimi toplayamıyorum.",
    "Sensör arızası her gün tekrarlıyor. Taleplerim karşılanmadığından ölçümler sürekli gecikiyor.",
    "Günlük form doldurulmayacak. Bu işin kimseye faydası yok, uğraşmıyorum.",
    "Anten temizliği listeden çıkarıldı. Kontrol etmeyeceğim, sonucu da umursamıyorum.",
    "Akşam yemeğine katılmadım. Toplantıya da gitmeyeceğim, hiçbir şey değişmiyor.",
]
ORNEK_ETIKETLER = [0, 0, 0, 1, 1, 1, 2, 2, 2]


@pytest.fixture(scope="session")
def kucuk_tokenizer(tmp_path_factory):
    harfler = list("abcçdefgğhıijklmnoöprsştuüvyzwxq") + list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ") + list("0123456789.,!?'-:;()/%")
    sozluk = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + harfler + ["##" + h for h in harfler]
    yol = tmp_path_factory.mktemp("tokenizer") / "vocab.txt"
    yol.write_text("\n".join(sozluk), encoding="utf-8")
    return BertTokenizer(str(yol), do_lower_case=False)


@pytest.fixture
def kucuk_model_uret(kucuk_tokenizer):
    def _uret(stilo_kullan: bool = True) -> HybridPolarModel:
        ayar = BertConfig(vocab_size=len(kucuk_tokenizer), hidden_size=32, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
        return HybridPolarModel(BertModel(ayar), N_OZELLIK, stilo_kullan=stilo_kullan)
    return _uret


@pytest.fixture(scope="session")
def ornek_veri():
    metinler, etiketler = ORNEK_METINLER * 3, ORNEK_ETIKETLER * 3
    olcek = Olcekleyici().uydur(stilometri_matrisi(metinler), etiketler)
    return PolarDataset(metinler, etiketler, olcek), olcek


@pytest.fixture(scope="session")
def kucuk_model_dizini(tmp_path_factory, kucuk_tokenizer, ornek_veri):
    """Diskte tam bir model klasörü (ağırlık + config + tokenizer + ölçekleyici)."""
    veri, olcek = ornek_veri
    ayar = BertConfig(vocab_size=len(kucuk_tokenizer), hidden_size=32, num_hidden_layers=1,
                      num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
    model = HybridPolarModel(BertModel(ayar), N_OZELLIK)
    sonuc = egit(model, kucuk_tokenizer, veri, veri, EgitimAyari(epoch=2, batch=9, sabir=None), gunluk=lambda s: None)
    dizin = tmp_path_factory.mktemp("model")
    model.kaydet(dizin, sonuc["en_iyi_durum"], {"sicaklik": 1.5, "taban_model": "test", "n_egitim": len(veri),
                                                "n_dogrulama": len(veri), "en_iyi_epoch": sonuc["en_iyi_epoch"]})
    kucuk_tokenizer.save_pretrained(str(dizin / TOKENIZER_DIZINI))
    (dizin / OLCEKLEYICI_DOSYASI).write_text(json.dumps(olcek.sozluk()), encoding="utf-8")
    return dizin
