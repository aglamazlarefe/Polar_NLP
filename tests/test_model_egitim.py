import numpy as np
import torch

from polar_nlp.config import N_SINIF, EgitimAyari
from polar_nlp.egitim import egit, metrikler, sicaklik_bul, tahmin_et
from polar_nlp.model import HybridPolarModel
from polar_nlp.stilometri import N_OZELLIK

SESSIZ = {"gunluk": lambda s: None}


def test_forward_sekli_hibrit_ve_yalniz_bert(kucuk_model_uret):
    ids, maske = torch.randint(5, 50, (4, 12)), torch.ones(4, 12, dtype=torch.long)
    hibrit, yalniz = kucuk_model_uret(True), kucuk_model_uret(False)
    assert hibrit(ids, maske, torch.zeros(4, N_OZELLIK)).shape == (4, N_SINIF)
    assert yalniz(ids, maske, None).shape == (4, N_SINIF)        # stilometri gerekmez
    assert not hasattr(yalniz, "stilo_dali")


def test_stilometri_hibrit_ciktisini_etkiler(kucuk_model_uret):
    model = kucuk_model_uret(True).eval()
    ids, maske = torch.randint(5, 50, (2, 10)), torch.ones(2, 10, dtype=torch.long)
    with torch.no_grad():
        a = model(ids, maske, torch.zeros(2, N_OZELLIK))
        b = model(ids, maske, torch.ones(2, N_OZELLIK) * 3)
    assert not torch.allclose(a, b)


def test_en_iyi_durum_gercek_anlik_goruntudur(kucuk_model_uret, kucuk_tokenizer, ornek_veri, monkeypatch):
    """Regresyon: eski kod `state_dict().copy()` (sığ kopya) kullanıyordu; "en iyi" ağırlıklar
    sonraki epoch'larla birlikte değişiyor, yani hep SON epoch kaydediliyordu."""
    import polar_nlp.egitim as egitim_modulu
    veri, _ = ornek_veri
    sahte_f1 = iter([0.9, 0.5, 0.4, 0.3])       # en iyi epoch kesin olarak 1 olsun
    gercek = egitim_modulu.metrikler
    monkeypatch.setattr(egitim_modulu, "metrikler",
                        lambda y, t: {**gercek(y, t), "macro_f1": next(sahte_f1)})
    model = kucuk_model_uret()
    anlik = {}

    def kaydet(satir):                           # epoch 1 günlüğü yazılırken ağırlıkları yakala
        if "epoch 1/" in satir:
            anlik.update({k: v.detach().clone() for k, v in model.state_dict().items()})

    sonuc = egit(model, kucuk_tokenizer, veri, veri, EgitimAyari(epoch=4, batch=9, sabir=None), gunluk=kaydet)
    assert sonuc["en_iyi_epoch"] == 1 and len(sonuc["gecmis"]) == 4
    for k, v in sonuc["en_iyi_durum"].items():
        assert torch.equal(v, anlik[k]), f"{k}: en iyi durum epoch 1 ağırlıklarıyla aynı değil"
    for k, v in model.state_dict().items():      # model en iyi ağırlıklara geri yüklenmiş olmalı
        assert torch.equal(v.cpu(), anlik[k])


def test_zamanlayici_sifira_iner_ve_erken_durdurma(kucuk_model_uret, kucuk_tokenizer, ornek_veri, monkeypatch):
    veri, _ = ornek_veri
    tam = egit(kucuk_model_uret(), kucuk_tokenizer, veri, veri, EgitimAyari(epoch=3, batch=9, sabir=None), **SESSIZ)
    assert tam["gecmis"][-1]["lr"] == 0.0        # eski kodda adım sayısı yanlıştı, LR sıfıra inmiyordu
    assert tam["gecmis"][0]["lr"] > tam["gecmis"][1]["lr"] > 0

    import polar_nlp.egitim as egitim_modulu
    sahte_f1 = iter([0.9, 0.5, 0.4, 0.3, 0.2, 0.1])
    gercek = egitim_modulu.metrikler
    monkeypatch.setattr(egitim_modulu, "metrikler", lambda y, t: {**gercek(y, t), "macro_f1": next(sahte_f1)})
    erken = egit(kucuk_model_uret(), kucuk_tokenizer, veri, veri, EgitimAyari(epoch=6, batch=9, sabir=2), **SESSIZ)
    assert len(erken["gecmis"]) == 3 and erken["en_iyi_epoch"] == 1


def test_ayni_tohum_ayni_sonuc(kucuk_model_uret, kucuk_tokenizer, ornek_veri):
    veri, _ = ornek_veri

    def kos():
        torch.manual_seed(0)
        model = kucuk_model_uret()
        egit(model, kucuk_tokenizer, veri, veri, EgitimAyari(epoch=2, batch=9, sabir=None, seed=7), **SESSIZ)
        return tahmin_et(model, kucuk_tokenizer, veri)

    assert np.allclose(kos(), kos(), atol=1e-6)


def test_egitim_kaybi_azalir_ve_sinif_agirligi_calisir(kucuk_model_uret, kucuk_tokenizer, ornek_veri):
    veri, _ = ornek_veri
    ayar = EgitimAyari(epoch=4, batch=9, sabir=None, sinif_agirligi=True, lr_bert=1e-3)
    gecmis = egit(kucuk_model_uret(), kucuk_tokenizer, veri, veri, ayar, **SESSIZ)["gecmis"]
    assert gecmis[-1]["egitim_kaybi"] < gecmis[0]["egitim_kaybi"]


def test_kaydet_yukle_ayni_ciktiyi_verir(kucuk_model_uret, kucuk_tokenizer, ornek_veri, tmp_path):
    veri, _ = ornek_veri
    for stilo in (True, False):
        model = kucuk_model_uret(stilo).eval()
        model.kaydet(tmp_path / str(stilo), ek_bilgi={"sicaklik": 1.2})
        yuklenen, bilgi = HybridPolarModel.kayittan(tmp_path / str(stilo))
        assert bilgi["stilo_kullan"] is stilo and bilgi["sicaklik"] == 1.2
        assert np.allclose(tahmin_et(model, kucuk_tokenizer, veri), tahmin_et(yuklenen, kucuk_tokenizer, veri), atol=1e-6)
    assert not list(tmp_path.rglob("*.pkl")) and not list(tmp_path.rglob("*.bin"))   # pickle yok


def test_metrikler_ve_sicaklik():
    m = metrikler([0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 2, 2])
    assert m["accuracy"] == 5 / 6
    assert m["karisiklik_matrisi"] == [[2, 0, 0], [0, 1, 1], [0, 0, 2]]
    assert m["sinif"]["1"]["recall"] == 0.5
    eksik = metrikler([0, 0], [0, 0])            # testte olmayan sınıflar hata vermemeli
    assert len(eksik["karisiklik_matrisi"]) == N_SINIF
    asiri_emin = np.array([[9.0, 0, 0], [9.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]])
    assert sicaklik_bul(asiri_emin, [0, 1, 1, 2]) > 1.0          # yanlışken çok emin → yumuşat
