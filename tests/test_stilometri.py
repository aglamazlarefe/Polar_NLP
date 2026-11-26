import numpy as np
import pytest

from polar_nlp.stilometri import (N_OZELLIK, OZELLIK_ADLARI, OZELLIK_ETIKETLERI, Olcekleyici, cumlelere_ayir,
                                  kelimelere_ayir, stilometri_cikar, stilometri_matrisi, tr_kucult)


def ozellik(metin: str, ad: str) -> float:
    return float(stilometri_cikar(metin)[OZELLIK_ADLARI.index(ad)])


def test_tr_kucult_turkce_harfleri_dogru_cevirir():
    assert tr_kucult("ISITICI İSTASYON") == "ısıtıcı istasyon"
    assert "ISITICI".lower() != "ısıtıcı"          # standart lower() neden yetmiyor


def test_kelimelere_ayir_sayilari_atar_kesmeyi_korur():
    assert kelimelere_ayir("GPS'i 3 kez, 08:00'de açtım.") == ["gps'i", "kez", "de", "açtım"]


def test_cumlelere_ayir_ondalik_sayiyi_bolmez():
    assert len(cumlelere_ayir("Basınç 3.5 bar ölçüldü. Sistem çalışıyor! Sorun yok mu?")) == 3


@pytest.mark.parametrize("metin", ["Kontrol ediyorum.", "Vanayı kapattım.", "Yarın bakacağım.", "Bana haber gelmedi.",
                                   "Talep ettiğim parça yok.", "İşlerimi bitiremedim.", "Kayıt tarafımca tutuldu."])
def test_birinci_tekil_yakalanir(metin):
    assert ozellik(metin, "birinci_tekil") > 0


@pytest.mark.parametrize("metin", ["Eğitim ve yardım planlandı.", "Yalıtım paneli söküldü.", "Tüm sistem çalışıyor.",
                                   "Durum raporu iletildi.", "Hortum değiştirildi."])
def test_birinci_tekil_yanlis_pozitif_vermez(metin):
    assert ozellik(metin, "birinci_tekil") == 0


@pytest.mark.parametrize("metin", ["Sinyal gelmiyor.", "Bakım yapılmadı.", "Ölçüm alınmayacak.", "Cihaz çalışmaz.",
                                   "Yedek parça yok.", "Bu yeterli değil.", "Isıtıcı yetersiz.", "Beklemeden çıktık."])
def test_olumsuzluk_yakalanir(metin):
    assert ozellik(metin, "olumsuzluk") > 0


@pytest.mark.parametrize("metin", ["Telsiz bataryası şarj edildi.", "Maden numunesi alındı.", "Bakım tamamlandı."])
def test_olumsuzluk_yanlis_pozitif_vermez(metin):
    assert ozellik(metin, "olumsuzluk") == 0


def test_resmi_bildirme_ve_edilgen():
    assert ozellik("Bakım tamamlanmıştır.", "resmi_bildirme") > 0
    assert ozellik("Prosedür uygulandı.", "resmi_bildirme") == 0
    for metin in ["Filtre değiştirildi.", "Numune alındı.", "Panel temizlendi.", "Kayıt yapılmıştır."]:
        assert ozellik(metin, "edilgen") > 0, metin
    for metin in ["Cihazı kullandım.", "Depoda yedek bulunuyor.", "Havalandırma çalışıyor."]:
        assert ozellik(metin, "edilgen") == 0, metin


def test_mutlakci_ve_nedensellik():
    assert ozellik("Her zaman aynı sorun, asla çözülmüyor.", "mutlakci") == pytest.approx(2 / 6)
    assert ozellik("Fırtına yüzünden çıkamadık.", "nedensellik") > 0
    assert ozellik("Parça gelmediğinden bekliyoruz.", "nedensellik") > 0


def test_sayisal_ozellikler():
    v = dict(zip(OZELLIK_ADLARI, stilometri_cikar("Bakım yapıldı. Sistem çalışıyor ve sistem kararlı.")))
    assert v["cumle_sayisi"] == 2
    assert v["ort_cumle_uzunlugu"] == pytest.approx(3.5)
    assert v["ttr"] == pytest.approx(6 / 7)


@pytest.mark.parametrize("metin", ["", "   ", "Tamam", "123 456", None])
def test_uc_durumlar_hata_vermez(metin):
    v = stilometri_cikar(metin)
    assert v.shape == (N_OZELLIK,) and np.isfinite(v).all()


def test_ozellik_listesi_tutarli():
    assert len(OZELLIK_ADLARI) == len(set(OZELLIK_ADLARI)) == N_OZELLIK
    assert set(OZELLIK_ETIKETLERI) == set(OZELLIK_ADLARI)
    assert stilometri_matrisi(["a b.", "c d."]).shape == (2, N_OZELLIK)
    assert stilometri_matrisi([]).shape == (0, N_OZELLIK)


def test_olcekleyici_json_gidis_donus_ve_kirpma():
    ham = stilometri_matrisi(["Bakım yapıldı.", "Sinyal gelmiyor, uğraşmıyorum.", "Kontrol ettim ve kaydettim."])
    olcek = Olcekleyici().uydur(ham, [0, 2, 0])
    kopya = Olcekleyici.sozlukten(olcek.sozluk())
    assert np.allclose(olcek.donustur(ham), kopya.donustur(ham))
    assert np.abs(olcek.donustur(ham * 1000)).max() <= Olcekleyici.KIRPMA
    assert np.isfinite(olcek.objektife_gore_z(ham)).all()      # sabit sütunlar sıfıra bölmez


def test_olcekleyici_ozellik_listesi_degisirse_reddeder():
    olcek = Olcekleyici().uydur(stilometri_matrisi(["a b.", "c d e."]), [0, 1])
    bozuk = {**olcek.sozluk(), "ozellikler": ["eski"]}
    with pytest.raises(ValueError):
        Olcekleyici.sozlukten(bozuk)
