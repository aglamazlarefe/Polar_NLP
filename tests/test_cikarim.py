import pytest

from polar_nlp.cikarim import PolarAnalizci, model_var_mi, polar_analiz_et
from polar_nlp.config import SINIF_ADLARI
from polar_nlp.stilometri import OZELLIK_ADLARI

METIN = "Günlük form doldurulmayacak. Bu işin kimseye faydası yok, uğraşmıyorum."


@pytest.fixture(scope="module")
def analizci(kucuk_model_dizini):
    return PolarAnalizci(kucuk_model_dizini)


def test_cikti_sozlesmesi(analizci):
    s = analizci.analiz_et([METIN])[0]
    assert set(s) == {"sinif", "sinif_adi", "guven_skoru", "olasiliklar", "risk_skoru", "belirsiz", "stilometri"}
    assert s["sinif_adi"] == SINIF_ADLARI[s["sinif"]]
    assert list(s["olasiliklar"]) == SINIF_ADLARI
    assert sum(s["olasiliklar"].values()) == pytest.approx(1.0, abs=1e-3)
    assert s["guven_skoru"] == max(s["olasiliklar"].values())
    assert 0.0 <= s["risk_skoru"] <= 1.0
    assert isinstance(s["belirsiz"], bool)
    assert list(s["stilometri"]) == OZELLIK_ADLARI
    assert s["stilometri"]["olumsuzluk"]["deger"] > 0


def test_risk_skoru_olasiliklardan_hesaplanir(analizci):
    s = analizci.analiz_et([METIN])[0]
    p = list(s["olasiliklar"].values())
    assert s["risk_skoru"] == pytest.approx(0.5 * p[1] + 1.0 * p[2], abs=1e-3)


def test_toplu_ve_tekil_ayni(analizci):
    metinler = [METIN, "Bakım tamamlandı.", "Yine gecikti, işlerime vakit kalmıyor. Çok yoruldum."]
    toplu = analizci.analiz_et(metinler, batch=2)
    for metin, t in zip(metinler, toplu):
        tek = analizci.analiz_et([metin])[0]
        assert tek["sinif"] == t["sinif"]
        assert tek["risk_skoru"] == pytest.approx(t["risk_skoru"], abs=2e-3)   # doldurma farkı


def test_uc_girdiler(analizci):
    assert analizci.analiz_et([]) == []
    for metin in ["", "a", "kelime " * 400]:
        s = analizci.analiz_et([metin])[0]
        assert sum(s["olasiliklar"].values()) == pytest.approx(1.0, abs=1e-3)


def test_kelime_katkilari(analizci):
    katkilar = analizci.kelime_katkilari(METIN)
    assert [k for k, _ in katkilar] == METIN.split()
    assert all(-1.0 <= v <= 1.0 for _, v in katkilar)
    assert analizci.kelime_katkilari("tek") == [("tek", 0.0)]


def test_sicaklik_model_bilgisinden_okunur(analizci):
    assert analizci.sicaklik == 1.5


def test_model_var_mi_ve_eksik_model(kucuk_model_dizini, tmp_path):
    assert model_var_mi(kucuk_model_dizini)
    assert not model_var_mi(tmp_path)
    with pytest.raises(FileNotFoundError, match="egit_final"):
        PolarAnalizci(tmp_path)


def test_polar_analiz_et_kisayolu(kucuk_model_dizini):
    assert polar_analiz_et(METIN, kucuk_model_dizini)["sinif"] in (0, 1, 2)
