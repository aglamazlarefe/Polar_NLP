import pandas as pd
import pytest

from polar_nlp.config import DEMO_YOLU, N_SINIF, V1_YOLU, VERI_YOLU
from polar_nlp.veri import PolarDataset, dogrulama_ayir, harmanlayici, katlar, veri_yukle


def csv_yaz(tmp_path, satirlar, sutunlar=("Metin", "Etiket")):
    yol = tmp_path / "veri.csv"
    pd.DataFrame(satirlar, columns=list(sutunlar)).to_csv(yol, index=False)
    return yol


def test_v1_ozgun_veri_degismedi():
    df = veri_yukle(V1_YOLU)
    assert len(df) == 300
    assert df["Etiket"].value_counts().to_dict() == {0: 102, 1: 99, 2: 99}


def test_ana_veri_seti_semasi():
    df = veri_yukle(VERI_YOLU)
    assert {"Metin", "Etiket", "Konu", "Kaynak", "Zorluk"} <= set(df.columns)
    assert set(df["Etiket"]) == set(range(N_SINIF))
    assert set(df["Kaynak"]) <= {"v1", "v2"} and set(df["Zorluk"]) <= {"normal", "zor"}
    assert (df["Kaynak"] == "v1").sum() == 300          # v1 eksiksiz içinde
    assert df["Etiket"].value_counts().min() / df["Etiket"].value_counts().max() > 0.8   # dengeli
    assert "bilinmiyor" not in set(df["Konu"])


def test_demo_gunlukleri_egitim_verisinde_yok():
    demo = pd.read_csv(DEMO_YOLU)
    assert {"Personel", "Tarih", "Metin"} <= set(demo.columns)
    assert not set(demo["Metin"]) & set(veri_yukle(VERI_YOLU)["Metin"])


@pytest.mark.parametrize("satirlar, mesaj", [
    ([("a b", 0), ("a b", 1)], "yinelenen"),
    ([("a b", 0), ("c d", 5)], "aralığında"),
    ([("a b", 0), ("   ", 1)], "boş"),
])
def test_veri_yukle_bozuk_veriyi_reddeder(tmp_path, satirlar, mesaj):
    with pytest.raises(ValueError, match=mesaj):
        veri_yukle(csv_yaz(tmp_path, satirlar))


def test_veri_yukle_eksik_sutun(tmp_path):
    with pytest.raises(ValueError, match="eksik sütun"):
        veri_yukle(csv_yaz(tmp_path, [("a", 0)], sutunlar=("Yazi", "Etiket")))


def test_veri_yukle_varsayilan_sutunlari_ekler(tmp_path):
    df = veri_yukle(csv_yaz(tmp_path, [("a b", 0), ("c d", 1)]))
    assert df["Kaynak"].tolist() == ["v1", "v1"] and df["Zorluk"].tolist() == ["normal", "normal"]


def test_katlar_konu_sizdirmaz_ve_her_ornegi_bir_kez_test_eder():
    df = veri_yukle(VERI_YOLU)
    gorulen = []
    for egitim, test in katlar(df, 5):
        assert not set(df["Konu"].iloc[egitim]) & set(df["Konu"].iloc[test])
        assert set(df["Etiket"].iloc[test]) == set(range(N_SINIF))
        gorulen.extend(test)
    assert sorted(gorulen) == list(range(len(df)))


def test_dogrulama_ayir_ayrik_ve_katmanli():
    df = veri_yukle(V1_YOLU)
    y = df["Etiket"].to_numpy()
    egitim, dogrulama = dogrulama_ayir(df.index.to_numpy(), y)
    assert not set(egitim) & set(dogrulama) and len(egitim) + len(dogrulama) == len(df)
    assert set(y[dogrulama]) == set(range(N_SINIF))


def test_harmanlayici_dinamik_doldurur(kucuk_tokenizer, ornek_veri):
    veri, olcek = ornek_veri
    parti = harmanlayici(kucuk_tokenizer)([veri[0], veri[3]])
    assert parti["input_ids"].shape == parti["attention_mask"].shape
    assert parti["input_ids"].shape[1] < 128                      # sabit 128'e doldurulmuyor
    assert parti["stilometri"].shape[0] == 2 and parti["etiket"].tolist() == [0, 1]
    etiketsiz = PolarDataset(["Deneme metni."], None, olcek)
    assert etiketsiz[0][2] == -1
