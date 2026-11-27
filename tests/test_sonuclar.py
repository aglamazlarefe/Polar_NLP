"""Raporlanan sonuçların gerçek çalıştırmadan geldiğini güvenceye alır."""

import json
import re
from pathlib import Path

import pytest

from polar_nlp.config import DEMO_TREND_YOLU, KOK, METRIK_YOLU, N_SINIF, SONUC_DIZINI

MODELLER = {"tfidf_lr", "stilometri_lr", "yalniz_bert", "hibrit"}


@pytest.fixture(scope="module")
def metrik():
    if not METRIK_YOLU.exists():
        pytest.fail("results/metrics.json yok — `python scripts/deney.py` çalıştırın")
    return json.loads(METRIK_YOLU.read_text(encoding="utf-8"))


def test_metrik_semasi(metrik):
    assert {"meta", "veri", "deneyler", "final"} <= set(metrik)
    assert metrik["meta"]["hizli_mod"] is False, "metrics.json duman testi çıktısı olamaz"
    assert {"A_v1_cv", "B_tum_cv", "C_v1_v2", "D_v2_v1"} <= set(metrik["deneyler"])
    for ad, deney in metrik["deneyler"].items():
        assert set(deney["modeller"]) == MODELLER, ad
        for model, o in deney["modeller"].items():
            assert 0.0 <= o["macro_f1_ort"] <= 1.0
            cm = o["havuz"]["karisiklik_matrisi"]
            assert len(cm) == N_SINIF and sum(map(sum, cm)) == deney["n_test"], (ad, model)


def test_capraz_dogrulama_her_ornegi_bir_kez_test_eder(metrik):
    veri = metrik["veri"]
    assert metrik["deneyler"]["B_tum_cv"]["n_test"] == veri["n"]
    assert metrik["deneyler"]["A_v1_cv"]["n_test"] == veri["kaynak_dagilimi"]["v1"]
    assert metrik["deneyler"]["C_v1_v2"]["n_test"] == veri["kaynak_dagilimi"]["v2"]
    assert len(metrik["deneyler"]["B_tum_cv"]["modeller"]["hibrit"]["katlar"]) == metrik["meta"]["n_kat"]


def test_final_model_kaydi(metrik):
    final = metrik["final"]
    assert len(final["gecmis"]) >= final["en_iyi_epoch"] >= 1
    assert final["n_egitim"] + final["n_dogrulama"] == metrik["veri"]["n"]


def test_demo_trend_gercek_cikti():
    demo = json.loads(DEMO_TREND_YOLU.read_text(encoding="utf-8"))
    assert len(demo["kayitlar"]) == 42
    assert all(0.0 <= k["risk_skoru"] <= 1.0 and k["sinif"] in range(N_SINIF) for k in demo["kayitlar"])


def test_grafikler_uretilmis():
    for ad in ["confusion_matrix.png", "model_karsilastirma.png", "egitim_egrisi.png",
               "stilometri_karsilastirma.png", "ozellik_radar.png", "demo_trend.png"]:
        yol = SONUC_DIZINI / ad
        assert yol.exists() and yol.stat().st_size > 10_000, ad
        assert yol.stat().st_mtime >= METRIK_YOLU.stat().st_mtime - 5, f"{ad} metrics.json'dan eski — gorseller.py çalıştırın"


@pytest.mark.parametrize("dosya", ["scripts/gorseller.py", "dashboard.py"])
def test_kaynakta_elle_yazilmis_sonuc_yok(dosya):
    """Eski kodda grafik verileri ve "%100 doğruluk" gibi iddialar kaynağa gömülüydü."""
    kaynak = (KOK / dosya).read_text(encoding="utf-8")
    assert not re.search(r"%\s?100|Accuracy.{0,40}1\.00|600 sentetik", kaynak)
    # 0.xxxx biçiminde dörtten fazla ondalık sayı içeren liste = gömülü ölçüm verisi
    assert not re.search(r"\[\s*(?:0\.\d{3,}\s*,\s*){3,}", kaynak), "gömülü sonuç dizisi bulundu"
    assert "KVKK ve GDPR uyumludur" not in kaynak
