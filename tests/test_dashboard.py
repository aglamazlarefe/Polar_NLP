"""Panelin dört sayfası, model varken ve yokken, istisnasız çalışmalı."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from polar_nlp.config import SINIF_ADLARI

DASHBOARD = str(Path(__file__).resolve().parent.parent / "dashboard.py")
SAYFALAR = ["🔍 Tekil Analiz", "📊 Toplu Analiz", "📈 Model Sonuçları", "ℹ️ Sistem Hakkında"]


def uygulama(monkeypatch, model_dizini, sonuc_dizini) -> AppTest:
    monkeypatch.setenv("POLAR_MODEL_DIZINI", str(model_dizini))
    monkeypatch.setenv("POLAR_SONUC_DIZINI", str(sonuc_dizini))
    return AppTest.from_file(DASHBOARD, default_timeout=120)


def sayfaya_git(at: AppTest, sayfa: str) -> AppTest:
    at.run()
    at.sidebar.radio[0].set_value(sayfa).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def dugme(at: AppTest, etiket: str):
    return next(b for b in at.button if b.label == etiket)


def sahte_metrik() -> dict:
    def model(f1):
        return {"katlar": [{"macro_f1": f1, "accuracy": f1, "n_test": 10}] * 2, "macro_f1_ort": f1,
                "macro_f1_std": 0.01, "accuracy_ort": f1, "accuracy_std": 0.01,
                "alt_kumeler": {"zor": {"n": 5, "accuracy": 0.6, "macro_f1": 0.6}}}

    def deney(aciklama):
        return {"aciklama": aciklama, "n_test": 20, "modeller": {m: model(f) for m, f in [
            ("tfidf_lr", 0.7), ("stilometri_lr", 0.6), ("yalniz_bert", 0.8), ("hibrit", 0.82)]},
            "mcnemar": {"hibrit_vs_yalniz_bert": {"p": 0.4}}}

    return {"meta": {"tarih": "2026-01-01T00:00:00", "commit": "abc1234", "n_kat": 2, "cihaz": "cpu"},
            "siniflar": SINIF_ADLARI,
            "veri": {"n": 40, "etiket_denetimi": {"cohen_kappa": 0.9, "etiket_uyusmazligindan_atilan": ["x"]}},
            "deneyler": {"A_v1_cv": deney("A"), "C_v1_v2": deney("C"), "B_tum_cv": deney("B")}}


@pytest.mark.parametrize("sayfa", SAYFALAR)
def test_sayfalar_model_yokken_calisir(monkeypatch, tmp_path, sayfa):
    at = sayfaya_git(uygulama(monkeypatch, tmp_path / "yok", tmp_path / "yok"), sayfa)
    assert any("Model bulunamadı" in e.value for e in at.sidebar.error)
    if sayfa in SAYFALAR[:2]:
        assert any("egit_final" in w.value for w in at.warning)


@pytest.mark.parametrize("sayfa", SAYFALAR)
def test_sayfalar_model_varken_calisir(monkeypatch, kucuk_model_dizini, tmp_path, sayfa):
    (tmp_path / "metrics.json").write_text(json.dumps(sahte_metrik()), encoding="utf-8")
    at = sayfaya_git(uygulama(monkeypatch, kucuk_model_dizini, tmp_path), sayfa)
    assert any("Model hazır" in s.value for s in at.sidebar.success)


def test_tekil_analiz_akisi(monkeypatch, kucuk_model_dizini, tmp_path):
    at = sayfaya_git(uygulama(monkeypatch, kucuk_model_dizini, tmp_path), SAYFALAR[0])
    dugme(at, "Analizi Başlat").click().run()
    assert any("metin girin" in e.value for e in at.error)          # boş metin
    dugme(at, "Örnek 3").click().run()
    assert "reddediyorum" in at.text_area[0].value
    dugme(at, "Analizi Başlat").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label.startswith("Risk skoru") for m in at.metric)
    assert len(at.dataframe) == 1                                    # stilometri tablosu
    at.run()                                                         # yeniden çizimde sonuç kaybolmamalı
    assert any(m.label.startswith("Risk skoru") for m in at.metric)


def test_toplu_analiz_demo_akisi(monkeypatch, kucuk_model_dizini, tmp_path):
    at = sayfaya_git(uygulama(monkeypatch, kucuk_model_dizini, tmp_path), SAYFALAR[1])
    dugme(at, "Demo verisini kullan").click().run()
    dugme(at, "Toplu Analizi Başlat").click().run()
    assert not at.exception, [e.value for e in at.exception]
    etiketler = [m.label for m in at.metric]
    assert etiketler[:3] == SINIF_ADLARI and "Ortalama risk" in etiketler
    assert sum(int(m.value) for m in at.metric[:3]) == 42            # 3 personel × 14 gün
    at.run()                                                         # eski hata: yeniden çizimde sonuç siliniyordu
    assert "Ortalama risk" in [m.label for m in at.metric]


def test_model_sonuclari_sayilari_dosyadan_okur(monkeypatch, kucuk_model_dizini, tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps(sahte_metrik()), encoding="utf-8")
    at = sayfaya_git(uygulama(monkeypatch, kucuk_model_dizini, tmp_path), SAYFALAR[2])
    metin = " ".join(m.value for m in at.markdown)
    assert "0.820" in metin and "0.800" in metin                    # sahte dosyadaki değerler
    assert any(m.label.startswith("Kör denetçi") and m.value == "0.90" for m in at.metric)
