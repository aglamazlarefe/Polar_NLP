"""Polar NLP paneli.

    streamlit run dashboard.py

KURAL: Bu dosyada elle yazılmış performans sayısı bulunmaz. Model başarımı
results/metrics.json'dan, model bilgisi egitilmis_hybrid_model/model_info.json'dan okunur.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from polar_nlp.cikarim import PolarAnalizci, model_var_mi
from polar_nlp.config import (BELIRSIZLIK_ESIGI, DEMO_YOLU, MODEL_DIZINI, SINIF_ADLARI, SINIF_RENKLERI,
                              SONUC_DIZINI)
from polar_nlp.stilometri import OZELLIK_ETIKETLERI

# Testler ve farklı kurulumlar için ortam değişkeniyle değiştirilebilir.
MODEL_YOLU = Path(os.environ.get("POLAR_MODEL_DIZINI") or MODEL_DIZINI)
SONUC_YOLU = Path(os.environ.get("POLAR_SONUC_DIZINI") or SONUC_DIZINI)

SAYFALAR = ["🔍 Tekil Analiz", "📊 Toplu Analiz", "📈 Model Sonuçları", "ℹ️ Sistem Hakkında"]
ORNEKLER = {
    "Örnek 1": "Jeneratör manifold bakımı standart prosedüre göre tamamlanmıştır. Voltaj dalgalanması gözlenmedi. "
               "Yakıt pompası basıncı nominal değerlerde seyrediyor.",
    "Örnek 2": "VHF telsiz bataryalarının kalibrasyon süreci yine beklenenden fazla sürdü. Eski nesil şarj "
               "üniteleriyle bu işlemi yapmak günlük rutinleri gereksiz yere yavaşlatıyor. Albedo ölçümüne zaman kalmadı.",
    "Örnek 3": "Telsiz anteninin günlük kar kürüme işi anlamsız bir rutindir. Cihaz zaten sinyal almıyor. "
               "Kürek çekmek için dışarı çıkmayı reddediyorum.",
}
ONERILER = [
    "Kayıt nötr bir rapor niteliğinde. Olağan takip yeterlidir.",
    "Kayıtta yük ve zorlanma ifadeleri var. İş yükü ve dinlenme düzeni gözden geçirilebilir.",
    "Kayıtta görevden ve ortamdan uzaklaşma ifadeleri var. Yüz yüze bir görüşme değerlendirilebilir.",
]
MODEL_ETIKETLERI = {"tfidf_lr": "TF-IDF + Lojistik Regresyon", "stilometri_lr": "Yalnız stilometri",
                    "yalniz_bert": "Yalnız BERTurk", "hibrit": "Hibrit (BERTurk + stilometri)"}

st.set_page_config(page_title="Polar NLP", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="st-"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.ust-cizgi { height: 2px; background: linear-gradient(90deg, #4fc3f7, #00e5ff); margin-bottom: 22px; }
.rozet { padding: 4px 12px; border-radius: 12px; font-weight: 700; color: #0a1628; display: inline-block; }
.kart { background: #16243a; padding: 14px 18px; border-radius: 10px; border-left: 5px solid #4fc3f7; margin-bottom: 10px; }
.katki { line-height: 2.1; font-size: 1.02rem; }
.katki span { padding: 2px 4px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- yardımcılar


@st.cache_resource(show_spinner="Model yükleniyor…")
def analizci_al(yol: str) -> PolarAnalizci:
    return PolarAnalizci(yol)


@st.cache_data
def json_oku(yol: str, degisim: float) -> dict:
    return json.loads(Path(yol).read_text(encoding="utf-8"))


def json_yukle(yol: Path) -> dict | None:
    return json_oku(str(yol), yol.stat().st_mtime) if yol.exists() else None


def rozet(sinif: int) -> str:
    return f"<span class='rozet' style='background:{SINIF_RENKLERI[sinif]}'>{html.escape(SINIF_ADLARI[sinif])}</span>"


def kart(icerik_html: str, renk: str = "#4fc3f7") -> None:
    st.markdown(f"<div class='kart' style='border-left-color:{renk}'>{icerik_html}</div>", unsafe_allow_html=True)


def yuzde(x: float) -> str:
    return f"%{x * 100:.1f}"


def model_yok_uyarisi() -> None:
    st.warning(f"Eğitilmiş model bulunamadı (`{MODEL_YOLU.name}/`). Önce `python scripts/egit_final.py` çalıştırın. "
               "Bu sayfa model olmadan çalışmaz.")


def csv_oku(dosya) -> pd.DataFrame:
    for kodlama in ("utf-8-sig", "cp1254"):
        try:
            dosya.seek(0)
            return pd.read_csv(dosya, encoding=kodlama)
        except UnicodeDecodeError:
            continue
    raise ValueError("Dosya kodlaması okunamadı (UTF-8 bekleniyor).")


MODEL_HAZIR = model_var_mi(MODEL_YOLU)

# ---------------------------------------------------------------- kenar çubuğu

with st.sidebar:
    st.markdown("<h1 style='text-align:center'>🧊 POLAR NLP</h1>", unsafe_allow_html=True)
    st.divider()
    sayfa = st.radio("Navigasyon", SAYFALAR, label_visibility="collapsed", key="sayfa")
    st.divider()
    st.markdown("### Durum")
    if MODEL_HAZIR:
        st.success("✅ Model hazır")
    else:
        st.error("❌ Model bulunamadı")
    st.caption("Prototip · sentetik veriyle eğitildi · tıbbi tanı aracı değildir")

st.markdown("<div class='ust-cizgi'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- sayfa 1: tekil analiz


def ornek_sec(metin: str) -> None:
    st.session_state["tekil_metin"] = metin
    st.session_state.pop("tekil_sonuc", None)


def tekil_sonuc_goster(kayit: dict) -> None:
    sonuc = kayit["sonuc"]
    st.divider()
    sol, sag = st.columns([2, 1])
    with sol:
        st.markdown(f"### Tespit edilen üslup: {rozet(sonuc['sinif'])}", unsafe_allow_html=True)
        if sonuc["belirsiz"]:
            st.warning(f"Model bu metinde kararsız (en yüksek olasılık {yuzde(sonuc['guven_skoru'])}, "
                       f"eşik {yuzde(BELIRSIZLIK_ESIGI)}). Sonuç insan değerlendirmesi olmadan kullanılmamalı.")
        for ad, p in sonuc["olasiliklar"].items():
            st.progress(min(max(p, 0.0), 1.0), text=f"{ad} — {yuzde(p)}")
    with sag:
        st.metric("Risk skoru (0–1)", f"{sonuc['risk_skoru']:.2f}",
                  help="0.5 × P(Bilişsel Yorgunluk) + 1.0 × P(Kış Sendromu)")
        kart(f"<b>Takip önerisi</b><br>{html.escape(ONERILER[sonuc['sinif']])}", SINIF_RENKLERI[sonuc["sinif"]])

    st.markdown("### Stilometrik göstergeler")
    st.caption("z-skoru: değerin, eğitim verisindeki objektif raporların ortalamasından kaç standart sapma uzakta olduğu. "
               "Bu göstergeler kural tabanlı sayımlardır; kararı tek başına belirlemezler.")
    tablo = pd.DataFrame([{"Gösterge": OZELLIK_ETIKETLERI.get(ad, ad), "Değer": d["deger"], "Objektife göre z": d["z"]}
                          for ad, d in sonuc["stilometri"].items()])
    st.dataframe(tablo, hide_index=True, width="stretch")

    st.markdown("### Hangi kelimeler etkili oldu?")
    st.caption("Her kelime tek tek çıkarılıp tahmin yinelenir. Renkli: kelime bu sınıf kararını destekliyor; "
               "gri: karşı yönde. Yaklaşık bir açıklamadır.")
    katkilar = kayit.get("katkilar")
    if katkilar is None:
        st.info("Kelime katkıları bu metin için hesaplanmadı.")
        return
    en_buyuk = max((abs(k) for _, k in katkilar), default=0.0) or 1.0
    renk = SINIF_RENKLERI[sonuc["sinif"]].lstrip("#")
    r, g, b = (int(renk[i:i + 2], 16) for i in (0, 2, 4))
    parcalar = []
    for kelime, katki in katkilar:
        siddet = min(abs(katki) / en_buyuk, 1.0) * 0.85
        zemin = f"rgba({r},{g},{b},{siddet:.2f})" if katki > 0 else f"rgba(144,164,174,{siddet:.2f})"
        parcalar.append(f"<span style='background:{zemin}' title='{katki:+.3f}'>{html.escape(kelime)}</span>")
    st.markdown(f"<div class='katki'>{' '.join(parcalar)}</div>", unsafe_allow_html=True)


if sayfa == SAYFALAR[0]:
    st.title("Tekil Metin Analizi")
    st.write("Bir seyir defteri kaydını girin; model yazım üslubunu üç sınıftan birine atar.")
    if not MODEL_HAZIR:
        model_yok_uyarisi()
    else:
        st.session_state.setdefault("tekil_metin", "")
        st.caption("Veri setinden örnekler:")
        for sutun, (ad, metin) in zip(st.columns(len(ORNEKLER)), ORNEKLER.items()):
            sutun.button(ad, on_click=ornek_sec, args=(metin,), width="stretch")
        metin = st.text_area("Analiz edilecek metin", height=150, key="tekil_metin",
                             placeholder="Metni buraya yazın ya da bir örnek seçin…")
        if st.button("Analizi Başlat", type="primary"):
            if not metin.strip():
                st.error("Lütfen bir metin girin.")
            else:
                analizci = analizci_al(str(MODEL_YOLU))
                with st.spinner("Metin inceleniyor…"):
                    st.session_state["tekil_sonuc"] = {
                        "metin": metin,
                        "sonuc": analizci.analiz_et([metin])[0],
                        "katkilar": analizci.kelime_katkilari(metin),
                    }
        kayit = st.session_state.get("tekil_sonuc")
        if kayit and kayit.get("metin") == metin:      # metin değiştiyse eski sonucu gösterme
            tekil_sonuc_goster(kayit)


# ---------------------------------------------------------------- sayfa 2: toplu analiz


def toplu_calistir(df: pd.DataFrame) -> pd.DataFrame:
    analizci = analizci_al(str(MODEL_YOLU))
    metinler = df["Metin"].tolist()
    cubuk = st.progress(0.0, text="İşleniyor…")
    sonuclar, adim = [], 32
    for i in range(0, len(metinler), adim):
        sonuclar.extend(analizci.analiz_et(metinler[i:i + adim]))
        cubuk.progress(min((i + adim) / len(metinler), 1.0), text=f"İşleniyor: {min(i + adim, len(metinler))}/{len(metinler)}")
    cubuk.empty()
    cikti = df.reset_index(drop=True).copy()
    cikti["Sınıf"] = [s["sinif_adi"] for s in sonuclar]
    cikti["Sınıf No"] = [s["sinif"] for s in sonuclar]
    cikti["Güven"] = [s["guven_skoru"] for s in sonuclar]
    cikti["Risk Skoru"] = [s["risk_skoru"] for s in sonuclar]
    cikti["Belirsiz"] = [s["belirsiz"] for s in sonuclar]
    return cikti


def toplu_sonuc_goster(sonuc: pd.DataFrame) -> None:
    st.markdown("### Sonuçlar")
    gosterilecek = [s for s in ["Personel", "Tarih", "Metin", "Sınıf", "Güven", "Risk Skoru", "Belirsiz"] if s in sonuc.columns]
    st.dataframe(sonuc[gosterilecek], hide_index=True, width="stretch",
                 column_config={"Güven": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                "Risk Skoru": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

    st.markdown("### Özet")
    sutunlar = st.columns(len(SINIF_ADLARI) + 2)
    for i, ad in enumerate(SINIF_ADLARI):
        sutunlar[i].metric(ad, int((sonuc["Sınıf No"] == i).sum()))
    sutunlar[-2].metric("Ortalama risk", f"{sonuc['Risk Skoru'].mean():.2f}")
    sutunlar[-1].metric("Belirsiz kayıt", int(sonuc["Belirsiz"].sum()))

    if {"Personel", "Tarih"} <= set(sonuc.columns):
        st.markdown("### Kişi bazlı risk seyri")
        seyir = sonuc.assign(Tarih=pd.to_datetime(sonuc["Tarih"], errors="coerce")).dropna(subset=["Tarih"])
        if seyir.empty:
            st.info("Tarih sütunu okunamadı (beklenen biçim: YYYY-AA-GG).")
        else:
            pivot = seyir.pivot_table(index="Tarih", columns="Personel", values="Risk Skoru", aggfunc="mean")
            st.line_chart(pivot, y_label="Risk skoru", x_label="Tarih")
            st.caption("Risk skoru tek kayıtta gürültülüdür; tek güne değil birkaç günlük eğilime bakın.")

    st.markdown("### En yüksek riskli kayıtlar")
    for _, s in sonuc.sort_values("Risk Skoru", ascending=False).head(5).iterrows():
        kim = f"{html.escape(str(s['Personel']))} · " if "Personel" in sonuc.columns else ""
        tarih = f"{html.escape(str(s['Tarih']))} · " if "Tarih" in sonuc.columns else ""
        kart(f"{kim}{tarih}risk <b>{s['Risk Skoru']:.2f}</b> · {rozet(int(s['Sınıf No']))}<br>"
             f"<span style='color:#b0bec5'>{html.escape(str(s['Metin']))}</span>", SINIF_RENKLERI[int(s["Sınıf No"])])

    st.download_button("Sonuçları CSV olarak indir", sonuc.to_csv(index=False).encode("utf-8-sig"),
                       file_name="polar_analiz_sonuclari.csv", mime="text/csv")


def demo_yukle() -> None:
    st.session_state["toplu_girdi"] = pd.read_csv(DEMO_YOLU).drop(columns=["Beklenen"], errors="ignore")
    st.session_state["toplu_kaynak"] = "Kurgusal demo günlükleri (3 personel × 14 gün)"
    st.session_state.pop("toplu_sonuc", None)
    st.session_state.pop("toplu_imza", None)      # aynı dosya yeniden seçilirse tekrar okunsun


if sayfa == SAYFALAR[1]:
    st.title("Toplu Analiz")
    st.write("Birden çok kaydı CSV olarak yükleyin. Zorunlu sütun: `Metin`. İsteğe bağlı `Personel` ve `Tarih` "
             "sütunları varsa kişi bazlı risk seyri de çizilir.")
    if not MODEL_HAZIR:
        model_yok_uyarisi()
    else:
        sol, sag = st.columns([3, 1])
        dosya = sol.file_uploader("CSV dosyası", type=["csv"])
        if DEMO_YOLU.exists():
            sag.write("")
            sag.button("Demo verisini kullan", on_click=demo_yukle, width="stretch",
                       help="Kurgusal 14 günlük günlükler. Model eğitiminde kullanılmadı.")
        if dosya is not None:
            imza = (dosya.name, hashlib.md5(dosya.getvalue()).hexdigest())
            if st.session_state.get("toplu_imza") != imza:
                try:
                    st.session_state["toplu_girdi"] = csv_oku(dosya)
                    st.session_state["toplu_kaynak"] = dosya.name
                except Exception as hata:   # bozuk dosya kullanıcıya açıkça bildirilir
                    st.session_state.pop("toplu_girdi", None)
                    st.error(f"CSV okunamadı: {hata}")
                st.session_state["toplu_imza"] = imza
                st.session_state.pop("toplu_sonuc", None)

        girdi = st.session_state.get("toplu_girdi")
        if girdi is not None:
            if "Metin" not in girdi.columns:
                st.error("CSV dosyasında `Metin` başlıklı bir sütun bulunmalıdır.")
            else:
                gecerli = girdi[girdi["Metin"].notna() & (girdi["Metin"].astype(str).str.strip() != "")].copy()
                gecerli["Metin"] = gecerli["Metin"].astype(str).str.strip()
                st.caption(f"Kaynak: {st.session_state.get('toplu_kaynak', '')} — {len(gecerli)} kayıt"
                           + (f" ({len(girdi) - len(gecerli)} boş satır atlandı)" if len(gecerli) < len(girdi) else ""))
                if gecerli.empty:
                    st.error("Analiz edilecek metin yok.")
                elif st.button("Toplu Analizi Başlat", type="primary"):
                    st.session_state["toplu_sonuc"] = toplu_calistir(gecerli)
        if "toplu_sonuc" in st.session_state:
            toplu_sonuc_goster(st.session_state["toplu_sonuc"])


# ---------------------------------------------------------------- sayfa 3: model sonuçları


def karsilastirma_tablosu(metrik: dict) -> pd.DataFrame:
    satirlar = []
    for deney in metrik["deneyler"].values():
        for model, o in deney["modeller"].items():
            cok_kat = len(o["katlar"]) > 1
            zor = (o.get("alt_kumeler") or {}).get("zor")
            satirlar.append({
                "Deney": deney["aciklama"],
                "Model": MODEL_ETIKETLERI.get(model, model),
                "Macro-F1": f"{o['macro_f1_ort']:.3f}" + (f" ± {o['macro_f1_std']:.3f}" if cok_kat else ""),
                "Doğruluk": f"{o['accuracy_ort']:.3f}" + (f" ± {o['accuracy_std']:.3f}" if cok_kat else ""),
                "Zor örneklerde doğruluk": "—" if not zor else f"{zor['accuracy']:.3f} (n={zor['n']})",
                "Test n": deney["n_test"],
            })
    return pd.DataFrame(satirlar)


def bulgular(metrik: dict) -> list[str]:
    d, cumleler = metrik["deneyler"], []

    def f1(deney, model):
        return d[deney]["modeller"][model]["macro_f1_ort"]

    if "A_v1_cv" in d and "C_v1_v2" in d:
        cumleler.append(
            f"Özgün veride (v1) çapraz doğrulamada hibrit model macro-F1 **{f1('A_v1_cv', 'hibrit'):.2f}** alıyor; "
            f"aynı veriyle eğitilip kalıp kelimeleri azaltılmış yeni veride (v2) sınandığında **{f1('C_v1_v2', 'hibrit'):.2f}**. "
            "Aradaki fark, v1 skorunun ne kadarının veri setine özgü kalıplardan geldiğini gösterir.")
    ana = "B_tum_cv" if "B_tum_cv" in d else next(iter(d))
    fark = f1(ana, "hibrit") - f1(ana, "yalniz_bert")
    p = d[ana].get("mcnemar", {}).get("hibrit_vs_yalniz_bert", {}).get("p")
    anlam = "" if p is None else (f" (McNemar p = {p:.3f}; " + ("fark istatistiksel olarak anlamlı" if p < 0.05 else "fark istatistiksel olarak anlamlı değil") + ")")
    cumleler.append(f"{d[ana]['aciklama']}: hibrit **{f1(ana, 'hibrit'):.3f}**, yalnız BERTurk **{f1(ana, 'yalniz_bert'):.3f}** "
                    f"(fark {fark:+.3f}){anlam}.")
    cumleler.append(f"Aynı deneyde taban çizgileri: TF-IDF + LR **{f1(ana, 'tfidf_lr'):.3f}**, yalnız stilometri "
                    f"**{f1(ana, 'stilometri_lr'):.3f}** (şans düzeyi 0.33).")
    return cumleler


if sayfa == SAYFALAR[2]:
    st.title("Model Sonuçları")
    metrik = json_yukle(SONUC_YOLU / "metrics.json")
    if not metrik or "deneyler" not in metrik:
        st.warning("`results/metrics.json` bulunamadı. Sonuçları üretmek için `python scripts/deney.py` çalıştırın.")
    else:
        meta, veri = metrik["meta"], metrik["veri"]
        st.caption(f"Kaynak: results/metrics.json · {meta['tarih'][:10]} · commit {meta['commit']} · "
                   f"{veri['n']} metin · {meta['n_kat']}-kat konu-gruplu çapraz doğrulama · cihaz: {meta['cihaz']}")
        if meta.get("bert_yontemi_aciklama"):
            st.caption(f"Karşılaştırmadaki BERT modelleri: {meta['bert_yontemi_aciklama']} "
                       "Paneldeki nihai model ise ince ayarla eğitilmiştir (bkz. eğitim eğrisi).")
        st.markdown("### Öne çıkan bulgular")
        for cumle in bulgular(metrik):
            st.markdown(f"- {cumle}")
        st.markdown("### Karşılaştırma tablosu")
        st.dataframe(karsilastirma_tablosu(metrik), hide_index=True, width="stretch")
        st.caption("± değerleri katlar arası standart sapmadır. Tek bölmeli deneylerde (v1→v2, v2→v1) sapma yoktur.")

        denetim = veri.get("etiket_denetimi")
        if denetim:
            st.markdown("### Veri ve etiket denetimi")
            c1, c2, c3 = st.columns(3)
            c1.metric("Toplam metin", veri["n"])
            c2.metric("Kör denetçi uyumu (Cohen κ)", f"{denetim['cohen_kappa']:.2f}")
            c3.metric("Etiket uyuşmazlığından atılan", len(denetim["etiket_uyusmazligindan_atilan"]))

        st.markdown("### Grafikler")
        aciklamalar = {
            "model_karsilastirma.png": "Dört modelin dört deneydeki macro-F1 değerleri.",
            "confusion_matrix.png": "Hibrit modelin çapraz doğrulamadaki tüm test tahminleri (her örnek yalnız bir kez test edilir).",
            "egitim_egrisi.png": "Panelde kullanılan nihai modelin eğitim süreci.",
            "stilometri_karsilastirma.png": "Kural tabanlı stilometrik göstergelerin sınıflara göre ortalamaları.",
            "ozellik_radar.png": "Aynı göstergelerin sınıf profili olarak görünümü.",
            "demo_trend.png": "Kurgusal senaryo: eğitimde kullanılmayan 14 günlük günlüklerde modelin gerçek çıktısı.",
        }
        for ad, aciklama in aciklamalar.items():
            yol = SONUC_YOLU / ad
            if yol.exists():
                st.image(str(yol), width="stretch")
                st.caption(aciklama)
            else:
                st.info(f"`{ad}` henüz üretilmedi (`python scripts/gorseller.py`).")

        st.markdown("### Sınırlılıklar")
        st.markdown(
            "- Veri **sentetiktir** (dil modeliyle üretildi); gerçek personel kayıtlarında doğrulama yapılmadı.\n"
            "- Sınıflar klinik tanı değil, yazım üslubu kategorileridir.\n"
            "- Veri seti küçüktür; skorlar farklı veri ve tohumlarla değişebilir.\n"
            "- Stilometrik göstergeler sezgisel ek örüntüleridir; biçimbilimsel çözümleme yapılmaz.")


# ---------------------------------------------------------------- sayfa 4: sistem hakkında

if sayfa == SAYFALAR[3]:
    st.markdown("<h1 style='text-align:center'>Kalıcı Türk Bilim Üssü Personeli İçin<br>Yapay Zeka Dil Analizi Tabanlı<br>"
                "Psiko-Sosyal Karar Destek Sistemi</h1>", unsafe_allow_html=True)
    st.divider()
    sol, sag = st.columns(2)
    with sol:
        st.subheader("🤖 Mimari")
        st.code("""
┌──────────────────────────────────────┐
│           Ham Türkçe metin           │
└──────────────────┬───────────────────┘
        ┌──────────┴──────────┐
        ▼                     ▼
  ┌───────────┐       ┌───────────────┐
  │  BERTurk  │       │  Stilometri   │
  │   [CLS]   │       │ (ek örüntüleri)│
  │ 768 boyut │       │   10 özellik  │
  └─────┬─────┘       └───────┬───────┘
        │              Linear(10→32)
        │              ReLU+LayerNorm
        └──────────┬──────────┘
                   ▼  birleştirme (800)
          ┌─────────────────┐
          │ Linear(800→256) │
          │ ReLU + Dropout  │
          │ Linear(256→3)   │
          └────────┬────────┘
                   ▼
   sınıf olasılıkları → risk skoru
""")
    with sag:
        st.subheader("🛠 Teknik bilgiler")
        bilgi = json_yukle(MODEL_YOLU / "model_info.json")
        metrik = json_yukle(SONUC_YOLU / "metrics.json")
        satirlar = []
        if bilgi:
            satirlar += [("Taban model", bilgi.get("taban_model", "—")),
                         ("Eğitim tarihi", str(bilgi.get("tarih", "—"))[:10]),
                         ("Eğitim / doğrulama örneği", f"{bilgi.get('n_egitim', '—')} / {bilgi.get('n_dogrulama', '—')}"),
                         ("Seçilen epoch", bilgi.get("en_iyi_epoch", "—")),
                         ("Stilometrik özellik sayısı", bilgi.get("n_stilo", "—")),
                         ("Kalibrasyon sıcaklığı", f"{bilgi.get('sicaklik', 1.0):.2f}")]
        if metrik and "veri" in metrik:
            satirlar += [("Veri seti", f"{metrik['veri']['n']} sentetik Türkçe metin"),
                         ("Sınıf sayısı", len(metrik.get("siniflar", SINIF_ADLARI)))]
        if satirlar:
            st.dataframe(pd.DataFrame(satirlar, columns=["Bileşen", "Değer"]).astype(str), hide_index=True, width="stretch")
            st.caption("Başarım ölçütleri için “Model Sonuçları” sayfasına bakın.")
        else:
            st.info("Model henüz eğitilmedi; teknik bilgiler eğitimden sonra burada görünür.")
    st.divider()
    st.subheader("⚖️ Etik not")
    st.info("Bu sistem tıbbi tanı aracı değildir; yalnızca yazım üslubundaki örüntüleri işaretleyen bir karar destek "
            "prototipidir. Sonuçlar bir insanın değerlendirmesinin yerine geçmez. Panel, girilen metinleri diske "
            "kaydetmez. Gerçek personel verisiyle kullanılmadan önce açık rıza, KVKK uyum değerlendirmesi ve alan "
            "uzmanı doğrulaması gerekir; bunların hiçbiri bu prototip için yapılmamıştır.")
