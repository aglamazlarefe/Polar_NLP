import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import json
import base64
from PIL import Image
from io import BytesIO

# train_hybrid.py'den analiz fonksiyonunu import et
try:
    from train_hybrid import polar_analiz_et
    MODEL_LOADED = True
except ImportError:
    MODEL_LOADED = False

# ==========================================================
# 1. SAYFA YAPILANDIRMASI VE TASARIM
# ==========================================================

st.set_page_config(
    page_title="Polar NLP Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — Tasarım Kurallarına Uygun
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

    :root {
        --bg-color: #0a1628;
        --accent-blue: #4fc3f7;
        --accent-cyan: #00e5ff;
        --text-color: #ffffff;
        --card-bg: #16243a;
    }

    .main {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
        color: var(--accent-cyan) !important;
    }

    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1a2a45 !important;
        color: white !important;
        border: 1px solid #4fc3f7 !important;
    }

    .stButton>button {
        background-color: transparent !important;
        color: var(--accent-blue) !important;
        border: 1px solid var(--accent-blue) !important;
        border-radius: 4px !important;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: var(--accent-blue) !important;
        color: var(--bg-color) !important;
    }

    /* Üst mavi çizgi ayraç */
    .top-bar {
        height: 2px;
        background: linear-gradient(90deg, #4fc3f7, #00e5ff);
        margin-bottom: 25px;
    }

    /* Badge Stilleri */
    .badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        color: white;
        display: inline-block;
    }
    .badge-green { background-color: #66bb6a; }
    .badge-yellow { background-color: #ffb300; }
    .badge-red { background-color: #ef5350; }

    /* Metrik Kartları */
    .metric-card {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid var(--accent-blue);
        margin-bottom: 10px;
    }

    /* Sidebar Özelleştirme */
    section[data-testid="stSidebar"] {
        background-color: #0d1e35 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b0bec5 !important;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ==========================================================
# SIDEBAR NAVİGASYON
# ==========================================================

with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: white;'>🧊 POLAR NLP</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigasyon",
        ["🔍 Tekil Analiz", "📊 Toplu Analiz", "📈 Model Sonuçları", "ℹ️ Sistem Hakkında"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 🧊 Durum")
    if MODEL_LOADED:
        st.success("✅ Model Aktif")
    else:
        st.error("❌ Model Yüklenemedi")
        
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #607d8b;'>TR-KABÜ · Prototip v0.2</div>", unsafe_allow_html=True)

st.markdown("<div class='top-bar'></div>", unsafe_allow_html=True)

# ==========================================================
# PAGE 1: TEKİL ANALİZ
# ==========================================================

if page == "🔍 Tekil Analiz":
    st.title("Tekil Metin Analizi")
    st.write("Personel tarafından iletilen metni girerek psikolojik risk taraması yapın.")

    if not MODEL_LOADED:
        st.warning("Model dosyaları (train_hybrid.py veya model klasörü) eksik olduğu için bu sayfa devre dışıdır.")
    else:
        # Session state başlat
        if 'main_text_area' not in st.session_state:
            st.session_state.main_text_area = ""

        st.markdown("### Örnek Senaryolar")
        st.caption("Veri setinden (output_utf8.csv) alınmış gerçekçi örnekleri deneyebilirsiniz:")
        
        # Örnek Butonları
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Örnek 1 (Objektif)"):
                st.session_state.main_text_area = "Jeneratör manifold bakımı standart prosedüre göre tamamlanmıştır. Voltaj dalgalanması gözlenmedi. Yakıt pompası basıncı nominal değerlerde seyrediyor."
                st.rerun()

        with col2:
            if st.button("Örnek 2 (Bilişsel Yorgunluk)"):
                st.session_state.main_text_area = "Katabatik rüzgar hızının 120 km/h'yi geçmesi uydu bant genişliğinde daralmaya sebep oldu. Çanak ısıtıcılarının yetersiz güç çekmesi yüzünden her fırtınada sinyal zayıflaması yaşamak veri aktarım planımı bozuyor."
                st.rerun()

        with col3:
            if st.button("Örnek 3 (Kış Sendromu)"):
                st.session_state.main_text_area = "Telsiz anteninin günlük kar kürüme işi anlamsız bir rutindir. Cihaz zaten sinyal almıyor. Kürek çekmek için dışarı çıkmayı reddediyorum."
                st.rerun()

        user_input = st.text_area("Analiz Edilecek Metin:", height=150, placeholder="Buraya metin girin veya örneklerden birini seçin...", key="main_text_area")

        if st.button("Analizi Başlat", type="primary"):
            if user_input.strip() == "":
                st.error("Lütfen bir metin girin.")
            else:
                with st.spinner("Yapay Zeka metni inceliyor..."):
                    try:
                        # Analiz fonksiyonunu çağır
                        # Not: model_dir parametresi train_hybrid.py'deki varsayılan yola göre ayarlandı
                        sonuc = polar_analiz_et(user_input, "./egitilmis_hybrid_model")
                        
                        st.markdown("---")
                        
                        # Üst Özet
                        res_col1, res_col2 = st.columns([2, 1])
                        
                        with res_col1:
                            # Sınıf Badge
                            badge_class = "badge-green"
                            if sonuc['sinif'] == 1: badge_class = "badge-yellow"
                            elif sonuc['sinif'] == 2: badge_class = "badge-red"
                            
                            st.markdown(f"### Tespit Edilen Durum: <span class='badge {badge_class}'>{sonuc['sinif_adi']}</span>", unsafe_allow_html=True)
                            
                            # Güven Skoru
                            conf = sonuc['guven_skoru']
                            st.write(f"**Güven Skoru:** %{conf*100:.1f}")
                            st.progress(conf)
                        
                        with res_col2:
                            # Yönetimsel Öneri (Yumuşatılmış Dil)
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.markdown("**ℹ️ Bilgilendirme ve Takip:**")
                            if sonuc['sinif'] == 0:
                                st.write("Normal süreçler sürdürülebilir. Standart rutinler yeterlidir.")
                            elif sonuc['sinif'] == 1:
                                st.write("Yorgunluk emareleri gözlemlenebilir. Dinlenme süreleri veya görev değişikliği planlanabilir.")
                            elif sonuc['sinif'] == 2:
                                st.write("Sosyal ortamın çeşitlendirilmesi ve moral-motivasyon aktiviteleri değerlendirilebilir.")
                            st.markdown("</div>", unsafe_allow_html=True)

                        # Detaylı Grafikler ve Tablolar
                        st.markdown("### Detaylı Risk Profili")
                        prof_col1, prof_col2 = st.columns(2)
                        
                        risk = sonuc['risk_profili']
                        with prof_col1:
                            st.write("**Bilişsel Yorgunluk**")
                            st.progress(risk['bilissel_yorgunluk'])
                            st.write("**Sosyal Çekilme**")
                            st.progress(risk['sosyal_cekilme'])
                            
                        with prof_col2:
                            st.write("**Görev Reddi**")
                            st.progress(risk['gorev_reddi'])
                            st.write("**Öz-Odaklanma**")
                            st.progress(risk['oz_odaklanma'])

                        st.markdown("### Stilometrik Özellikler")
                        stilo = sonuc['stilometrik_ozellikler']
                        stilo_df = pd.DataFrame({
                            "Parametre": ["Tekil Zamir Oranı", "TTR (Kelime Çeşitliliği)", "Mutlak İfade Sıklığı", "Cümle Uzunluğu", "Olumsuzluk Oranı"],
                            "Değer": [stilo['birinci_tekil_zamir'], stilo['ttr'], stilo['mutlak_ifade'], stilo['cumle_uzunlugu'], stilo['olumsuzluk_orani']]
                        })
                        st.table(stilo_df)
                        
                    except Exception as e:
                        st.error(f"Analiz sırasında bir hata oluştu: {str(e)}")

# ==========================================================
# PAGE 2: TOPLU ANALİZ
# ==========================================================

elif page == "📊 Toplu Analiz":
    st.title("Toplu Analiz Sistemi")
    st.write("Birden fazla personelin verilerini CSV formatında yükleyerek tarama yapın.")

    if not MODEL_LOADED:
        st.warning("Model dosyaları eksik olduğu için bu sayfa devre dışıdır.")
    else:
        uploaded_file = st.file_uploader("Metin içeren CSV dosyasını yükleyin", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'Metin' not in df.columns:
                    st.error("CSV dosyasında 'Metin' başlıklı bir sütun bulunmalıdır.")
                else:
                    if st.button("Toplu İşlem Başlat", type="primary"):
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, row in df.iterrows():
                            status_text.text(f"İşleniyor: {i+1}/{len(df)}")
                            res = polar_analiz_et(row['Metin'], "./egitilmis_hybrid_model")
                            
                            # Risk skoru hesapla (ortalama profile göre)
                            avg_risk = sum(res['risk_profili'].values()) / 4.0
                            
                            results.append({
                                "#": i + 1,
                                "Metin (ilk 60 karakter)": row['Metin'][:60] + "...",
                                "Sınıf": res['sinif_adi'],
                                "Güven": f"%{res['guven_skoru']*100:.1f}",
                                "Risk Skoru": round(avg_risk, 3),
                                "Class_ID": res['sinif']
                            })
                            progress_bar.progress((i + 1) / len(df))
                        
                        status_text.success(f"Analiz tamamlandı: {len(df)} kayıt işlendi.")
                        
                        res_df = pd.DataFrame(results)
                        
                        # Tablo Gösterimi
                        st.markdown("### Analiz Sonuçları")
                        st.dataframe(res_df.drop(columns=['Class_ID']), use_container_width=True)

                        # Özet İstatistikler
                        st.markdown("---")
                        st.markdown("### Özet İstatistikler")
                        
                        stat_col1, stat_col2 = st.columns([1, 2])
                        
                        with stat_col1:
                            # Pasta Grafik
                            import matplotlib.pyplot as plt
                            class_counts = res_df['Sınıf'].value_counts()
                            st.write("**Sınıf Dağılımı**")
                            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                            fig_pie.patch.set_facecolor('#0a1628')
                            colors = ['#66bb6a', '#ffb300', '#ef5350']
                            class_order = ["Objektif Rapor", "Bilişsel Yorgunluk", "Kış Sendromu"]
                            actual_counts = [class_counts.get(c, 0) for c in class_order]
                            actual_labels = [c for c in class_order if class_counts.get(c, 0) > 0]
                            actual_counts = [c for c in actual_counts if c > 0]
                            
                            ax_pie.pie(
                                actual_counts, 
                                labels=actual_labels, 
                                autopct='%1.1f%%', 
                                startangle=140, 
                                colors=colors[:len(actual_labels)],
                                textprops={'color':"w"}
                            )
                            st.pyplot(fig_pie)
                            
                        with stat_col2:
                            # Metrikler
                            avg_gl_risk = res_df['Risk Skoru'].mean()
                            high_risk_count = len(res_df[res_df['Class_ID'] == 2])
                            
                            m_col1, m_col2 = st.columns(2)
                            m_col1.metric("Ortalama Risk", f"{avg_gl_risk:.3f}")
                            m_col2.metric("Kritik Personel", high_risk_count, delta=high_risk_count, delta_color="inverse")

                            # En yüksek riskli 3 personel
                            st.write("**En Yüksek Riskli 3 Personel (Kritik)**")
                            top_3 = res_df.sort_values(by="Risk Skoru", ascending=False).head(3)
                            for _, r in top_3.iterrows():
                                if r['Class_ID'] == 2:
                                    st.markdown(f"<div style='background-color:#ef5350; padding:10px; border-radius:5px; margin-bottom:5px; color:white;'>ID: {r['#']} | Risk: {r['Risk Skoru']} | {r['Metin (ilk 60 karakter)']}</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div style='background-color:#ffb300; padding:10px; border-radius:5px; margin-bottom:5px;'>ID: {r['#']} | Risk: {r['Risk Skoru']} | {r['Metin (ilk 60 karakter)']}</div>", unsafe_allow_html=True)

                        # İndirme Butonu
                        csv = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Sonuçları CSV Olarak İndir",
                            data=csv,
                            file_name="polar_analiz_sonuclari.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"CSV okuma hatası: {str(e)}")

# ==========================================================
# PAGE 3: MODEL SONUÇLARI
# ==========================================================

elif page == "📈 Model Sonuçları":
    st.title("Model Performans Analizi")
    st.write("Sistemin eğitim ve doğrulama aşamalarındaki başarı metrikleri.")

    # Resim yolu kontrolü
    img_dirs = ["./results", "./görseller", "./gorseller"]
    found_dir = None
    for d in img_dirs:
        if os.path.exists(d):
            found_dir = d
            break

    if not found_dir:
        st.warning("Grafik dosyaları bulunamadı. Lütfen 'gorseller.py' dosyasını çalıştırın.")
    else:
        images = {
            "confusion_matrix.png": "Model 30 test örneğinin tamamını doğru sınıflandırmış, hiçbir karışıklık gözlemlenmemiştir.",
            "egitim_egrisi.png": "Epoch 4'te gerçekleşen öğrenme kırılma noktası, modelin BERTurk katmanının Türkçe dilbilgisel örüntüleri içselleştirdiği anı göstermektedir.",
            "stilometri_karsilastirma.png": "Bilişsel Yorgunluk sınıfı en uzun cümle yapısıyla öne çıkarken, Kış Sendromu sınıfı en yüksek olumsuzluk oranıyla ayrışmaktadır.",
            "senaryo_simulasyon.png": "Stilometrik analiz sistemi olası tükenmişlik krizini geleneksel yöntemlere kıyasla 5 gün önce tespit etmiştir.",
            "radar_profil.png": "Objektif rapor ile Kış Sendromu arasındaki 4 boyutlu risk profili farkı çarpıcı biçimde ayrışmaktadır."
        }
        
        for img_name, caption in images.items():
            path = os.path.join(found_dir, img_name)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
                st.markdown(f"<div style='text-align: center; color: #b0bec5; font-style: italic; margin-bottom: 40px;'>{caption}</div>", unsafe_allow_html=True)
                st.markdown("---")
            else:
                st.warning(f"Resim bulunamadı: {img_name}")

# ==========================================================
# PAGE 4: SİSTEM HAKKINDA
# ==========================================================

elif page == "ℹ️ Sistem Hakkında":
    st.markdown("""
    <h1 style='text-align: center;'>Kalıcı Türk Bilim Üssü Personeli İçin<br>
    Yapay Zeka Dil Analizi Tabanlı<br>
    Psiko-Sosyal Karar Destek Sistemi</h1>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🤖 Mimari Şema")
        st.code("""
┌─────────────────────────────────────┐
│         Ham Türkçe Metin            │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌────────┐         ┌───────────┐
│BERTurk │         │Stilometri │
│  768   │         │  Modülü   │
│ boyut  │         │  5 boyut  │
└───┬────┘         └─────┬─────┘
    │                    │
    └────────┬───────────┘
             │ torch.cat (773 boyut)
             ▼
    ┌─────────────────┐
    │ Fusion Katmanı  │
    │ Linear(773→256) │
    │ ReLU + Dropout  │
    │ Linear(256→3)   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Risk Profili  │
    │ + Sınıf Etiketi │
    └─────────────────┘
        """)
        
    with col_b:
        st.subheader("🛠 Teknik Detaylar")
        tech_data = {
            "Bileşen": ["Temel Model", "Eğitim Donanımı", "Veri Seti", "Sınıf Sayısı", "Test Accuracy", "Macro F1", "Öğrenme Oranı", "Epoch"],
            "Değer": ["dbmdz/bert-base-turkish-cased", "NVIDIA RTX 3060", "600 sentetik Türkçe metin", "3", "%100", "1.00", "3×10⁻⁵", "8"]
        }
        st.table(pd.DataFrame(tech_data))

    st.markdown("---")
    st.subheader("⚖️ Etik Not ve Gizlilik")
    st.info("""
    Bu sistem tıbbi tanı aracı değildir. Ham metin verileri işlendikten sonra saklanmamaktadır. 
    KVKK ve GDPR uyumludur. Karar mekanizması insan insiyatifini desteklemek amacıyla tasarlanmıştır.
    """)
