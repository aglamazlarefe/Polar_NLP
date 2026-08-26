"""
=============================================================
Polar NLP — Sonuç Görselleştirme Modülü
=============================================================
Antarktika üssü personel psikoloji analiz sisteminin
eğitim ve test sonuçlarını görselleştiren 5 grafik üretir.
Kütüphaneler: matplotlib, seaborn, numpy
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# === GLOBAL TASARIM SABİTLERİ ===
BG_COLOR       = '#0a1628'   # Koyu lacivert — Polar teması
TEXT_COLOR     = '#cfe8f5'   # Açık mavi-beyaz
GRID_COLOR     = '#607d8b'   # Gri-mavi grid
GRID_ALPHA     = 0.15
KAYIT_KLASORU  = 'results'   # Tüm PNG'ler bu klasöre kaydedilir


def _apply_dark_theme(ax):
    """Tüm eksenlere ortak koyu tema uygular."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)


# ==============================================================
# GRAFİK 1: Karmaşıklık Matrisi (Confusion Matrix) Isı Haritası
# ==============================================================
def grafik_1_confusion_matrix():
    """10×10 mükemmel sınıflandırma matrisini seaborn heatmap ile çizer."""

    # Gerçek test sonuçları — her sınıftan 10 örnek, 0 hata
    cm = np.array([
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10]
    ])
    sinif_etiketleri = ['Objektif', 'Bil. Yorgunluk', 'Kış Sendromu']

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG_COLOR)
    _apply_dark_theme(ax)

    # Isı haritası çizimi
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=sinif_etiketleri,
        yticklabels=sinif_etiketleri,
        linewidths=0.5,
        linecolor=GRID_COLOR,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )

    # Renk çubuğu etiket rengi düzeltme
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cbar.ax.set_facecolor(BG_COLOR)

    # Tick etiketleri rengi
    ax.set_xticklabels(sinif_etiketleri, color=TEXT_COLOR, fontsize=10)
    ax.set_yticklabels(sinif_etiketleri, color=TEXT_COLOR, fontsize=10, rotation=0)

    ax.set_title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=13,
                 color=TEXT_COLOR, pad=12, fontweight='bold')
    ax.set_xlabel('Tahmin Edilen Sınıf', fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel('Gerçek Sınıf', fontsize=10, color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(os.path.join(KAYIT_KLASORU, 'confusion_matrix.png'), dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close('all')
    print("  [OK] results/confusion_matrix.png kaydedildi")


# ==============================================================
# GRAFİK 2: Model Eğitim Eğrisi (Çift Y Ekseni)
# ==============================================================
def grafik_2_egitim_egrisi():
    """Train Loss ve Val F1 skorlarını twinx ile çift eksende gösterir."""

    # Gerçek eğitim logu verileri
    epochs     = list(range(1, 9))
    train_loss = [0.5400, 0.4960, 0.4617, 0.3397, 0.1735, 0.0784, 0.0403, 0.0230]
    val_f1     = [0.1667, 0.1667, 0.1667, 0.9327, 1.0000, 1.0000, 1.0000, 1.0000]

    LOSS_COLOR = '#4fc3f7'   # Açık mavi — train loss için
    F1_COLOR   = '#66bb6a'   # Yeşil — val F1 için

    fig, ax1 = plt.subplots(figsize=(9, 5), facecolor=BG_COLOR)
    _apply_dark_theme(ax1)

    # Sol eksen: Train Loss
    ln1 = ax1.plot(epochs, train_loss, color=LOSS_COLOR, marker='o',
                   linewidth=2.5, markersize=7, label='Train Loss', zorder=3)
    ax1.set_xlabel('Epoch', color=TEXT_COLOR, fontsize=11)
    ax1.set_ylabel('Train Loss', color=LOSS_COLOR, fontsize=11)
    ax1.tick_params(axis='y', colors=LOSS_COLOR)
    ax1.tick_params(axis='x', colors=TEXT_COLOR)
    ax1.set_xticks(epochs)
    ax1.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linestyle='--')

    # Sağ eksen: Validation F1
    ax2 = ax1.twinx()
    ax2.set_facecolor(BG_COLOR)
    ax2.tick_params(axis='y', colors=F1_COLOR)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    ln2 = ax2.plot(epochs, val_f1, color=F1_COLOR, marker='s', linestyle='--',
                   linewidth=2.5, markersize=7, label='Val F1', zorder=3)
    ax2.set_ylabel('Validation F1', color=F1_COLOR, fontsize=11)
    ax2.set_ylim(0, 1.12)
    ax2.yaxis.label.set_color(F1_COLOR)

    # Epoch 4 kırılma noktası anotasyonu
    ax1.annotate(
        'Öğrenme\nKırılma Noktası',
        xy=(4, 0.3397),
        xytext=(5.2, 0.42),
        fontsize=9.5,
        color='white',
        fontweight='bold',
        arrowprops=dict(
            arrowstyle='->', color='white', lw=1.8,
            connectionstyle='arc3,rad=0.2'
        ),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a2a45', edgecolor=GRID_COLOR, alpha=0.9)
    )

    # Birleşik lejant
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right', facecolor='#1a2a45',
               edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    ax1.set_title('Model Eğitim Süreci', fontsize=13, color=TEXT_COLOR,
                  pad=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(KAYIT_KLASORU, 'egitim_egrisi.png'), dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close('all')
    print("  [OK] results/egitim_egrisi.png kaydedildi")


# ==============================================================
# GRAFİK 3: Stilometrik Özellik Dağılımı (Grouped Bar)
# ==============================================================
def grafik_3_stilometri():
    """
    Stilometrik özellikleri 2 subplot ile gösterir:
    Sol: Cümle Uzunluğu (7-11 aralığı)
    Sağ: Diğer 4 özellik (0-0.015 aralığı)
    """

    SINIF_RENKLER    = ['#66bb6a', '#ffb300', '#ef5350']
    sinif_etiketleri = ['Objektif (0)', 'Bil. Yorgunluk (1)', 'Kış Sendromu (2)']
    bar_w = 0.22

    # Sol subplot — sadece cümle uzunluğu değerleri
    cumle_deger = np.array([7.90, 10.63, 9.20])

    # Sağ subplot — diğer 4 özellik (satır: özellik, sütun: sınıf)
    diger_ozellikler = ['Olumsuzluk\nOranı', 'Mutlak\nİfade', 'Tekil\nZamir']
    diger_deger = np.array([
        [0.003, 0.001, 0.010],   # olumsuzluk_orani
        [0.000, 0.001, 0.001],   # mutlak_ifade
        [0.000, 0.000, 0.001],   # tekil_zamir
    ])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5),
                                     facecolor=BG_COLOR,
                                     gridspec_kw={'width_ratios': [1, 2.2]})
    _apply_dark_theme(ax_l)
    _apply_dark_theme(ax_r)

    # ── Sol Subplot: Cümle Uzunluğu ─────────────────────────────────
    x_l = np.arange(1)
    for i, (renk, etiket) in enumerate(zip(SINIF_RENKLER, sinif_etiketleri)):
        ofset = (i - 1) * bar_w
        ax_l.bar(x_l + ofset, cumle_deger[i], width=bar_w,
                 color=renk, label=etiket, alpha=0.88,
                 edgecolor=BG_COLOR, linewidth=0.6)

    ax_l.set_xticks(x_l)
    ax_l.set_xticklabels(['Cümle\nUzunluğu'], color=TEXT_COLOR, fontsize=10)
    ax_l.set_ylim(7, 11)
    ax_l.set_ylabel('Ortalama Kelime Sayısı', color=TEXT_COLOR, fontsize=9.5)
    ax_l.set_title('Cümle Uzunluğu', color=TEXT_COLOR, fontsize=11,
                   fontweight='bold', pad=10)
    ax_l.grid(axis='y', color=GRID_COLOR, alpha=GRID_ALPHA, linestyle='--')
    ax_l.legend(facecolor='#1a2a45', edgecolor=GRID_COLOR,
                labelcolor=TEXT_COLOR, fontsize=8.5)

    # ── Sağ Subplot: Diğer 4 Özellik ────────────────────────────────
    x_r = np.arange(len(diger_ozellikler))
    for i, (renk, _) in enumerate(zip(SINIF_RENKLER, sinif_etiketleri)):
        ofset = (i - 1) * bar_w
        ax_r.bar(x_r + ofset, diger_deger[:, i], width=bar_w,
                 color=renk, alpha=0.88, edgecolor=BG_COLOR, linewidth=0.6)

    ax_r.set_xticks(x_r)
    ax_r.set_xticklabels(diger_ozellikler, color=TEXT_COLOR, fontsize=9.5)
    ax_r.set_ylim(0.000, 0.015)
    ax_r.set_ylabel('Ortalama Oran', color=TEXT_COLOR, fontsize=9.5)
    ax_r.set_title('Olumsuzluk, Mutlak İfade, Tekil Zamir',
                   color=TEXT_COLOR, fontsize=11, fontweight='bold', pad=10)
    ax_r.grid(axis='y', color=GRID_COLOR, alpha=GRID_ALPHA, linestyle='--')

    # Ana başlık
    fig.suptitle('Sınıf Bazlı Stilometrik Özellik Dağılımı',
                 fontsize=13, color=TEXT_COLOR, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(os.path.join(KAYIT_KLASORU, 'stilometri_karsilastirma.png'),
                dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close('all')
    print("  [OK] results/stilometri_karsilastirma.png kaydedildi")


# ==============================================================
# GRAFİK 4: Senaryo Simülasyonu — 14 Günlük Risk Takibi
# ==============================================================
def grafik_4_senaryo():
    """Jeneratör arızası senaryosunda 14 günlük psikolojik risk seyrini gösterir."""

    gunler = list(range(1, 15))   # 1'den 14'e
    risk = [0.15, 0.18, 0.20, 0.22, 0.45, 0.58, 0.62,
            0.71, 0.78, 0.85, 0.88, 0.91, 0.93, 0.95]

    RISK_COLOR     = '#ef5350'   # Kırmızı — risk çizgisi
    ORTA_ESIK      = 0.60
    YUKSEK_ESIK    = 0.80

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BG_COLOR)
    _apply_dark_theme(ax)

    # Alan dolgusu
    ax.fill_between(gunler, risk, alpha=0.15, color=RISK_COLOR)

    # Ana risk çizgisi
    ax.plot(gunler, risk, color=RISK_COLOR, linewidth=2.5, marker='o',
            markersize=5.5, zorder=4, label='Psikolojik Risk Skoru')

    # Eşik çizgileri
    ax.axhline(ORTA_ESIK, color='#ffb300', linestyle='--', linewidth=1.8,
               label='Orta Risk Eşiği (0.60)', alpha=0.85)
    ax.axhline(YUKSEK_ESIK, color=RISK_COLOR, linestyle='--', linewidth=1.8,
               label='Yüksek Risk Eşiği (0.80)', alpha=0.85)

    # Gün 5-9 arası arka plan vurgusu — sistemimiz bunu fark eder
    ax.axvspan(5, 9, alpha=0.08, color='#66bb6a',
               label='Sistemimiz fark eder', zorder=1)

    # Gün 10-14 arası arka plan vurgusu — geleneksel yöntem
    ax.axvspan(10, 14, alpha=0.08, color=RISK_COLOR,
               label='Geleneksel yöntem fark eder', zorder=1)

    # Gün 5 anotasyonu — Orta Risk Uyarısı
    ax.annotate(
        '⚠ Orta Risk Uyarısı',
        xy=(5, risk[4]),
        xytext=(5.5, 0.38),
        fontsize=9.5,
        color='#ffb300',
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#ffb300', lw=1.5,
                        connectionstyle='arc3,rad=-0.2'),
        bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_COLOR,
                  edgecolor='#ffb300', alpha=0.9)
    )

    # Gun 10 anotasyonu - Mudahale Onerisi
    ax.annotate(
        '[!] Mudahale Onerisi',
        xy=(10, risk[9]),
        xytext=(9.2, 0.70),
        fontsize=9.5,
        color=RISK_COLOR,
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=RISK_COLOR, lw=1.5,
                        connectionstyle='arc3,rad=0.2'),
        bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_COLOR,
                  edgecolor=RISK_COLOR, alpha=0.9)
    )

    # Bölge etiketleri
    ax.text(6.5, 0.05, 'Sistemimiz fark eder', ha='center',
            color='#66bb6a', fontsize=9, alpha=0.85)
    ax.text(12.0, 0.05, 'Geleneksel\nyöntem', ha='center',
            color=RISK_COLOR, fontsize=9, alpha=0.85)

    ax.set_xlim(1, 14)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(gunler)
    ax.set_xticklabels([f'Gün {g}' for g in gunler], rotation=30,
                       ha='right', color=TEXT_COLOR, fontsize=8.5)
    ax.set_xlabel('Gün', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('Risk Skoru', color=TEXT_COLOR, fontsize=10)
    ax.set_title('Jeneratör Arızası Senaryosu — 14 Günlük Risk Takibi',
                 fontsize=13, color=TEXT_COLOR, pad=12, fontweight='bold')
    ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linestyle='--')
    ax.legend(facecolor='#1a2a45', edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8.5, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(KAYIT_KLASORU, 'senaryo_simulasyon.png'), dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close('all')
    print("  [OK] results/senaryo_simulasyon.png kaydedildi")


# ==============================================================
# GRAFİK 5: Psikolojik Risk Profili — Radar Grafiği (Polar)
# ==============================================================
def grafik_5_radar():
    """İki kişilik profili (Objektif vs Kış Sendromu) radar grafiğinde karşılaştırır."""

    kategoriler = ['Bilişsel\nYorgunluk', 'Sosyal\nÇekilme',
                   'Görev\nReddi', 'Öz-\nOdaklanma']
    n_kat = len(kategoriler)

    # Açılar: n kategori için eşit dağılım, kapalı poligon için ilk açı tekrar eklenir
    acılar = np.linspace(0, 2 * np.pi, n_kat, endpoint=False).tolist()
    acılar += acılar[:1]   # Kapalı poligon için kapama

    # Profil değerleri
    degerler_objektif = [0.15, 0.10, 0.05, 0.08]
    degerler_kis      = [0.72, 0.90, 0.95, 0.65]

    # Kapalı poligon için değerler de tekrarlanır
    degerler_objektif += degerler_objektif[:1]
    degerler_kis      += degerler_kis[:1]

    fig = plt.figure(figsize=(11, 5.5), facecolor=BG_COLOR)

    # ------- Sol: Objektif Rapor -------
    ax1 = fig.add_subplot(121, projection='polar', facecolor=BG_COLOR)
    ax1.plot(acılar, degerler_objektif, color='#66bb6a', linewidth=2.2)
    ax1.fill(acılar, degerler_objektif, color='#66bb6a', alpha=0.25)
    ax1.set_xticks(acılar[:-1])
    ax1.set_xticklabels(kategoriler, color=TEXT_COLOR, fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                        color=TEXT_COLOR, fontsize=7, alpha=0.7)
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.grid(color=GRID_COLOR, alpha=0.3, linestyle='--')
    ax1.set_facecolor(BG_COLOR)
    # Polar eksenlerinin dış çerçeve ve ızgara rengi
    ax1.spines['polar'].set_color(GRID_COLOR)
    ax1.title.set_text('Objektif Rapor')
    ax1.title.set_color('#66bb6a')
    ax1.title.set_fontsize(12)
    ax1.title.set_fontweight('bold')
    ax1.title.set_position((0.5, 1.15))

    # ------- Sağ: Kış Sendromu -------
    ax2 = fig.add_subplot(122, projection='polar', facecolor=BG_COLOR)
    ax2.plot(acılar, degerler_kis, color='#ef5350', linewidth=2.2)
    ax2.fill(acılar, degerler_kis, color='#ef5350', alpha=0.25)
    ax2.set_xticks(acılar[:-1])
    ax2.set_xticklabels(kategoriler, color=TEXT_COLOR, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                        color=TEXT_COLOR, fontsize=7, alpha=0.7)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.grid(color=GRID_COLOR, alpha=0.3, linestyle='--')
    ax2.set_facecolor(BG_COLOR)
    ax2.spines['polar'].set_color(GRID_COLOR)
    ax2.title.set_text('Kış Sendromu')
    ax2.title.set_color('#ef5350')
    ax2.title.set_fontsize(12)
    ax2.title.set_fontweight('bold')
    ax2.title.set_position((0.5, 1.15))

    # Genel başlık
    fig.suptitle('Psikolojik Risk Profili Karşılaştırması', fontsize=13,
                 color=TEXT_COLOR, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(os.path.join(KAYIT_KLASORU, 'radar_profil.png'), dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close('all')
    print("  [OK] results/radar_profil.png kaydedildi")


# ==============================================================
# ANA ÇALIŞTIRICI
# ==============================================================
def main():
    # Kayıt klasörünü oluştur (zaten varsa atla)
    os.makedirs(KAYIT_KLASORU, exist_ok=True)
    print("=" * 55)
    print("  Polar NLP - Gorseller uretiliyor...")
    print("  Kayit klasoru: results/")
    print("=" * 55)
    grafik_1_confusion_matrix()
    grafik_2_egitim_egrisi()
    grafik_3_stilometri()
    grafik_4_senaryo()
    grafik_5_radar()
    print("=" * 55)
    print("  Tamamlandi. 5 dosya results/ klasorune kaydedildi.")
    print("  - results/confusion_matrix.png")
    print("  - results/egitim_egrisi.png")
    print("  - results/stilometri_karsilastirma.png")
    print("  - results/senaryo_simulasyon.png")
    print("  - results/radar_profil.png")
    print("=" * 55)


if __name__ == '__main__':
    main()
