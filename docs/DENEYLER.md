# Deneyler: nasıl çalıştırılır, nasıl okunur

Güncel sayılar: [SONUCLAR.md](SONUCLAR.md) (otomatik üretilir). Bu belge yöntemi anlatır.

## Çalıştırma sırası

```bash
.venv\Scripts\python scripts/deney.py        # 1) karşılaştırmalar  -> results/metrics.json ("deneyler")
.venv\Scripts\python scripts/egit_final.py   # 2) panelin modeli    -> egitilmis_hybrid_model/, metrics.json ("final"), demo_trend.json
.venv\Scripts\python scripts/gorseller.py    # 3) grafikler         -> results/*.png
.venv\Scripts\python scripts/rapor.py        # 4) tablo             -> docs/SONUCLAR.md
.venv\Scripts\python -m pytest               # 5) tutarlılık denetimi
```

Veri ya da `EgitimAyari` değişirse beş adımın hepsi yinelenir. `deney.py` yarıda kesilirse aynı komutla kaldığı
yerden sürer. Önce hattın çalıştığını görmek için `scripts/deney.py --hizli` (yaklaşık 2 dk, `results/`'a dokunmaz).

## Deneyler

| Kod | Eğitim | Test | Sorduğu soru |
|---|---|---|---|
| A | v1 (çapraz doğ.) | v1 | Özgün veride model ne kadar başarılı görünüyor? |
| B | v1+v2 (çapraz doğ.) | v1+v2 | Birleşik, daha zor veride gerçekçi başarım nedir? **Ana sonuç budur.** |
| C | v1 | v2 | v1'de öğrenilen şey, kalıp kelimeleri azaltılmış yeni veriye taşınıyor mu? |
| D | v2 | v1 | Tersi: zor veride öğrenilen şey özgün veriye taşınıyor mu? |

## Modeller

| Ad | Ne ölçer |
|---|---|
| `tfidf_lr` | Yalnız kelime/ikili kelime sayımıyla ne kadar gidilebiliyor? Yüksekse veri kelime sızdırıyor demektir. |
| `stilometri_lr` | 10 kural tabanlı özellik tek başına ne kadar bilgi taşıyor? |
| `yalniz_bert` | Stilometri olmadan BERTurk. |
| `hibrit` | BERTurk + stilometri. Projenin önerdiği model. |

## Sonuçları okurken

- **± değeri** katlar arası standart sapmadır. Beş katla hesaplandığı için kabadır; iki modelin aralıkları
  örtüşüyorsa "biri daha iyi" demeyin.
- **McNemar p-değeri** iki modelin aynı örneklerdeki hatalarını karşılaştırır. p ≥ 0.05 ise fark bu veriyle
  gösterilememiştir. Hibrit ile yalnız-BERT arasında anlamlı fark çıkmazsa stilometrinin katkısı **doğruluk değil
  açıklanabilirliktir** (paneldeki göstergeler) — raporda böyle yazın.
- **A ile C arasındaki uçurum**, A'daki skorun ne kadarının v1'e özgü kalıplardan geldiğini gösterir.
- **"Zor" alt kümesi**, sınıflar arası farkın bilerek inceltildiği örneklerdir; gerçek dünyaya en yakın gösterge budur.
- `final.dogrulama_*` değerleri model seçiminde kullanılan parçadandır; **test skoru olarak raporlanmaz**.
- Tek tohumla çalışılır. Daha sağlam bir karşılaştırma için bkz. `YOL_HARITASI.md` ("çoklu tohum").

## Tekrarlanabilirlik

Tohum, sürümler, commit ve ayarlar `metrics.json → meta` altında kayıtlıdır. CPU'da aynı tohum aynı sonucu verir;
GPU'da küçük sayısal farklar olabilir. Örnek bazında tahminler `outputs/deney_onbellek/*/*__tahminler.csv`
dosyalarındadır (hata analizi için; git'e girmez).
