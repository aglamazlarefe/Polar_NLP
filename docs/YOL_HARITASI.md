# Yol haritası

Sıradaki işler, öncelik sırasıyla. Her madde tek başına yapılabilir; başlamadan önce [CLAUDE.md](../CLAUDE.md)
kurallarını okuyun. Bir işi bitirince bu dosyadan silin ve gerekiyorsa sonuçları yeniden üretin
(`docs/DENEYLER.md` → "Çalıştırma sırası").

## 0. İnce ayarlı çapraz doğrulamayı tamamlamak (önce bu)

**Neden:** `results/metrics.json`'daki BERT sonuçları dondurulmuş gömmelerle alındı (zaman kısıtı). Hibritin ince
ayarlı BERT'e karşı katkısı henüz ölçülmedi.

1. Mümkünse NVIDIA GPU'lu makinede (ör. RTX 3060) çalıştırın; CPU'da 4–7 saat sürer ve makinede başka ağır iş
   varsa çok daha uzun.
2. `KOMIT=0 DENEY_ARG="--ince-ayar" bash scripts/tum_hat.sh` — deney, nihai model, grafikler, rapor ve testler
   sırayla çalışır; durum `outputs/HAT_DURUM.txt` dosyasına yazılır. Yarıda kesilirse aynı komut kaldığı yerden sürer.
3. CPU'da 4 saati aşacağı öngörülürse kat sayısı kendiliğinden 3'e iner (`meta.not`).
4. `docs/SONUCLAR.md`'yi ve README'deki yorumları yeni sayılara göre gözden geçirin (sayıları elle yazmayın).

## 1. Gerçek veriyle doğrulama (en değerli adım)

**Neden:** Bütün sonuçlar sentetik veriyle elde edildi. Gerçek seyir defteri dili farklı olabilir.

1. Etik izin ve açık rıza olmadan gerçek personel metni toplamayın.
2. Kamuya açık kaynak arayın: yayımlanmış kutup seferi günlükleri, analog görev (Mars-500, SIRIUS, Concordia)
   çalışmalarının ekleri. Lisansı not edin.
3. 50–100 metni `docs/ETIKETLEME_KILAVUZU.md`'ye göre **iki insan** bağımsız etiketlesin; Cohen κ hesaplayın.
4. Bu veriyi eğitime katmadan önce yalnız **test** olarak kullanın: `scripts/deney.py`'ye "E: sentetikte eğit →
   gerçekte test" deneyi ekleyin (`deney_plani()` içine bir giriş; `Kaynak` sütununa `gercek` değeri).

## 2. Çoklu tohum

**Neden:** Küçük veride BERT ince ayarı tohumdan tohuma oynar; tek tohumla "hibrit daha iyi/kötü" denemez.

1. `scripts/deney.py`'ye `--tohumlar 42 43 44` seçeneği ekleyin; BERT koşularını her tohum için yineleyin
   (`EgitimAyari.seed`). Önbellek anahtarı ayarı içerdiği için dosyalar çakışmaz.
2. `model_ozeti()` içinde tohumlar arası ortalama ve standart sapmayı ayrı alanlarda saklayın.
3. `tests/test_sonuclar.py` şemasını güncelleyin. Süre üç katına çıkar; GPU'lu makinede çalıştırın.

## 3. Biçimbilimsel çözümleyiciyle stilometri

**Neden:** Düzenli ifadeler yanlış pozitif üretir (bkz. `MIMARI.md`).

1. `zeyrek` (saf Python) kurun; `stilometri.py`'deki `_birinci_tekil`, `_olumsuz`, `_edilgen` fonksiyonlarını
   çözümleyici çıktısındaki eklere (`A1sg`, `Neg`, `Pass`) bakacak biçimde yazın.
2. Hız: çözümleme yavaştır; kelime bazında `functools.lru_cache` kullanın.
3. `tests/test_stilometri.py`'deki örneklerin hepsi geçmeli; yanlış pozitif testlerine yenilerini ekleyin.
4. `stilometri_lr` skorunu önce/sonra karşılaştırın; iyileşme yoksa bağımlılığı eklemeyin.

## 4. Veriyi çeşitlendirme

- Tek cümlelik ve 60+ kelimelik kayıtlar (şu an hepsi 15–50 kelime).
- Karışık kayıtlar: ilk yarısı objektif, sonu yakınma. Bunlar için kılavuza kural ekleyin.
- Üslup dışı gürültü: yazım hataları, kısaltmalar, saat/koordinat yoğun kayıtlar.
- Hat: `docs/VERI_KARTI.md` → "Yeni veri eklemek". Üretici ve denetçi için **farklı** modeller kullanın.

## 5. Panel

- Kişi bazlı seyirde hareketli ortalama (3 gün) ve "art arda N gün eşik üstü" uyarısı. Eşik değerlerini
  `config.py`'ye koyun; bunların doğrulanmamış olduğunu arayüzde belirtin.
- Kelime katkıları için daha hızlı bir yöntem (gradyan × girdi); occlusion 60+ kelimede yavaşlar.
- Sonuçları PDF rapor olarak dışa aktarma.

## 6. Mühendislik

- GitHub Actions: `pytest` (tests klasörü internet/GPU gerektirmez; `tests/test_sonuclar.py` için `results/`
  depoda olmalı).
- `pyproject.toml` ile paketleme; `scripts/` içindeki `sys.path` satırları kaldırılabilir.
- Ağırlıkları Hugging Face Hub'a ya da Git LFS'e koyma (şu an `egitilmis_hybrid_model/` depoda değil; her
  kurulumda `egit_final.py` çalıştırılmalı).

## Yapılmaması gerekenler

- Test katına bakarak hiperparametre seçmek.
- v1 dosyasını (`data/polar_v1.csv`) "düzeltmek" — özgün veridir, karşılaştırmanın sabit noktasıdır.
- Sınıf ayıran kelimeleri stilometri sözlüğüne eklemek (ör. "tarafımca" zaten 1. tekil olarak sayılıyor; "anlamsız"
  eklemek ezber olur).
