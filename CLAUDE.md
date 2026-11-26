# Polar NLP — yapay zeka asistanları için çalışma kılavuzu

Antarktika üssü seyir defteri metinlerini üç **yazım üslubu** sınıfına ayıran model ve Streamlit paneli:
`0 Objektif Rapor` · `1 Bilişsel Yorgunluk` · `2 Kış Sendromu`. Model = BERTurk + kural tabanlı stilometri (hibrit).
Veri **sentetiktir**. Kod, yorumlar ve belgeler Türkçedir; kullanıcıyla Türkçe konuşun.

## Komutlar (Windows; her zaman `.venv` kullanın)

```bash
.venv\Scripts\python -m pytest                 # tüm testler, ~15 sn, internet/GPU gerekmez
.venv\Scripts\python scripts/deney.py --hizli  # deney hattı duman testi (~2 dk), results/'a dokunmaz
.venv\Scripts\python scripts/deney.py          # TAM deneyler -> results/metrics.json (CPU'da saatler sürer)
.venv\Scripts\python scripts/egit_final.py     # panelin modeli -> egitilmis_hybrid_model/ + results/demo_trend.json
.venv\Scripts\python scripts/gorseller.py      # results/*.png (yalnız JSON'dan)
.venv\Scripts\python scripts/rapor.py          # docs/SONUCLAR.md (yalnız JSON'dan)
.venv\Scripts\python scripts/veri_analiz.py    # veri sağlık raporu
.venv\Scripts\streamlit run dashboard.py       # panel
```

Kurulum `requirements.txt` başında yazıyor. Bash kullanıyorsanız yol `.venv/Scripts/python.exe`.

## Dosya haritası

| Yol | Ne işe yarar |
|---|---|
| `polar_nlp/config.py` | **Tek ayar kaynağı**: yollar, sınıf adları, `EgitimAyari`. Sabitleri başka yerde yeniden tanımlamayın. |
| `polar_nlp/stilometri.py` | Türkçe ek örüntüleriyle 10 özellik + `Olcekleyici`. |
| `polar_nlp/veri.py` | CSV doğrulama, konu-gruplu katlar, `PolarDataset`. |
| `polar_nlp/model.py` | `HybridPolarModel` (`stilo_kullan=False` → yalnız-BERT). Kaydet/yükle. |
| `polar_nlp/egitim.py` | Tek eğitim döngüsü `egit()`, `tahmin_et()`, `metrikler()`. |
| `polar_nlp/cikarim.py` | `PolarAnalizci` (modeli bir kez yükler), `polar_analiz_et()`. |
| `scripts/` | Çalıştırılabilir adımlar (yukarıdaki komutlar) + `veri_birlestir.py`. |
| `data/polar_v2.csv` | Ana veri (v1 + genişletme). `data/polar_v1.csv` özgün 300 örnek — **değiştirmeyin**. |
| `data/ham/` | Üretim ve kör etiket denetimi kayıtları (kaynak izlenebilirliği). |
| `results/` | `metrics.json`, `demo_trend.json`, grafikler. Git'e girer. |
| `egitilmis_hybrid_model/`, `outputs/` | Model ağırlıkları, önbellek. Git'e **girmez**. |
| `docs/` | `MIMARI`, `DENEYLER`, `VERI_KARTI`, `ETIKETLEME_KILAVUZU`, `SONUCLAR` (otomatik), `YOL_HARITASI`. |

## Değişmez kurallar

1. **Sonuç sayısı uydurmayın, kaynağa gömmeyin.** Raporlanan her sayı `results/metrics.json`'dan okunur; o dosyayı
   yalnız `scripts/deney.py` ve `scripts/egit_final.py` yazar. `docs/SONUCLAR.md` ve grafikler elle düzenlenmez,
   yeniden üretilir. `tests/test_sonuclar.py` bunu denetler.
2. **Sonuç kötü çıkarsa olduğu gibi raporlayın.** Hibritin kazanması için ayar kurcalamayın; test verisine bakarak
   seçim yapmayın (model seçimi yalnız doğrulama parçasıyla yapılır).
3. **Klinik iddia yazmayın.** "Tanı koyar", "KVKK uyumludur", "X gün önce tespit eder" gibi doğrulanmamış ifadeler
   yasak. Veri sentetik, sınıflar üslup kategorisi.
4. **Türkçe metinde `str.lower()` kullanmayın**; `polar_nlp.stilometri.tr_kucult` kullanın (I→ı, İ→i).
5. **Pickle yok.** Ağırlıklar `safetensors`, geri kalan her şey JSON.
6. **Veri sızıntısı yok.** Ölçekleyici ve her türlü istatistik yalnız eğitim parçasından hesaplanır; katlar konu
   bazında ayrılır (`veri.katlar`). `data/demo_gunlukler.csv` eğitime girmez.
7. İş bitmeden önce `pytest` çalıştırın; hepsi geçmeden "tamam" demeyin.
8. Commit yalnız kullanıcı isterse; `main`'e doğrudan yazmayın, **push etmeyin**.

## Sık yapılan işler

**Stilometri özelliği eklemek:** `stilometri.py`'de fonksiyonu yazıp `_OZELLIKLER` listesine ekleyin →
`tests/test_stilometri.py`'ye olumlu ve olumsuz örnek ekleyin → `egit_final.py` ile modeli yeniden eğitin (özellik
sayısı değiştiği için eski model yüklenmez, bilerek hata verir) → `deney.py`, `gorseller.py`, `rapor.py`.
Kelime listesini veri setine bakarak seçmeyin; gerekçesi literatürden gelmeli.

**Veri eklemek:** `docs/VERI_KARTI.md`'deki hattı izleyin (üret → kör denetim → `veri_birlestir.py`). Yeni metinleri
doğrudan `polar_v2.csv`'ye elle yapıştırmayın.

**Sınıf eklemek/değiştirmek:** `config.SINIF_ADLARI`, `SINIF_KISA`, `SINIF_RENKLERI`, `RISK_AGIRLIKLARI` birlikte
güncellenir; `docs/ETIKETLEME_KILAVUZU.md` yazılır; tüm veri yeniden etiketlenir; her şey yeniden eğitilir.

**Hiperparametre:** yalnız `config.EgitimAyari`. Değiştirirseniz deneyleri baştan çalıştırın (önbellek ayara göre
anahtarlanır, eski sonuç karışmaz).

## Bilinen tuzaklar

- Tam deney CPU'da uzun sürer; arka planda çalıştırın. Yarıda kesilirse aynı komut kaldığı yerden devam eder
  (`outputs/deney_onbellek/`).
- `AutoModel.from_pretrained` çıktısındaki "UNEXPECTED / MISSING" tablosu beklenen bir uyarıdır (dil modeli başlığı
  atılıyor, sınıflandırıcı sıfırdan başlıyor).
- Streamlit'te `use_container_width` kullanmayın; `width="stretch"` kullanın.
- Panelde model yolu `POLAR_MODEL_DIZINI`, sonuç klasörü `POLAR_SONUC_DIZINI` ortam değişkeniyle değiştirilebilir
  (testler bunu kullanır).
- `.vscode/launch.json` bir ekip üyesinin kişisel dosyasıdır; dokunmayın.

## Alt agent kullanımı

Alt agent'lar varsayılan olarak **Sonnet** ile çalışır (`.claude/agents/` altındakiler böyle tanımlıdır). Veri
üretimi gibi yaratıcı/zor işler dışında daha büyük model seçmeyin. Etiket denetimi yapan agent'a **asla** üretici
etiketlerini ya da `data/ham/denetim_anahtar.csv` dosyasını göstermeyin.
