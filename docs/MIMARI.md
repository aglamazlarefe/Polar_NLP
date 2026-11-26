# Mimari

## Veri akışı

```
data/kaynak/*.docx ──docx_to_csv──▶ data/polar_v1.csv ─┐
data/ham/uretim_*.jsonl ──kör denetim──veri_birlestir──┴▶ data/polar_v2.csv
                                                            │
                      ┌─────────────────────────────────────┤
                      ▼                                     ▼
            scripts/deney.py                       scripts/egit_final.py
     (4 model × 4 deney, çapraz doğrulama)      (nihai hibrit model, %90/%10)
                      │                                     │
                      ▼                                     ▼
            results/metrics.json  ◀── "final" ──   egitilmis_hybrid_model/
                      │                            results/demo_trend.json
          ┌───────────┼───────────┐                         │
          ▼           ▼           ▼                         ▼
   gorseller.py   rapor.py   dashboard.py  ◀──────── PolarAnalizci
   results/*.png  docs/SONUCLAR.md
```

Her ok tek yönlüdür: sayılar yalnız eğitim scriptlerinden çıkar, görselleştirme ve panel yalnız okur.

## Model (`polar_nlp/model.py`)

```
metin ─▶ BERTurk (dbmdz/bert-base-turkish-cased) ─▶ [CLS] vektörü (768)
metin ─▶ stilometri (10 sayı) ─▶ ölçekleme ─▶ Linear(10→32) + ReLU + LayerNorm (32)
                                   birleştir (800) ─▶ Dropout ─▶ Linear(800→256) ─▶ ReLU ─▶ Dropout ─▶ Linear(256→3)
```

- Stilometri kendi küçük katmanından geçer; 768 boyutlu vektörün yanında 10 ham sayı kolayca yok sayılırdı.
- `stilo_kullan=False` aynı sınıfı yalnız-BERT modeline çevirir. Karşılaştırma (ablation) böyle yapılır; iki model
  arasındaki tek fark stilometri dalıdır.
- Kayıt biçimi: `model.safetensors`, `config.json` (BERT yapılandırması), `tokenizer/`, `olcekleyici.json`,
  `model_info.json`. Yükleme yalnız yerel dosyalarla yapılır, internet gerekmez.

## Eğitim (`polar_nlp/egitim.py`)

- AdamW; BERT gövdesi `lr_bert`, rastgele başlatılan katmanlar `lr_bas` (daha yüksek) alır.
- Doğrusal ısınma + doğrusal azalma; her parti bir optimizer adımı. Gradyan kırpma 1.0.
- Her epoch sonunda doğrulama macro-F1 ölçülür; eşitlikte düşük doğrulama kaybı kazanır. En iyi epoch'un
  ağırlıkları **derin kopya** ile saklanır, `sabir` epoch iyileşme olmazsa durulur.
- Karma duyarlık (AMP) yalnız CUDA'da açılır. CPU'da dinamik doldurma sayesinde epoch başına süre kısadır.
- `tohumla()` ile tekrarlanabilir; aynı tohum aynı sonucu verir (CPU'da).

## Stilometri (`polar_nlp/stilometri.py`)

| Özellik | Nasıl ölçülür | Gerekçe |
|---|---|---|
| `birinci_tekil` | zamirler (ben, bana…) + fiil ekleri (-yorum, -dım, -acağım, -dığım) + çoğul iyelik (-lerim) | LIWC "I" kategorisi; Türkçe'de kişi çoğunlukla ekle verilir |
| `olumsuzluk` | değil/yok + -mA- ekli fiiller (-mıyor, -madı, -mayacak, -maz…) + -sIz | LIWC "negate" |
| `resmi_bildirme` | -DIr ile biten sözcükler (tamamlanmıştır, gereksizdir) | resmî/mesafeli üslup |
| `edilgen` | -Il-, -lAn-, l+-In- gövdeli fiiller (yapıldı, temizlendi, alındı) | öznesiz, kişisiz anlatım |
| `mutlakci` | asla, hiç, hep, tamamen, her zaman… | Al-Mosaiwi & Johnstone (2018), mutlakçı dil |
| `nedensellik` | yüzünden, nedeniyle, rağmen… + -dığından | LIWC "cause"; gerekçelendirme ve suçlama |
| `ort_cumle_uzunlugu`, `ort_kelime_uzunlugu`, `ttr`, `cumle_sayisi` | sayım | klasik stilometri |

**Sınırlılık:** Bunlar düzenli ifadelerdir, biçimbilimsel çözümleme değildir. Bilinen hata türleri: `-DIm` ile biten
adlar (istisna listesiyle azaltıldı: yardım, eğitim, yalıtım…), `-sIz` ile biten ama olumsuz olmayan sözcükler
("telsiz" hariç tutulur), etken `-lAn-` fiilleri ("kullandı", "odaklandı" hariç tutulur). Daha doğru ölçüm için
Zemberek/zeyrek gibi bir çözümleyiciye geçilebilir (bkz. `YOL_HARITASI.md`).

Özellik sözlükleri veri setine bakılarak seçilmedi; amaç, sınıfı ele veren kelimeleri ezberletmek değil, üslubu
ölçmektir. v2 verisini üreten modellere bu özellik listesi gösterilmedi.

## Çıkarım (`polar_nlp/cikarim.py`)

`PolarAnalizci.analiz_et(metinler)` her metin için şunu döndürür:

| Alan | Anlamı |
|---|---|
| `sinif`, `sinif_adi`, `guven_skoru` | en olası sınıf ve olasılığı |
| `olasiliklar` | üç sınıfın olasılığı (sıcaklık ölçeklemesinden sonra) |
| `risk_skoru` | `0.5·P(Bilişsel Yorgunluk) + 1.0·P(Kış Sendromu)` — tek sayılık özet, 0–1 |
| `belirsiz` | en yüksek olasılık `BELIRSIZLIK_ESIGI` (0.60) altındaysa `True` |
| `stilometri` | her özelliğin ham değeri ve eğitimdeki objektif raporlara göre z-skoru |

`kelime_katkilari(metin)` her kelimeyi tek tek çıkarıp tahmini yineler (occlusion) ve olasılık düşüşünü verir.

Eski sürümdeki dört boyutlu "risk profili" (ör. sosyal çekilme = olumsuzluk oranı) ve "güven < 0.5 ise sınıfı
değiştir" kuralı kaldırıldı: ikisi de doğrulanmamış sezgisel eşlemelerdi.

## Değerlendirme tasarımı (`scripts/deney.py`)

- **Konu-gruplu katlar:** aynı konu (ör. `soguk_depolama`) hem eğitimde hem testte bulunmaz. Model konuya özgü
  kalıpları ezberleyerek puan alamaz.
- **İç doğrulama:** her katta eğitim kısmının %15'i model seçimi için ayrılır; test katına yalnız bir kez bakılır.
- **Dört model:** TF-IDF + lojistik regresyon (kelime ezberi taban çizgisi), yalnız stilometri, yalnız BERT, hibrit.
- **C ve D deneyleri** veri kaynakları arasında genellemeyi ölçer; v1'deki yüksek skorun veri setine özgü
  kalıplardan mı geldiği buradan anlaşılır.
- **McNemar testi:** hibrit ile yalnız-BERT'in aynı örneklerdeki doğru/yanlışları karşılaştırılır.
