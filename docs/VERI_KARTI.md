# Veri kartı

Sayısal özetler (sınıf dağılımı, denetim uyumu, atılan kayıtlar) [SONUCLAR.md](SONUCLAR.md) dosyasında ve
`data/ham/denetim_ozeti.json` içindedir; burada yöntem anlatılır.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `data/polar_v1.csv` | Özgün 300 metin (`Metin, Etiket`). `data/kaynak/seyir_defteri_veri_seti.docx`'ten `scripts/docx_to_csv.py` ile çıkarıldı. **Değiştirilmez.** |
| `data/polar_v2.csv` | Ana veri: v1 + genişletme. Sütunlar `Metin, Etiket, Konu, Kaynak (v1/v2), Zorluk (normal/zor), Senaryo`. |
| `data/demo_gunlukler.csv` | 3 kurgusal personel × 14 gün (`Personel, Tarih, Metin, Beklenen`). Yalnız panel demosu; **eğitime girmez**. |
| `data/ham/v1_konu.csv` | v1 metinlerine atanan konu (grup bazlı çapraz doğrulama için). |
| `data/ham/uretim_[A-D].jsonl` | Genişletme metinleri, üretici etiketleriyle. |
| `data/ham/denetim_*.csv`, `denetim_anahtar.csv`, `denetim_ozeti.json` | Kör etiket denetimi kayıtları. |

## Etiketler

`0 Objektif Rapor`, `1 Bilişsel Yorgunluk`, `2 Kış Sendromu` — tanımlar ve sınır vakalar
[ETIKETLEME_KILAVUZU.md](ETIKETLEME_KILAVUZU.md). Bunlar **yazım üslubu** kategorileridir, klinik tanı değildir.

## v1 (özgün veri)

Proje ekibi tarafından hazırlanmış sentetik metinler. Bilinen sorunları (`scripts/veri_analiz.py data/polar_v1.csv`):

- Kalıp kelimeler: bazı kelimeler neredeyse yalnız bir sınıfta geçer (ör. "tarafımca", "iptal" → sınıf 2;
  "sağlandı", "tespit" → sınıf 0). Basit bir kelime sayım modeli bu yüzden yüksek skor alabilir.
- Sınıflar arasında cümle uzunluğu farkı vardır (sınıf 1 daha uzun cümleler kurar).
- Satırlar 0,1,2,0,1,2… sırasıyla dizilidir (eğitimde karıştırıldığı için sorun değildir).

## v2 (genişletme) — nasıl üretildi

1. **Kılavuz.** Üç sınıfın işlemsel tanımı v1 örneklerinden çıkarılıp yazıldı.
2. **Üretim.** Dört paralel dil modeli oturumu (Claude Opus), her biri farklı bir konu alanında 25 senaryo yazdı
   (A enerji/mekanik, B iletişim/bilişim, C bilimsel ölçüm, D yaşam/lojistik). **Üçlü tasarım:** her senaryo tek bir
   somut olaydır ve aynı olay üç tutumla (0/1/2) yazılır; böylece konu sınıfı ele vermez.
3. **Kısıtlar.** 2–4 cümle, 15–50 kelime; bir üçlünün metinleri yakın uzunlukta (en uzun/en kısa ≤ 1.3) ve en uzun
   metnin sınıfı senaryodan senaryoya değişir; anlatım kipi (edilgen, 1. tekil, 1. çoğul, geniş zaman) her sınıfta
   bulunur; v1'deki kalıp kelimeler kendi sınıflarında kullanılmaz, mümkünse diğer sınıflarda kullanılır. Senaryoların
   beşte biri "zor": sınıflar arası fark bilerek inceltilmiştir (çözülemeyen arızanın nötr raporu, şikâyet kelimesi
   olmayan zihinsel zorlanma, kibar dille örtük bırakma).
4. **Kör denetim.** Metinler karıştırılıp maskeli kimliklerle (`k000`…) etiketsiz olarak bir denetçi modele (Claude
   Sonnet, `.claude/agents/etiket-denetcisi.md`) verildi; denetçi yalnız kılavuzu gördü. Uyuşmazlıklar ikinci bir
   kör hakeme gidecek, o da üreticiyle uyuşmazsa kayıt atılacaktı (`scripts/veri_birlestir.py`).
5. **Yalıtılmış ikinci denetim.** İlk denetimin uyumu tam çıktığı ve denetçinin dosya erişimi kayıttan
   doğrulanamadığı için, zor örnekler ağırlıklı bir örneklem proje dışındaki bir klasöre kopyalanıp (yalnız metin
   + kılavuz) başka bir denetçi oturumuna verildi. Sonucu `denetim_ozeti.json → yalitilmis_orneklem_denetimi`.
6. **Programatik süzgeç.** Uzunluk sınırı dışındakiler ve v1 dahil yakın kopyalar (karakter 5-gram Jaccard > 0.6)
   atılır.
7. **Konu ataması (v1).** v1 metinlerine aynı 17 konuluk listeden konu bir dil modeli oturumuyla atandı.

## Sınırlılıklar

- **Tüm veri sentetiktir.** Gerçek kutup personelinin dili daha dağınık, daha kısa ve daha az "edebî" olabilir.
- Üretici ve denetçiler aynı model ailesindendir; ortak yanlılıklar denetimde görünmez. İnsan denetimi yapılmadı.
- Üçlü tasarım sınıfları dengeli kılar ama gerçek dağılımda objektif raporlar çok daha yaygındır.
- v2'de cümle sayısı neredeyse hep 3'tür; v1 ile v2 arasında uzunluk farkı vardır (v2 daha uzun). Kaynaklar
  arası deneylerde (C, D) bu fark da etkilidir.
- Bir üretici raporuna göre "-medik/-madık" ile biten cümleler sınıf 2'de yoğunlaşıyor; bu, işi bırakma tutumunun
  doğal sonucu ama modelin ipucu olarak kullanabileceği bir örüntüdür.

## Yeni veri eklemek

1. `data/ham/uretim_<harf>.jsonl` biçiminde yazın (alanlar: `id, senaryo, konu, metin, etiket, zorluk`;
   `konu` `scripts/veri_birlestir.py → KONULAR` listesinden).
2. `python scripts/veri_birlestir.py hazirla` → etiketsiz `denetim_girdi.jsonl`.
3. `etiket-denetcisi` agent'ına girdiyi verin → `denetim_1.csv` (`kimlik,etiket`).
4. `python scripts/veri_birlestir.py uyusmazlik` → varsa `denetim_2_girdi.jsonl`; ikinci kör hakem → `denetim_2.csv`.
5. `python scripts/veri_birlestir.py birlestir` → `data/polar_v2.csv`, `denetim_ozeti.json`.
6. `python scripts/veri_analiz.py` ile sızıntıyı kontrol edin, sonra deneyleri baştan çalıştırın.
