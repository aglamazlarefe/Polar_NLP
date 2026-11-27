# Sonuçlar

> Bu dosya `python scripts/rapor.py` ile `results/metrics.json`'dan **otomatik üretilir. Elle düzenlemeyin.**

- Tarih: 2026-11-27 · commit `a29c936` · cihaz: cpu · süre: 1.4 dk
- Taban model: `dbmdz/bert-base-turkish-cased` · 5-kat konu-gruplu çapraz doğrulama · tohum 42
- Veri: 600 metin — Objektif Rapor: 202, Bilişsel Yorgunluk: 199, Kış Sendromu: 199 · kaynak v1: 300, v2: 300 · zorluk normal: 540, zor: 60
- BERT modelleri: BERTurk dondurulmuş: her metnin son katman ortalama havuzlamalı gömmesi bir kez çıkarılır; yalnız-BERT = gömme + lojistik regresyon, hibrit = gömme + stilometri + lojistik regresyon (C değeri eğitim katı içinde 3-kat CV ile seçilir).

## A_v1_cv — Özgün veri (v1) üzerinde 5-kat konu-gruplu çapraz doğrulama

Test örneği: 300

| Model | Macro-F1 | Doğruluk | Zor örnekler | Normal örnekler | v1 kısmı | v2 kısmı |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.884 ± 0.024 | 0.885 ± 0.025 | — | 0.887 (n=300) | 0.887 (n=300) | — |
| Yalnız stilometri | 0.862 ± 0.043 | 0.865 ± 0.041 | — | 0.863 (n=300) | 0.863 (n=300) | — |
| Yalnız BERTurk | 0.986 ± 0.016 | 0.986 ± 0.016 | — | 0.987 (n=300) | 0.987 (n=300) | — |
| Hibrit | 0.983 ± 0.017 | 0.984 ± 0.016 | — | 0.983 (n=300) | 0.983 (n=300) | — |

McNemar (Hibrit ↔ Yalnız BERTurk): yalnız hibritin doğru bildiği 0, yalnız diğerinin doğru bildiği 1 örnek; p = 1.0000

McNemar (Hibrit ↔ TF-IDF + LR): yalnız hibritin doğru bildiği 32, yalnız diğerinin doğru bildiği 3 örnek; p = 0.0000

Hibrit model — sınıf bazında (tüm test tahminleri birleştirilerek):

| Sınıf | Kesinlik | Duyarlılık | F1 | n |
|---|---|---|---|---|
| Objektif Rapor | 1.000 | 1.000 | 1.000 | 102 |
| Bilişsel Yorgunluk | 0.990 | 0.960 | 0.974 | 99 |
| Kış Sendromu | 0.961 | 0.990 | 0.975 | 99 |

Karışıklık matrisi (satır = gerçek, sütun = tahmin): `[[102, 0, 0], [0, 95, 4], [0, 1, 98]]`

## C_v1_v2 — v1'de eğit, v2'de test (dağılım dışı genelleme)

Test örneği: 300

| Model | Macro-F1 | Doğruluk | Zor örnekler | Normal örnekler | v1 kısmı | v2 kısmı |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.539 | 0.540 | 0.433 (n=60) | 0.567 (n=240) | — | 0.540 (n=300) |
| Yalnız stilometri | 0.466 | 0.473 | 0.567 (n=60) | 0.450 (n=240) | — | 0.473 (n=300) |
| Yalnız BERTurk | 0.749 | 0.747 | 0.683 (n=60) | 0.762 (n=240) | — | 0.747 (n=300) |
| Hibrit | 0.736 | 0.733 | 0.700 (n=60) | 0.742 (n=240) | — | 0.733 (n=300) |

McNemar (Hibrit ↔ Yalnız BERTurk): yalnız hibritin doğru bildiği 1, yalnız diğerinin doğru bildiği 5 örnek; p = 0.2188

McNemar (Hibrit ↔ TF-IDF + LR): yalnız hibritin doğru bildiği 90, yalnız diğerinin doğru bildiği 32 örnek; p = 0.0000

Hibrit model — sınıf bazında (tüm test tahminleri birleştirilerek):

| Sınıf | Kesinlik | Duyarlılık | F1 | n |
|---|---|---|---|---|
| Objektif Rapor | 0.934 | 0.710 | 0.807 | 100 |
| Bilişsel Yorgunluk | 0.669 | 0.830 | 0.741 | 100 |
| Kış Sendromu | 0.660 | 0.660 | 0.660 | 100 |

Karışıklık matrisi (satır = gerçek, sütun = tahmin): `[[71, 11, 18], [1, 83, 16], [4, 30, 66]]`

## D_v2_v1 — v2'de eğit, v1'de test

Test örneği: 300

| Model | Macro-F1 | Doğruluk | Zor örnekler | Normal örnekler | v1 kısmı | v2 kısmı |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.514 | 0.530 | — | 0.530 (n=300) | 0.530 (n=300) | — |
| Yalnız stilometri | 0.442 | 0.480 | — | 0.480 (n=300) | 0.480 (n=300) | — |
| Yalnız BERTurk | 0.783 | 0.800 | — | 0.800 (n=300) | 0.800 (n=300) | — |
| Hibrit | 0.770 | 0.790 | — | 0.790 (n=300) | 0.790 (n=300) | — |

McNemar (Hibrit ↔ Yalnız BERTurk): yalnız hibritin doğru bildiği 1, yalnız diğerinin doğru bildiği 4 örnek; p = 0.3750

McNemar (Hibrit ↔ TF-IDF + LR): yalnız hibritin doğru bildiği 100, yalnız diğerinin doğru bildiği 22 örnek; p = 0.0000

Hibrit model — sınıf bazında (tüm test tahminleri birleştirilerek):

| Sınıf | Kesinlik | Duyarlılık | F1 | n |
|---|---|---|---|---|
| Objektif Rapor | 0.759 | 0.990 | 0.860 | 102 |
| Bilişsel Yorgunluk | 0.769 | 0.909 | 0.833 | 99 |
| Kış Sendromu | 0.920 | 0.465 | 0.617 | 99 |

Karışıklık matrisi (satır = gerçek, sütun = tahmin): `[[101, 0, 1], [6, 90, 3], [26, 27, 46]]`

## B_tum_cv — Birleşik veri (v1+v2) üzerinde 5-kat konu-gruplu çapraz doğrulama

Test örneği: 600

| Model | Macro-F1 | Doğruluk | Zor örnekler | Normal örnekler | v1 kısmı | v2 kısmı |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.857 ± 0.036 | 0.857 ± 0.036 | 0.883 (n=60) | 0.854 (n=540) | 0.870 (n=300) | 0.843 (n=300) |
| Yalnız stilometri | 0.719 ± 0.029 | 0.722 ± 0.030 | 0.633 (n=60) | 0.730 (n=540) | 0.773 (n=300) | 0.667 (n=300) |
| Yalnız BERTurk | 0.969 ± 0.013 | 0.969 ± 0.012 | 0.933 (n=60) | 0.974 (n=540) | 0.977 (n=300) | 0.963 (n=300) |
| Hibrit | 0.968 ± 0.015 | 0.968 ± 0.015 | 0.933 (n=60) | 0.972 (n=540) | 0.973 (n=300) | 0.963 (n=300) |

McNemar (Hibrit ↔ Yalnız BERTurk): yalnız hibritin doğru bildiği 1, yalnız diğerinin doğru bildiği 2 örnek; p = 1.0000

McNemar (Hibrit ↔ TF-IDF + LR): yalnız hibritin doğru bildiği 79, yalnız diğerinin doğru bildiği 12 örnek; p = 0.0000

Hibrit model — sınıf bazında (tüm test tahminleri birleştirilerek):

| Sınıf | Kesinlik | Duyarlılık | F1 | n |
|---|---|---|---|---|
| Objektif Rapor | 0.952 | 0.985 | 0.968 | 202 |
| Bilişsel Yorgunluk | 0.980 | 0.970 | 0.975 | 199 |
| Kış Sendromu | 0.974 | 0.950 | 0.962 | 199 |

Karışıklık matrisi (satır = gerçek, sütun = tahmin): `[[199, 0, 3], [4, 193, 2], [6, 4, 189]]`

## Stilometrik özelliklerin sınıf ortalamaları (ham değer)

| Özellik | Objektif Rapor | Bilişsel Yorgunluk | Kış Sendromu |
|---|---|---|---|
| 1. tekil kişi oranı | 0.0106 | 0.0331 | 0.0205 |
| Olumsuzluk oranı | 0.0141 | 0.0359 | 0.0743 |
| Resmî bildirme (-DIr) oranı | 0.0189 | 0.0122 | 0.0492 |
| Edilgen çatı oranı | 0.1018 | 0.0388 | 0.0764 |
| Mutlakçı ifade oranı | 0.0017 | 0.0095 | 0.0044 |
| Nedensellik bağlacı oranı | 0.0050 | 0.0115 | 0.0122 |
| Ortalama cümle uzunluğu (kelime) | 9.0935 | 10.7091 | 9.9125 |
| Ortalama kelime uzunluğu (harf) | 6.8323 | 6.8094 | 6.8960 |
| Kelime çeşitliliği (TTR) | 0.9803 | 0.9818 | 0.9800 |
| Cümle sayısı | 2.9950 | 2.7186 | 2.8693 |

## Etiket denetimi (v2 üretimi)

- Üretilen metin: 300 · veri setine giren: 300
- Kör denetçi ile üretici etiketi uyumu: %100.0 (Cohen κ = 1.000)
- Uyuşmazlık: 0 · ikinci hakemin üretici etiketini onayladığı: 0 · atılan: 0
- Kalite süzgecinden atılan: uzunluk 0, yakın kopya 0
- Yalıtılmış ikinci denetçi (90 metinlik örneklem, zor örnekler ağırlıklı): uyum %100.0 (κ = 1.000); zor örneklerde %100.0
- Denetçiler de dil modelidir; üretici modelle benzer yanlılıklar taşıyabilirler. İnsan denetimi yapılmadı.

## Nihai model (panelde kullanılan)

- Eğitim 540 / doğrulama 60 örnek · seçilen epoch 2 · sıcaklık 1.20 · süre 11.3 dk
- Doğrulama macro-F1 0.829 — model seçiminde kullanıldığı için **test skoru değildir**.
- Kurgusal demo günlükleri (42 kayıt, eğitimde yok): senaryo tasarımıyla uyuşma %92.9
