import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import pickle
import json
import os
import re

# ==========================================================
# 1. STİLOMETRİ MODÜLÜ
# ==========================================================
def stilometri_cikar(metin: str) -> np.ndarray:
    metin_lower = metin.lower()
    kelimeler = re.findall(r'\b\w+\b', metin_lower)
    toplam_kelime = max(len(kelimeler), 1)

    # --- Özellik 1: birinci_tekil_zamir_orani ---
    # Stres altında "ben" kullanımı artar (Pennebaker, LIWC)
    birinci_tekil = {"ben", "bana", "benim", "bende", "benden"}
    f1 = sum(1 for k in kelimeler if k in birinci_tekil) / toplam_kelime

    # --- Özellik 2: TTR (Type-Token Ratio) ---
    # Bilişsel yorgunlukta kelime dağarcığı daralır
    f2 = len(set(kelimeler)) / toplam_kelime

    # --- Özellik 3: mutlak_ifade_orani ---
    # Esneklik kaybı ve görev reddinin dilsel göstergesi
    mutlak_ifadeler = {"asla", "hiçbir", "hiçbiri", "kesinlikle", "daima", "hiç", "tamamen"}
    mutlak_sayisi = sum(1 for k in kelimeler if k in mutlak_ifadeler)
    mutlak_sayisi += len(re.findall(r'\bher zaman\b', metin_lower))
    f3 = mutlak_sayisi / toplam_kelime

    # --- Özellik 4: ortalama_cumle_uzunlugu ---
    # Tükenmişlikte cümleler kısalır — veri setinde en ayırt edici özellik bu
    cumleler = re.split(r'[.!?]+', metin)
    cumle_sayisi = max(len([c for c in cumleler if c.strip()]), 1)
    f4 = toplam_kelime / cumle_sayisi

    # --- Özellik 5: olumsuzluk_orani ---
    # DEĞİŞTİRİLDİ: birinci_cogul_zamir yerine olumsuzluk orani
    # Gerekçe: veri setinde "biz" kullanımı tüm sınıflarda 0.0000 çıktı,
    # ayırt edici değildi. Olumsuzluk kelimeleri sınıf 2'de belirgin şekilde
    # daha yüksek → daha güçlü bir sinyal.
    olumsuzluk = {"değil", "yok", "hayır", "olmaz", "imkansız", "reddedildi",
                  "reddediyorum", "yapılmayacak", "uygulanmayacak", "iptal"}
    f5 = sum(1 for k in kelimeler if k in olumsuzluk) / toplam_kelime

    return np.array([f1, f2, f3, f4, f5], dtype=np.float32)


# ==========================================================
# DATASET CLASS
# ==========================================================
class PolarDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, scaler: MinMaxScaler, is_train=False):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.scaler = scaler

        raw_stylometry = np.array([stilometri_cikar(t) for t in self.texts])

        if is_train:
            self.stylometry_scaled = self.scaler.fit_transform(raw_stylometry)
        else:
            self.stylometry_scaled = self.scaler.transform(raw_stylometry)

        self.stylometry_scaled = self.stylometry_scaled.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        stilo = self.stylometry_scaled[idx]

        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'stylometry': torch.tensor(stilo),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ==========================================================
# 2. HYBRID POLAR MODEL MİMARİSİ
# ==========================================================
class HybridPolarModel(nn.Module):
    def __init__(self, model_name="dbmdz/bert-base-turkish-cased", num_labels=3):
        super(HybridPolarModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # DEĞİŞTİRİLDİ: Dropout 0.3 → 0.2
        # Gerekçe: 300 örneklik küçük veri setinde 0.3 çok fazla bilgi
        # kaybettiriyor, model yeterince öğrenemiyor.
        self.classifier = nn.Sequential(
            nn.Linear(768 + 5, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, stylometry):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        fusion = torch.cat([cls_output, stylometry], dim=1)
        logits = self.classifier(fusion)
        return logits


# ==========================================================
# İNFERENCE FONKSİYONU
# ==========================================================
def polar_analiz_et(metin: str, model_dir: str) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    model = HybridPolarModel("dbmdz/bert-base-turkish-cased", num_labels=3)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "pytorch_model.bin"),
        map_location=device,
        weights_only=True
    ))
    model.to(device)
    model.eval()

    raw_stilo = stilometri_cikar(metin)
    scaled_stilo = scaler.transform([raw_stilo])[0]

    encoding = tokenizer(
        metin,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    stilo_tensor = torch.tensor(scaled_stilo, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, stilo_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    sinif_isimleri = {0: "Objektif Rapor", 1: "Bilişsel Yorgunluk", 2: "Kış Sendromu"}

    ttr          = float(scaled_stilo[1])
    norm_cumle   = float(scaled_stilo[3])
    norm_mutlak  = float(scaled_stilo[2])
    norm_tekil   = float(scaled_stilo[0])
    norm_olumsuz = float(scaled_stilo[4])  # olumsuzluk_orani (eski: cogul zamir)

    bilissel_yorgunluk_skoru = (1.0 - ttr) * 0.6 + (1.0 - norm_cumle) * 0.4
    sosyal_cekilme_skoru     = float(norm_olumsuz)   # olumsuzluk → sosyal çekilme proxy
    gorev_reddi_skoru        = float(norm_mutlak)
    oz_odaklanma_skoru       = float(norm_tekil)

    # --- ÇELIŞKI ÇÖZÜMÜ ---
    # Eğer model güven skoru düşükse (< 0.50) ve risk profili yüksek risk
    # işaretliyorsa, risk profiline göre sınıfı override et.
    # Gerekçe: 300 örnekle eğitilen modelde BERT kısmı bazen stilometrik
    # sinyallerin aksine karar veriyor; bu durumda stilometri daha güvenilir.
    ortalama_risk = (bilissel_yorgunluk_skoru + sosyal_cekilme_skoru +
                     gorev_reddi_skoru + oz_odaklanma_skoru) / 4.0

    if confidence < 0.50 and ortalama_risk > 0.55:
        if gorev_reddi_skoru > 0.6 or sosyal_cekilme_skoru > 0.6:
            pred_class = 2  # Kış Sendromu
        else:
            pred_class = 1  # Bilişsel Yorgunluk
        confidence = float(probs[pred_class])

    result = {
        "sinif": pred_class,
        "sinif_adi": sinif_isimleri[pred_class],
        "guven_skoru": float(np.round(confidence, 4)),
        "risk_profili": {
            "bilissel_yorgunluk": float(np.round(np.clip(bilissel_yorgunluk_skoru, 0.0, 1.0), 4)),
            "sosyal_cekilme":     float(np.round(np.clip(sosyal_cekilme_skoru,     0.0, 1.0), 4)),
            "gorev_reddi":        float(np.round(np.clip(gorev_reddi_skoru,        0.0, 1.0), 4)),
            "oz_odaklanma":       float(np.round(np.clip(oz_odaklanma_skoru,       0.0, 1.0), 4))
        },
        "stilometrik_ozellikler": {
            "birinci_tekil_zamir": float(np.round(raw_stilo[0], 4)),
            "ttr":                 float(np.round(raw_stilo[1], 4)),
            "mutlak_ifade":        float(np.round(raw_stilo[2], 4)),
            "cumle_uzunlugu":      float(np.round(raw_stilo[3], 4)),
            "olumsuzluk_orani":    float(np.round(raw_stilo[4], 4))
        }
    }

    return result


# ==========================================================
# 3. YÜKLEME VE CUSTOM TRAINING LOOP
# ==========================================================
def main():
    print("Veri Yükleme ve Ön İşleme başlatılıyor...")
    df = pd.read_csv("output_utf8.csv")

    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['Etiket']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['Etiket']
    )
    print(f"Bölünme -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    model_ismi = "dbmdz/bert-base-turkish-cased"
    tokenizer  = AutoTokenizer.from_pretrained(model_ismi)
    scaler     = MinMaxScaler()

    train_dataset = PolarDataset(train_df['Metin'], train_df['Etiket'], tokenizer, scaler, is_train=True)
    val_dataset   = PolarDataset(val_df['Metin'],   val_df['Etiket'],   tokenizer, scaler, is_train=False)
    test_dataset  = PolarDataset(test_df['Metin'],  test_df['Etiket'],  tokenizer, scaler, is_train=False)

    # Stilometrik özellik analizi tablosu
    raw_stilos = np.array([stilometri_cikar(t) for t in df['Metin']])
    labels_arr = df['Etiket'].values
    print("\n[ARTIFACT 3] --- STİLOMETRİK ÖZELLİK ANALİZİ (Tüm Veri Kümesi) ---")
    stilo_names = [
        "birinci_tekil_zamir_orani",
        "ttr",
        "mutlak_ifade_orani",
        "ortalama_cumle_uzunlugu",
        "olumsuzluk_orani"          # güncellendi
    ]
    class_names = ["Sınıf 0 (Obj.)", "Sınıf 1 (Yorg.)", "Sınıf 2 (Kış)"]
    print(f"{'Özellik':<30} | {class_names[0]:<15} | {class_names[1]:<15} | {class_names[2]:<15}")
    for i, fname in enumerate(stilo_names):
        avg_0 = np.mean(raw_stilos[labels_arr == 0][:, i])
        avg_1 = np.mean(raw_stilos[labels_arr == 1][:, i])
        avg_2 = np.mean(raw_stilos[labels_arr == 2][:, i])
        print(f"{fname:<30} | {avg_0:<15.4f} | {avg_1:<15.4f} | {avg_2:<15.4f}")
    print("--------------------------------------------------------------------------\n")

    batch_size   = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    model = HybridPolarModel(model_name=model_ismi, num_labels=3)
    model.to(device)

    # DEĞİŞTİRİLDİ: Epoch 4 → 8
    # Gerekçe: Val F1 hâlâ epoch 4'te 0.61 ile artıyordu, erken duruyordu.
    epochs = 8

    # DEĞİŞTİRİLDİ: lr 2e-5 → 3e-5
    # Gerekçe: 300 örneklik küçük veri setinde biraz daha agresif öğrenme
    # convergence'ı hızlandırır.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # DEĞİŞTİRİLDİ: class_weights eklendi
    # Gerekçe: Kış Sendromu (sınıf 2) recall 0.10 ile neredeyse öğrenilmemişti.
    # 2.5x ağırlık vererek modeli bu sınıfa daha fazla dikkat etmeye zorluyoruz.
    class_weights = torch.tensor([1.0, 1.0, 2.5], dtype=torch.float32).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    scaler_amp      = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    best_f1         = 0.0
    best_model_state = None

    print("[ARTIFACT 1] --- TERMİNAL LOGU ---")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_loader):
            b_input_ids      = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_stylometry     = batch['stylometry'].to(device)
            b_labels         = batch['label'].to(device)

            try:
                with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    logits = model(b_input_ids, b_attention_mask, b_stylometry)
                    loss   = criterion(logits, b_labels)
                    loss   = loss / 2  # gradient_accumulation_steps = 2

                scaler_amp.scale(loss).backward()

                if (step + 1) % 2 == 0 or (step + 1) == len(train_loader):
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                    scheduler.step()
                    optimizer.zero_grad()
            except Exception as e:
                print(f"Eğitim sırasında hata: {e}")

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_true = [], []

        for batch in val_loader:
            b_input_ids      = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_stylometry     = batch['stylometry'].to(device)
            b_labels         = batch['label'].to(device)

            try:
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                        logits = model(b_input_ids, b_attention_mask, b_stylometry)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(b_labels.cpu().numpy())
            except Exception as e:
                print(f"Validation sırasında hata: {e}")

        val_acc = accuracy_score(val_true, val_preds)
        _, _, val_f1, _ = precision_recall_fscore_support(
            val_true, val_preds, average='macro', zero_division=0
        )
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 >= best_f1:
            best_f1          = val_f1
            best_model_state = model.state_dict().copy()

    print("----------------------------------\n")
    print("Eğitim tamamlandı. Test seti üzerinde değerlendirme yapılıyor...\n")

    model.load_state_dict(best_model_state)
    model.eval()

    test_preds, test_true = [], []
    for batch in test_loader:
        b_input_ids      = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        b_stylometry     = batch['stylometry'].to(device)
        b_labels         = batch['label'].to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                logits = model(b_input_ids, b_attention_mask, b_stylometry)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_true.extend(b_labels.cpu().numpy())

    print("[ARTIFACT 2] --- TEST SETİ RAPORU ---")
    cm = confusion_matrix(test_true, test_preds)
    print("Confusion Matrix:")
    print(cm)

    hedef_isimleri = ["Objektif (0)", "Bilişsel Yorgunluk (1)", "Kış Sendromu (2)"]
    rapor = classification_report(test_true, test_preds, target_names=hedef_isimleri, zero_division=0)
    print("\nClassification Report:")
    print(rapor)

    test_acc = accuracy_score(test_true, test_preds)
    _, _, test_mac_f1, _ = precision_recall_fscore_support(
        test_true, test_preds, average='macro', zero_division=0
    )
    print(f"Genel Accuracy: {test_acc:.4f} | Macro F1: {test_mac_f1:.4f}")
    print("----------------------------------\n")

    # Model kaydetme
    kayit_yolu = "./egitilmis_hybrid_model"
    os.makedirs(kayit_yolu, exist_ok=True)
    os.makedirs(os.path.join(kayit_yolu, "tokenizer"), exist_ok=True)

    print(f"Model, Tokenizer ve Scaler '{kayit_yolu}' dizinine kaydediliyor...")
    torch.save(best_model_state, os.path.join(kayit_yolu, "pytorch_model.bin"))
    model.bert.config.to_json_file(os.path.join(kayit_yolu, "config.json"))
    tokenizer.save_pretrained(os.path.join(kayit_yolu, "tokenizer"))

    with open(os.path.join(kayit_yolu, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(kayit_yolu, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_type":      "HybridPolarModel",
            "base_model":      model_ismi,
            "best_val_f1":     best_f1,
            "test_accuracy":   test_acc,
            "test_macro_f1":   test_mac_f1,
            "epochs_run":      epochs,
            "class_weights":   [1.0, 1.0, 2.5],
            "learning_rate":   3e-5,
            "dropout":         0.2
        }, f, indent=4)

    print("İşlem başarıyla tamamlandı. Oluşturulan dosyalar için ./egitilmis_hybrid_model/ dizinini kontrol edebilirsiniz.")

    # Inference testi
    test_metni = "Her zaman anten kulesindeki kabloları kontrol ediyorum. Asla merkeze sinyal gitmiyor."
    print("\n[Inference / polar_analiz_et() TESTİ]")
    print("Test Metni:", test_metni)
    sonuc = polar_analiz_et(test_metni, kayit_yolu)
    print(json.dumps(sonuc, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()