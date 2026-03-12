import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction

def compute_metrics(p: EvalPrediction):
    tahminler = np.argmax(p.predictions, axis=1)
    gercek_degerler = p.label_ids
    
    # Accuracy hesabı
    acc = accuracy_score(gercek_degerler, tahminler)
    # Sınıf dengesizliklerini hesaba katan macro ortalamalı metrikler
    precision, recall, f1, _ = precision_recall_fscore_support(gercek_degerler, tahminler, average='macro', zero_division=0)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # 1. Veri Yükleme ve Ön İşleme
    veri_dosyasi = "output_utf8.csv"  # Çıkartılan CSV veri setinin yolu
    df = pd.read_csv(veri_dosyasi)

    # Veriyi sınıflara göre dengeli (stratify) bir şekilde bölelim (%80 Train, %10 Validation, %10 Test)
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['Etiket'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['Etiket'])

    # Pandas DataFrame'lerini Hugging Face Dataset formatına çevirelim
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    # DatasetDict içerisinde birleştirelim
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    # 2. Tokenization
    model_ismi = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_ismi)

    def tokenize_function(ornekler):
        # Metinleri encode etme (padding ve truncation aktif, max_length=128 yeterli)
        return tokenizer(ornekler["Metin"], padding="max_length", truncation=True, max_length=128)

    # Tüm veri setlerine tokenization uygulayalım (batched=True işlemi hızlandırır)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Gerekli sütun düzenlemelerini yapalım: Trainer metin yerine 'labels' bekler
    tokenized_datasets = tokenized_datasets.rename_column("Etiket", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(["Metin"])
    tokenized_datasets.set_format("torch")

    # 4. Model Eğitimi (Training)
    # Önceden eğitilmiş (pre-trained) BERTurk modelini yükleyelim
    # Sınıf sayımız 3 (0: Objektif Rapor, 1: Bilişsel Yorgunluk, 2: Kış Sendromu)
    # 4. Model Eğitimi (Training)
    model = AutoModelForSequenceClassification.from_pretrained(model_ismi, num_labels=3)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        fp16=True, 
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard",
        logging_steps=10, # Virgül hatası düzeltildi
        # logging_dir artık burada gerekmiyor, report_to yeterli
    )

    # Trainer yapılandırmasını düzeltiyoruz
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],      # DÜZELTİLDİ: Tokenlanmış veri
        eval_dataset=tokenized_datasets["validation"], # DÜZELTİLDİ: Tokenlanmış veri
        processing_class=tokenizer,                    # transformers v4.47+ için processing_class
        compute_metrics=compute_metrics                # EKLENDİ: Metrik fonksiyonu
    )
    
    
    # Eğitimi (Fine-tuning) başlat
    print("Model eğitimi (Fine-Tuning) başlatılıyor...")
    trainer.train()

    # 5. Değerlendirme ve Kayıt
    print("\nEğitim tamamlandı. Test seti üzerinde değerlendirme yapılıyor...")

    # Test veri setinde tahmin yürütelim
    test_sonuclari = trainer.predict(tokenized_datasets["test"])
    tahminler = np.argmax(test_sonuclari.predictions, axis=1)
    gercek_etiketler = test_sonuclari.label_ids

    # Confusion Matrix (Karmaşıklık Matrisi)
    print("\nKarmaşıklık Matrisi (Confusion Matrix):")
    cm = confusion_matrix(gercek_etiketler, tahminler)
    print(cm)

    # Classification Report (Sınıflandırma Raporu)
    print("\nSınıflandırma Raporu:")
    hedef_isimleri = ["Objektif (0)", "Bilişsel Yorgunluk (1)", "Kış Sendromu (2)"]
    rapor = classification_report(gercek_etiketler, tahminler, target_names=hedef_isimleri)
    print(rapor)

    # Modelin ve Tokenizer'ın yerel diske kaydedilmesi
    kayit_yolu = "./egitilmis_berturk_modeli"
    print(f"\nModel ve Tokenizer '{kayit_yolu}' dizinine kaydediliyor...")
    trainer.save_model(kayit_yolu)
    tokenizer.save_pretrained(kayit_yolu)
    print("Tüm işlemler başarıyla tamamlandı!")

if __name__ == '__main__':
    main()
