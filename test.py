import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_yolu = "./egitilmis_berturk_modeli"
tokenizer = AutoTokenizer.from_pretrained(model_yolu)
model = AutoModelForSequenceClassification.from_pretrained(model_yolu)

def tahmin_et(metin):
    inputs = tokenizer(metin, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    sinif_id = torch.argmax(logits, dim=1).item()
    etiketler = ["Objektif Rapor", "Bilişsel Yorgunluk", "Kış Sendromu"]
    return etiketler[sinif_id]

# Örnek test
test_cumlesi = "Son zamanlarda kendimi çok bitkin hissediyorum ve sabahları uyanmakta zorlanıyorum."
print(f"Sonuç: {tahmin_et(test_cumlesi)}")