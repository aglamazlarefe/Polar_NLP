"""Eğitim döngüsü, tahmin ve metrikler. Tüm deneyler bu tek döngüyü kullanır."""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import N_SINIF, EgitimAyari
from .veri import PolarDataset, harmanlayici


def tohumla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cihaz_sec() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrikler(gercek, tahmin) -> dict:
    etiketler = list(range(N_SINIF))
    p, r, f, destek = precision_recall_fscore_support(gercek, tahmin, labels=etiketler, zero_division=0)
    return {
        "accuracy": float(accuracy_score(gercek, tahmin)),
        "macro_f1": float(f.mean()),
        "sinif": {
            str(i): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "destek": int(destek[i])}
            for i in etiketler
        },
        "karisiklik_matrisi": confusion_matrix(gercek, tahmin, labels=etiketler).tolist(),
    }


@torch.no_grad()
def tahmin_et(model, tokenizer, veri: PolarDataset, cihaz=None, batch: int = 32) -> np.ndarray:
    """Logit matrisi döndürür (n, n_sinif)."""
    cihaz = cihaz or cihaz_sec()
    model.eval()
    yukleyici = DataLoader(veri, batch_size=batch, shuffle=False, collate_fn=harmanlayici(tokenizer))
    parcalar = []
    for parti in yukleyici:
        logit = model(parti["input_ids"].to(cihaz), parti["attention_mask"].to(cihaz), parti["stilometri"].to(cihaz))
        parcalar.append(logit.float().cpu())
    if not parcalar:
        return np.zeros((0, N_SINIF), dtype=np.float32)
    return torch.cat(parcalar).numpy()


def _kayip(logit: np.ndarray, etiket) -> float:
    return float(nn.functional.cross_entropy(torch.from_numpy(logit), torch.tensor(etiket)).item())


def egit(model, tokenizer, egitim: PolarDataset, dogrulama: PolarDataset,
         ayar: EgitimAyari | None = None, cihaz=None, gunluk=print) -> dict:
    """Modeli eğitir; doğrulama macro-F1'ine göre en iyi epoch'un ağırlıklarını döndürür.

    Dönen sözlük: `en_iyi_durum` (state_dict kopyası), `en_iyi_epoch`, `gecmis`, `sure_sn`.
    Model nesnesi en iyi ağırlıklarla bırakılır.
    """
    ayar = ayar or EgitimAyari()
    cihaz = cihaz or cihaz_sec()
    tohumla(ayar.seed)
    model.to(cihaz)

    uretec = torch.Generator().manual_seed(ayar.seed)
    yukleyici = DataLoader(egitim, batch_size=ayar.batch, shuffle=True, generator=uretec,
                           collate_fn=harmanlayici(tokenizer))

    bas = model.bas_parametreleri()
    bas_kimlik = {id(p) for p in bas}
    govde = [p for p in model.parameters() if id(p) not in bas_kimlik]
    optimizer = torch.optim.AdamW(
        [{"params": govde, "lr": ayar.lr_bert}, {"params": bas, "lr": ayar.lr_bas}],
        weight_decay=ayar.agirlik_curumesi,
    )
    toplam_adim = len(yukleyici) * ayar.epoch          # her parti bir optimizer adımı
    zamanlayici = get_linear_schedule_with_warmup(optimizer, int(toplam_adim * ayar.isinma_orani), toplam_adim)

    agirlik = None
    if ayar.sinif_agirligi:
        siniflar = np.arange(N_SINIF)
        agirlik = torch.tensor(compute_class_weight("balanced", classes=siniflar, y=np.array(egitim.etiketler)),
                               dtype=torch.float32, device=cihaz)
    olcut = nn.CrossEntropyLoss(weight=agirlik)

    amp = cihaz.type == "cuda"
    olcekleyici_amp = torch.amp.GradScaler("cuda", enabled=amp)

    en_iyi_puan, en_iyi_durum, en_iyi_epoch, sabirsiz = None, None, 0, 0
    gecmis, baslangic = [], time.time()

    for epoch in range(1, ayar.epoch + 1):
        model.train()
        toplam_kayip = 0.0
        for parti in yukleyici:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=cihaz.type, enabled=amp):
                logit = model(parti["input_ids"].to(cihaz), parti["attention_mask"].to(cihaz),
                              parti["stilometri"].to(cihaz))
                kayip = olcut(logit.float(), parti["etiket"].to(cihaz))
            olcekleyici_amp.scale(kayip).backward()
            olcekleyici_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), ayar.grad_kirpma)
            olcekleyici_amp.step(optimizer)
            olcekleyici_amp.update()
            zamanlayici.step()
            toplam_kayip += kayip.item() * len(parti["etiket"])

        dog_logit = tahmin_et(model, tokenizer, dogrulama, cihaz)
        dog = metrikler(dogrulama.etiketler, dog_logit.argmax(1))
        dog_kayip = _kayip(dog_logit, dogrulama.etiketler)
        gecmis.append({
            "epoch": epoch,
            "egitim_kaybi": toplam_kayip / len(egitim),
            "dogrulama_kaybi": dog_kayip,
            "dogrulama_f1": dog["macro_f1"],
            "dogrulama_acc": dog["accuracy"],
            "lr": zamanlayici.get_last_lr()[0],
        })
        gunluk(f"  [epoch {epoch}/{ayar.epoch}] kayıp {gecmis[-1]['egitim_kaybi']:.4f} | "
               f"doğ. kayıp {dog_kayip:.4f} | doğ. F1 {dog['macro_f1']:.4f} | doğ. acc {dog['accuracy']:.4f}")

        puan = (round(dog["macro_f1"], 6), -dog_kayip)   # eşitlikte düşük kayıp kazanır
        if en_iyi_puan is None or puan > en_iyi_puan:
            en_iyi_puan, en_iyi_epoch, sabirsiz = puan, epoch, 0
            # Derin kopya şart: state_dict() canlı tensörlere bakar, sığ kopya sonraki
            # epoch'larla birlikte değişir.
            en_iyi_durum = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            sabirsiz += 1
            if ayar.sabir is not None and sabirsiz >= ayar.sabir:
                gunluk(f"  erken durdurma (en iyi epoch: {en_iyi_epoch})")
                break

    model.load_state_dict(en_iyi_durum)
    model.eval()
    return {"en_iyi_durum": en_iyi_durum, "en_iyi_epoch": en_iyi_epoch, "gecmis": gecmis,
            "sure_sn": time.time() - baslangic}


def sicaklik_bul(logit: np.ndarray, etiket) -> float:
    """Doğrulama logitleri üzerinde NLL'yi en aza indiren sıcaklık (olasılık kalibrasyonu)."""
    izgara = np.concatenate([np.linspace(0.5, 1.0, 11), np.linspace(1.1, 5.0, 40)])
    return float(min(izgara, key=lambda t: _kayip(logit / t, etiket)))
