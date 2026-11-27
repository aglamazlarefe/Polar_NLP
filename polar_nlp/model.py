"""Hibrit model: BERTurk [CLS] temsili + stilometri dalı → sınıflandırıcı."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel

from .config import N_SINIF

AGIRLIK_DOSYASI = "model.safetensors"
BILGI_DOSYASI = "model_info.json"


class HybridPolarModel(nn.Module):
    """`stilo_kullan=False` ile yalnız-BERT taban modeli elde edilir (ablation)."""

    def __init__(self, bert: nn.Module, n_stilo: int, n_sinif: int = N_SINIF,
                 stilo_kullan: bool = True, dropout: float = 0.2, stilo_gizli: int = 32):
        super().__init__()
        self.bert = bert
        self.n_stilo = n_stilo
        self.stilo_kullan = bool(stilo_kullan and n_stilo > 0)
        self.dropout_orani = dropout
        self.stilo_gizli = stilo_gizli
        giris = bert.config.hidden_size
        if self.stilo_kullan:
            # Stilometri önce kendi küçük katmanından geçer; 768 boyutlu BERT vektörünün
            # yanında 10 ham sayı olarak kaybolmasın diye.
            self.stilo_dali = nn.Sequential(nn.Linear(n_stilo, stilo_gizli), nn.ReLU(), nn.LayerNorm(stilo_gizli))
            giris += stilo_gizli
        self.siniflandirici = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(giris, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, n_sinif)
        )

    def forward(self, input_ids, attention_mask, stilometri=None):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        if self.stilo_kullan:
            cls = torch.cat([cls, self.stilo_dali(stilometri)], dim=1)
        return self.siniflandirici(cls)

    def bas_parametreleri(self):
        """Rastgele başlatılan (BERT dışı) parametreler — daha yüksek öğrenme oranı alır."""
        return [p for ad, p in self.named_parameters() if not ad.startswith("bert.")]

    # ------------------------------------------------------------------ kayıt / yükleme

    @classmethod
    def onceden_egitilmis(cls, model_adi: str, n_stilo: int, **kw) -> "HybridPolarModel":
        return cls(AutoModel.from_pretrained(model_adi), n_stilo, **kw)

    def kaydet(self, dizin: str | Path, durum: dict | None = None, ek_bilgi: dict | None = None) -> None:
        dizin = Path(dizin)
        dizin.mkdir(parents=True, exist_ok=True)
        durum = durum if durum is not None else self.state_dict()
        save_file({k: v.detach().cpu().contiguous() for k, v in durum.items()}, str(dizin / AGIRLIK_DOSYASI))
        self.bert.config.to_json_file(str(dizin / "config.json"))
        bilgi = {
            "model_tipi": "HybridPolarModel",
            "n_stilo": self.n_stilo,
            "n_sinif": self.siniflandirici[-1].out_features,
            "stilo_kullan": self.stilo_kullan,
            "dropout": self.dropout_orani,
            "stilo_gizli": self.stilo_gizli,
        }
        bilgi.update(ek_bilgi or {})
        (dizin / BILGI_DOSYASI).write_text(json.dumps(bilgi, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def kayittan(cls, dizin: str | Path) -> tuple["HybridPolarModel", dict]:
        """Kaydedilmiş modeli yalnız yerel dosyalardan kurar (internet gerekmez)."""
        dizin = Path(dizin)
        bilgi = json.loads((dizin / BILGI_DOSYASI).read_text(encoding="utf-8"))
        bert = AutoModel.from_config(AutoConfig.from_pretrained(str(dizin)))
        model = cls(bert, bilgi["n_stilo"], bilgi["n_sinif"], bilgi["stilo_kullan"],
                    bilgi.get("dropout", 0.2), bilgi.get("stilo_gizli", 32))
        model.load_state_dict(load_file(str(dizin / AGIRLIK_DOSYASI)))
        model.eval()
        return model, bilgi
