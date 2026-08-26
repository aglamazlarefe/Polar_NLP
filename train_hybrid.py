#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Polar NLP — BERTurk + Stilometri Hibrit Model Eğitim Scripti.

Kullanım:
    python train_hybrid.py              # Hibrit modeli eğit
    python -m polar_nlp.train_hybrid    # Aynı işlem (paket üzerinden)

Bu dosya, polar_nlp paketindeki asıl modüle yönlendirme yapar.
"""

from polar_nlp.train_hybrid import main

if __name__ == "__main__":
    main()