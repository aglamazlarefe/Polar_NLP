#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Polar NLP — BERTurk Baseline Fine-Tuning Scripti.

Kullanım:
    python train_berturk.py              # BERTurk modelini fine-tune et
    python -m polar_nlp.train_berturk    # Aynı işlem (paket üzerinden)

Bu dosya, polar_nlp paketindeki asıl modüle yönlendirme yapar.
"""

from polar_nlp.train_berturk import main

if __name__ == "__main__":
    main()