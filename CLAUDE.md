# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polar NLP is a Turkish NLP system for psycho-social risk screening of Antarctic base personnel. It uses a **hybrid BERTurk + Stylometry** architecture to classify logbook entries into 3 classes (Objective Report, Cognitive Fatigue, Winter Syndrome) and outputs a 4-axis risk profile.

## Architecture

```
BERTurk [CLS] (768-dim) ─┐
                          ├─→ Linear(773→256) → ReLU → Dropout → Linear(256→3) → Softmax
Stylometry (5-dim) ───────┘
```

- **Core model**: `polar_nlp/train_hybrid.py` — `HybridPolarModel` class, `polar_analiz_et()` inference API, `stilometri_cikar()` regex feature extractor
- **Baseline**: `polar_nlp/train_berturk.py` — vanilla HuggingFace Trainer fine-tune of `dbmdz/bert-base-turkish-cased`
- **Dashboard**: `dashboard.py` — Streamlit 4-page web app
- **Visualizations**: `gorseller.py` — matplotlib/seaborn report figures

## Key Commands

```bash
# Train hybrid model (BERTurk + stylometry fusion)
python train_hybrid.py

# Train baseline BERTurk model
python train_berturk.py

# Launch Streamlit dashboard
streamlit run dashboard.py

# Convert DOCX dataset to CSV
python temp_convert3.py

# Generate report figures
python gorseller.py
```

## Data Pipeline

`seyir defteri veri seti.docx` → `temp_convert3.py` → `output_utf8.csv` (300 rows, stratified 80/10/10 split)

## Code Conventions

- **Turkish variable/function names** for domain-specific code (stilometri, bilissel_yorgunluk, etc.)
- **English standard names** for ML boilerplate (model, tokenizer, dataloader)
- All print() logging prefixed with `[ARTIFACT N]` for report documentation
- Model artifacts save to `./egitilmis_hybrid_model/` or `./egitilmis_berturk_modeli/`

## Warnings

- `gorseller.py` uses hardcoded metric values (not computed from model) — update after training
- Dashboard reports "600 synthetic texts" but dataset has 300 rows
- `polar_analiz_et()` uses rule-based class override when confidence < 0.50 — review for production use
- Model checkpoint uses `copy.deepcopy()` for correct state_dict snapshot