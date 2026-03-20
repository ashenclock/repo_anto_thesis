# Multimodal Alzheimer's Detection — Thesis Codebase

Cross-lingual multimodal system for Alzheimer's disease detection combining audio, text, and clinical features. Supports Italian and English datasets (Sicily + ADReSSo) with cross-attention fusion, ensemble methods, and cross-lingual zero-shot transfer.

## Project Structure

```
.
├── config.yaml              # Main configuration (paths, model, training params)
├── src/                     # Core library
│   ├── config.py            # Configuration loading & validation
│   ├── data.py              # DataLoaders, splits, collate functions
│   ├── dataset.py           # Dataset classes (multimodal, tabular)
│   ├── engine.py            # Training & evaluation loops
│   ├── models.py            # Model architectures (cross-attention, fusion)
│   ├── tabular_engine.py    # XGBoost / tabular model training
│   └── utils.py             # Seed, memory, logging utilities
├── scripts/
│   ├── training/            # Training experiments
│   ├── inference/           # Inference & prediction
│   ├── data_prep/           # Data preparation & feature extraction
│   ├── analysis/            # Statistical analysis & report generation
│   └── visualization/       # Plotting & visualization
├── results/                 # Generated CSV/XLSX result tables
├── data/                    # Datasets, metadata, transcripts, features (not in git)
├── outputs/                 # Model checkpoints & predictions (not in git)
└── plots/                   # Generated figures (not in git)
```

## Scripts

### Training (`scripts/training/`)

| Script | Description |
|--------|-------------|
| `run.py` | Minimal K-fold training baseline |
| `run_all_tasks.py` | Per-task multimodal training across all speech tasks |
| `full_training.py` | MMSE cross-lingual regression (IT→EN, EN→IT, mixed) |
| `run_mmse_training.py` | MMSE regression with 10-fold ensemble (EN→IT zero-shot) |
| `run_moca_training.py` | MoCA regression with 10-fold ensemble |
| `run_cross_dataset_ensemble.py` | Cross-dataset ensemble with weighted loss |
| `run_balanced_mix.py` | Mixed IT+EN training with balanced sampling |
| `run_tabular_final.py` | Tabular XGBoost per-task training |
| `run_clinical_tabular.py` | Clinical-only tabular classification |

### Inference (`scripts/inference/`)

| Script | Description |
|--------|-------------|
| `inference_all.py` | Ensemble inference on IT + EN datasets (MMSE regression) |
| `predict_balanced_mix.py` | Classification evaluation on balanced mixed data |

### Data Preparation (`scripts/data_prep/`)

| Script | Description |
|--------|-------------|
| `prepare_dataset.py` | Build metadata CSV from raw audio (mic selection, VAD, SNR) |
| `transcribe_all.py` | ASR transcription (CrisperWhisper, NeMo, WhisperX) |
| `import_adresso_from_repo.py` | Unify Sicily + ADReSSo datasets |
| `extract_features.py` | OpenSMILE acoustic features (eGeMAPS, ComParE) |
| `extract_embeddings_batch.py` | Deep audio embeddings (Wav2Vec2, Whisper) |
| `precompute_audio.py` | Cache audio representations to disk |
| `prepare_phonemes.py` | Grapheme-to-phoneme conversion (espeak, IT) |
| `convert_mmse_to_moca_trzepacz.py` | MMSE→MoCA score conversion (Trzepacz 2015) |

### Analysis (`scripts/analysis/`)

| Script | Description |
|--------|-------------|
| `gen_results.py` | Full result tables (acoustic, text, architecture, voting) |
| `generate_final_reports.py` | Thesis report generation (per-task, per-model, patient-level) |
| `gen_crossdataset_final.py` | Cross-lingual & multilingual transfer tables |
| `finalize_patient_score.py` | Patient-level aggregation with voting strategies |
| `fuse_tasks.py` | Multi-task fusion (soft/hard voting) |
| `mci_final.py` | MCI/AD fusion strategies comparison |
| `xgb_final.py` | XGBoost meta-classifier on multimodal embeddings |
| `analyze_moca_significance.py` | Statistical significance tests on MoCA predictions |
| `analysis_cosine.py` | Cross-lingual embedding alignment & cosine similarity |
| `run_eda.py` | Exploratory data analysis on Italian dataset |

### Visualization (`scripts/visualization/`)

| Script | Description |
|--------|-------------|
| `visual.py` | Cross-lingual semantic sentence matching |
| `visual_audio.py` | Procrustes-aligned cross-lingual audio embeddings |
| `visualize_attention.py` | Multimodal cross-attention heatmaps |
| `visualize_audio_features.py` | Waveform, mel-spectrogram, MFCC visualization |
| `plot_excel_scores.py` | Clinical score distributions (MMSE/MoCA boxplots) |

## Setup

```bash
# Install dependencies
pip install torch transformers datasets opensmile librosa pandas scikit-learn xgboost

# Prepare data
python scripts/data_prep/prepare_dataset.py
python scripts/data_prep/transcribe_all.py

# Train
python scripts/training/run_all_tasks.py

# Generate results
python scripts/analysis/gen_results.py
```

## Configuration

All paths and hyperparameters are controlled via `config.yaml`. Key sections:

- **data**: dataset root, metadata files, transcript directories
- **model**: backbone selection (XLM-RoBERTa, UmBERTo, XPhoneBERT), fusion type, audio encoder
- **training**: learning rate, epochs, batch size, K-fold settings
