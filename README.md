# 🎓 Multimodal Alzheimer's Detection — Thesis Codebase

Cross-lingual and multimodal thesis framework for cognitive assessment from speech, text, and clinical metadata.

The repository supports experiments on Italian and English datasets (including Sicily + ADReSSo) with fusion strategies, task-specific training, and cross-lingual evaluation.

## 🔍 What is included

- End-to-end training/evaluation pipelines
- Data preparation and transcript processing utilities
- Cross-dataset and cross-lingual experiments
- Analysis scripts for tables, metrics, and figures

## 📁 Repository Layout

```text
.
├── config.yaml
├── src/
│   ├── config.py
│   ├── data.py
│   ├── dataset.py
│   ├── engine.py
│   ├── models.py
│   ├── tabular_engine.py
│   └── utils.py
├── scripts/
│   ├── training/
│   ├── inference/
│   ├── data_prep/
│   ├── analysis/
│   └── visualization/
├── results/
├── plots/
└── thesis.pdf
```

## 🚀 Typical Workflow

### 1) Prepare data

```bash
python scripts/data_prep/prepare_dataset.py
python scripts/data_prep/transcribe_all.py
```

### 2) Run training

```bash
python scripts/training/run_all_tasks.py
```

### 3) Generate reports

```bash
python scripts/analysis/gen_results.py
python scripts/analysis/generate_final_reports.py
```

## ⚙️ Notes

- The full pipeline is controlled through `config.yaml`
- Some datasets, checkpoints, and generated outputs are intentionally not versioned in Git
- This codebase is research-oriented and built for reproducible experimentation

