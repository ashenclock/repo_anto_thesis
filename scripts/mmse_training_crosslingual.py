import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal
from src.utils import set_seed, clear_memory

# === CONFIGURAZIONE PERCORSI ===
EXCEL_FILE = "data/metadata/dataset_10_02_Foglio_1_-_dataset.csv"
META_FILE = "data/metadata/mmse_experiment_metadata.csv" 
FOLDS_FILE = "data/metadata/dataset_folds.csv"
SCORES_EN_FILE = "data/metadata/adresso_FULL_mmse.csv"

OUT_DIR = Path("outputs/mmse_ONLY_XLMR_139")
PLOT_DIR = Path("plots/mmse_only_results")

# ==============================================================================
# 1. FUNZIONI DI SUPPORTO
# ==============================================================================

def to_f(val):
    try:
        s = str(val).lower().replace(',', '.').strip()
        if 'rifiuto' in s or 'nan' in s or not s: return np.nan
        return float(s)
    except: return np.nan

def clean_diag(d):
    d = str(d).strip().upper()
    if 'AD-LIEVE' in d or 'AD LIEVE' in d: return 'MILD-AD'
    if 'MCI' in d: return 'MCI'
    if 'CTR' in d or 'CONTROLLI' in d: return 'CTR'
    return 'UNK'

def load_italian_mmse_139():
    print("🇮🇹 Caricamento 139 Pazienti Italiani (MMSE)...")
    df_ex = pd.read_csv(EXCEL_FILE)
    df_ex['diag_ex'] = df_ex['Malattie Diagnosticate'].apply(clean_diag)
    df_ex = df_ex[df_ex['diag_ex'].isin(['CTR', 'MCI', 'MILD-AD'])].sort_values('Data Acquisizione').reset_index(drop=True)

    df_me = pd.read_csv(META_FILE)
    df_it = df_me[df_me['Language'] == 'IT'].copy()
    df_it['ID_num'] = df_it['Subject_ID'].apply(lambda x: int(x.split('_')[1]))
    df_it = df_it.sort_values(['ID_num', 'Subject_ID']).reset_index(drop=True)
    
    unique_ids = sorted(df_it['Subject_ID'].apply(lambda x: x.split('_Task_')[0]).unique(), key=lambda x: int(x.split('_')[1]))

    id_to_score = {}
    ex_ptr = 0
    for subj_id in unique_ids:
        meta_diag = df_it[df_it['Subject_ID'].str.startswith(subj_id)].iloc[0]['Diagnosis']
        found = False
        while ex_ptr < len(df_ex):
            ex_row = df_ex.iloc[ex_ptr]
            if ex_row['diag_ex'] == meta_diag:
                mmse = to_f(ex_row['MMSE'])
                if pd.isna(mmse):
                    mmse = 28.0 if meta_diag == 'CTR' else (26.0 if meta_diag == 'MCI' else 22.0)
                id_to_score[subj_id] = mmse
                ex_ptr += 1
                found = True
                break
            else: ex_ptr += 1
        if not found: id_to_score[subj_id] = 25.0

    df_it['mmse_score'] = df_it['Subject_ID'].apply(lambda x: id_to_score.get(x.split('_Task_')[0], 25.0))
    df_folds = pd.read_csv(FOLDS_FILE)
    f_map = df_folds.set_index('Subject_ID')['kfold'].to_dict()
    df_it['kfold'] = df_it['Subject_ID'].apply(lambda x: f_map.get(x.split('_Task_')[0], 0))
    return df_it

def load_english_mmse_test():
    print("🇬🇧 Caricamento ADReSSo Test (MMSE)...")
    df_me = pd.read_csv(META_FILE)
    df_en = df_me[df_me['Dataset'] == 'ADReSSo_Test'].copy()
    df_scores = pd.read_csv(SCORES_EN_FILE)
    df_scores['ID'] = df_scores['ID'].astype(str).str.strip().str.replace('"', '')
    id_map = dict(zip(df_scores['ID'], df_scores['Score']))
    df_en['Actual_MMSE'] = df_en['Subject_ID'].apply(lambda x: id_map.get(x.replace("EN_", "").replace("TEST_", "").split("_Task_")[0].strip(), np.nan))
    df_en = df_en.dropna(subset=['Actual_MMSE']).reset_index(drop=True)
    df_en['mmse_score'] = 0.0
    return df_en

# ==============================================================================
# 2. TRAINING & INFERENCE
# ==============================================================================

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using Device: {device}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config("config.yaml")
    config.task = "regression"
    config.model.output_dim = 1
    config.model.text.name = "xlm-roberta-base"
    
    print(f"⏳ Pre-loading Tokenizer and Model Weights ({config.model.text.name})...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    # Forza il download se non presente per evitare l'hang silenzioso
    _ = AutoModel.from_pretrained(config.model.text.name)
    print("✅ Model Weights ready.")

    # A. TRAIN SU ITALIA
    df_it = load_italian_mmse_139()
    all_it_preds, all_it_targets, all_it_ids = [], [], []
    unique_folds = sorted(df_it['kfold'].unique())

    print(f"\n🚀 START MMSE REGRESSION (XLM-R) - N={len(df_it)}")

    for f_id in unique_folds:
        print(f"\n🔹 Training Fold {int(f_id)}...")
        train_df = df_it[df_it['kfold'] != f_id].reset_index(drop=True)
        val_df = df_it[df_it['kfold'] == f_id].reset_index(drop=True)
        
        # --- FIX: num_workers=0 per evitare hang su Ubuntu ---
        train_loader = torch.utils.data.DataLoader(
            MultimodalDataset(train_df, config, tokenizer), 
            batch_size=16, shuffle=True, collate_fn=collate_multimodal, 
            drop_last=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            MultimodalDataset(val_df, config, tokenizer), 
            batch_size=16, shuffle=False, collate_fn=collate_multimodal, 
            num_workers=0
        )

        clear_memory()
        print("   🏗️ Building model...")
        model = build_model(config).to(device)
        nn.init.constant_(model.classifier[-1].bias, 26.0) 
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-5, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=len(train_loader)*25)
        criterion = nn.L1Loss()

        best_mae = 100.0
        pbar = tqdm(range(25), desc=f"   Fold {int(f_id)} Training")
        for epoch in pbar:
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                targets = batch.pop('labels').float()
                outputs = model(batch).squeeze(-1)
                loss = criterion(outputs, targets)
                optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
                train_loss += loss.item()

            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for b in val_loader:
                    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                    vt = b.pop('labels').float(); vo = model(b).squeeze(-1)
                    v_preds.extend(vo.cpu().numpy()); v_targets.extend(vt.cpu().numpy())
            
            m = mean_absolute_error(v_targets, v_preds)
            pbar.set_postfix({'Val_MAE': f"{m:.2f}"})
            if m < best_mae:
                best_mae = m
                torch.save(model.state_dict(), OUT_DIR / f"model_fold_{int(f_id)}.pt")

        # Recover best and predict OOF
        model.load_state_dict(torch.load(OUT_DIR / f"model_fold_{int(f_id)}.pt"))
        model.eval()
        with torch.no_grad():
            for b in val_loader:
                all_it_ids.extend(b.pop('id'))
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                all_it_preds.extend(model(batch).squeeze(-1).cpu().numpy())
                all_it_targets.extend(batch.pop('labels').cpu().numpy())

    # B. INFERENZA SU INGLESE
    print("\n🇬🇧 Esecuzione Zero-Shot su ADReSSo...")
    df_en = load_english_mmse_test()
    en_loader = torch.utils.data.DataLoader(MultimodalDataset(df_en, config, tokenizer), batch_size=16, shuffle=False, collate_fn=collate_multimodal, num_workers=0)
    en_preds_matrix = np.zeros((len(df_en), len(unique_folds)))

    for i, f_id in enumerate(unique_folds):
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(OUT_DIR / f"model_fold_{int(f_id)}.pt"))
        model.eval()
        with torch.no_grad():
            for j, b in enumerate(en_loader):
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                out = model(b).squeeze(-1)
                en_preds_matrix[j*16 : j*16 + len(out), i] = out.cpu().numpy()

    df_en['Predicted_MMSE'] = np.mean(en_preds_matrix, axis=1)

    # C. VISUALIZZAZIONE
    df_it_res = pd.DataFrame({'ID': all_it_ids, 'Actual': all_it_targets, 'Predicted': all_it_preds})
    df_it_res['Diagnosis'] = df_it_res['ID'].map(df_it.set_index('Subject_ID')['Diagnosis'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.boxplot(data=df_it_res, x='Diagnosis', y='Predicted', order=['CTR', 'MCI', 'MILD-AD'], palette="Set2", ax=ax1)
    ax1.set_title("ITALY: Predicted MMSE (OOF)")
    ax1.set_ylim(10, 31)

    sns.boxplot(data=df_en, x='Diagnosis', y='Predicted_MMSE', order=['CTR', 'AD'], palette="Set1", ax=ax2)
    ax2.set_title("ENGLISH: Predicted MMSE (Zero-Shot)")
    ax2.set_ylim(10, 31)

    plt.savefig(PLOT_DIR / "final_MMSE_only_comparison.png", dpi=300)
    
    mae_en = mean_absolute_error(df_en['Actual_MMSE'], df_en['Predicted_MMSE'])
    print("\n" + "="*50)
    print(f"🏆 PERFORMANCE MMSE ONLY (Zero-Shot EN):")
    print(f"   MAE: {mae_en:.2f} (Benchmark Pappagari: 3.85)")
    print("="*50)

if __name__ == "__main__":
    main()