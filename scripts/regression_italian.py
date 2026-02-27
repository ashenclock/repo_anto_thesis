import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# Setup path per importare moduli dalla cartella src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal
from src.utils import set_seed, clear_memory

# === CONFIGURAZIONE PERCORSI ===
EXCEL_FILE = "data/metadata/dataset_10_02_Foglio_1_-_dataset.csv"
META_FILE = "data/metadata/mmse_experiment_metadata.csv" 
FOLDS_FILE = "data/metadata/dataset_folds.csv"
OUT_DIR = Path("outputs/moca_regression_final_139")
PLOT_DIR = Path("plots/moca_sota_final")

def mmse_to_moca_trzepacz(mmse):
    """Conversione scientifica Trzepacz (2015)"""
    if pd.isna(mmse): return np.nan
    mapping = {
        30: 28, 29: 25, 28: 23, 27: 22, 26: 20, 25: 19, 24: 18, 
        23: 17, 22: 17, 21: 16, 20: 15, 19: 14, 18: 13, 17: 12, 
        16: 11, 15: 10, 14: 8, 13: 6, 12: 5, 11: 3, 10: 2, 9: 1
    }
    try:
        m_val = int(round(float(mmse)))
        return float(mapping.get(m_val, max(1, m_val - 2)))
    except: return np.nan

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    config.task = "regression"
    config.model.output_dim = 1
    config.modality = "multimodal_cross_attention"
    config.model.text.name = "Musixmatch/umberto-commoncrawl-cased-v1"
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device(config.device)

    print(f"\n{'='*60}")
    print(f"🚀 SOTA MoCA REGRESSION - MISSION 139")
    print(f"{'='*60}")

    # --- 1. SUPER MERGE CRONOLOGICO ---
    if not Path(EXCEL_FILE).exists():
        print(f"❌ Errore: File {EXCEL_FILE} non trovato nella root del server.")
        return

    df_ex = pd.read_csv(EXCEL_FILE)
    df_ex['diag_ex'] = df_ex['Malattie Diagnosticate'].apply(clean_diag)
    df_ex = df_ex[df_ex['diag_ex'].isin(['CTR', 'MCI', 'MILD-AD'])].sort_values('Data Acquisizione').reset_index(drop=True)

    df_me = pd.read_csv(META_FILE)
    df_it = df_me[df_me['Language'] == 'IT'].copy()
    
    # Ordine numerico degli ID (SUBJ_0001, 0002...)
    df_it['ID_num'] = df_it['Subject_ID'].apply(lambda x: int(x.split('_')[1]))
    df_it = df_it.sort_values(['ID_num', 'Subject_ID']).reset_index(drop=True)
    
    unique_ids = sorted(df_it['Subject_ID'].apply(lambda x: x.split('_Task_')[0]).unique(), key=lambda x: int(x.split('_')[1]))

    id_to_score = {}
    ex_ptr = 0
    for subj_id in unique_ids:
        # Trova la diagnosi nel metadata per questo ID
        meta_diag = df_it[df_it['Subject_ID'].str.startswith(subj_id)].iloc[0]['Diagnosis']
        
        # Sincronizza con l'excel
        while ex_ptr < len(df_ex):
            ex_row = df_ex.iloc[ex_ptr]
            if ex_row['diag_ex'] == meta_diag:
                moca = to_f(ex_row['MOCA'])
                mmse = to_f(ex_row['MMSE'])
                score = moca if pd.notna(moca) else mmse_to_moca_trzepacz(mmse)
                
                # Fallback estremo per non avere NaN
                if pd.isna(score):
                    score = 26.0 if meta_diag == 'CTR' else (22.0 if meta_diag == 'MCI' else 17.0)
                
                id_to_score[subj_id] = score
                ex_ptr += 1
                break
            else:
                ex_ptr += 1

    # Applichiamo i punteggi (fondamentale per il dataloader)
    df_it['mmse_score'] = df_it['Subject_ID'].apply(lambda x: id_to_score.get(x.split('_Task_')[0], 20.0))

    # Match Folds
    df_folds = pd.read_csv(FOLDS_FILE)
    f_id_col = 'Subject_ID' if 'Subject_ID' in df_folds.columns else 'ID'
    fold_map = df_folds.set_index(f_id_col)['kfold'].to_dict()
    df_it['kfold'] = df_it['Subject_ID'].apply(lambda x: fold_map.get(x.split('_Task_')[0], fold_map.get(x, 0)))

    print(f"✅ SINCRONIZZAZIONE COMPLETATA: {len(unique_ids)} pazienti (139 attesi).")
    print(df_it.groupby('Diagnosis')['mmse_score'].mean().round(2))

    # --- 2. TRAINING ---
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    all_oof_preds, all_oof_targets, all_oof_ids = [], [], []

    unique_folds = sorted(df_it['kfold'].unique())
    for f_id in unique_folds:
        print(f"\n--- 🔄 START FOLD {int(f_id)} ---")
        train_df = df_it[df_it['kfold'] != f_id].reset_index(drop=True)
        val_df = df_it[df_it['kfold'] == f_id].reset_index(drop=True)
        
        train_loader = torch.utils.data.DataLoader(MultimodalDataset(train_df, config, tokenizer), batch_size=8, shuffle=True, collate_fn=collate_multimodal, drop_last=True)
        val_loader = torch.utils.data.DataLoader(MultimodalDataset(val_df, config, tokenizer), batch_size=8, shuffle=False, collate_fn=collate_multimodal)

        model = build_model(config).to(device)
        # Bias init sulla media per aiutare il modello
        nn.init.constant_(model.classifier[-1].bias, df_it['mmse_score'].mean())
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=len(train_loader)*20)
        criterion = nn.L1Loss() # MAE

        best_mae = 100.0
        for epoch in range(20):
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/20", leave=False)
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                targets = batch.pop('labels').float()
                outputs = model(batch).squeeze(-1)
                loss = criterion(outputs, targets)
                optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
                train_loss += loss.item()
                pbar.set_postfix({'MAE': f"{loss.item():.2f}"})

            # Validazione
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for b in val_loader:
                    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                    vt = b.pop('labels').float()
                    vo = model(b).squeeze(-1)
                    v_preds.extend(vo.cpu().numpy()); v_targets.extend(vt.cpu().numpy())
            
            m = mean_absolute_error(v_targets, v_preds)
            if m < best_mae:
                best_mae = m
                torch.save(model.state_dict(), OUT_DIR / f"model_{int(f_id)}.pt")
                print(f"   Ep {epoch+1:02d} | Val MAE: {m:.2f} 🌟")

        # Inference OOF (Out-of-Fold)
        model.load_state_dict(torch.load(OUT_DIR / f"model_{int(f_id)}.pt"))
        model.eval()
        with torch.no_grad():
            for b in val_loader:
                all_oof_ids.extend(b.pop('id'))
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                t = b.pop('labels').float()
                o = model(batch).squeeze(-1)
                all_oof_preds.extend(o.cpu().numpy()); all_oof_targets.extend(t.cpu().numpy())

    # --- 3. ANALISI FINALE E PLOT ---
    res_df = pd.DataFrame({'ID': all_oof_ids, 'Actual': np.array(all_oof_targets).flatten(), 'Predicted': np.array(all_oof_preds).flatten()})
    diag_map = df_it.set_index('Subject_ID')['Diagnosis'].to_dict()
    res_df['Diagnosis'] = res_df['ID'].map(diag_map)
    res_df.to_csv(OUT_DIR / "oof_results_139.csv", index=False)

    final_mae = mean_absolute_error(res_df['Actual'], res_df['Predicted'])
    corr, _ = pearsonr(res_df['Actual'], res_df['Predicted'])
    print(f"\n🏆 CONCLUSO: MAE FINALE {final_mae:.2f} | Correlazione r: {corr:.3f}")

    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    palette = {"CTR": "#66c2a5", "MCI": "#8da0cb", "MILD-AD": "#fc8d62"}
    sns.boxplot(data=res_df, x='Diagnosis', y='Predicted', order=['CTR', 'MCI', 'MILD-AD'], palette=palette, width=0.6, fliersize=0)
    sns.stripplot(data=res_df, x='Diagnosis', y='Predicted', order=['CTR', 'MCI', 'MILD-AD'], color='black', alpha=0.4, jitter=True)
    plt.axhline(y=26, color='red', linestyle='--', label='Cut-off (26)')
    plt.title(f"SOTA Regression (N={len(res_df)})\nMAE: {final_mae:.2f} | r: {corr:.2f}")
    plt.savefig(PLOT_DIR / "sota_moca_final_139.png", dpi=300)
    print(f"📈 Grafico salvato: {PLOT_DIR}/sota_moca_final_139.png")

if __name__ == "__main__":
    main()