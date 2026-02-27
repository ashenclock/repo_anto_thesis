import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer
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
from src.utils import set_seed

# === CONFIGURAZIONE PERCORSI (ESATTAMENTE COME NEL TRAINING) ===
EXCEL_FILE = "data/metadata/dataset_10_02_Foglio_1_-_dataset.csv"
META_FILE = "data/metadata/mmse_experiment_metadata.csv" 
FOLDS_FILE = "data/metadata/dataset_folds.csv"
OUT_DIR = Path("outputs/moca_regression_final_139") # Cartella dove sono i modelli .pt
PLOT_DIR = Path("plots/moca_sota_final")

# === FUNZIONI DI SUPPORTO (COPIATE DA regression_italian.py) ===

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
    print(f"\n{'='*60}")
    print(f"🛠️  RECOVERY MODE: Generazione risultati per 139 Pazienti")
    print(f"{'='*60}")
    
    # 1. Configurazione Modello
    config = load_config("config.yaml")
    config.task = "regression"
    config.model.output_dim = 1
    config.modality = "multimodal_cross_attention"
    config.model.text.name = "Musixmatch/umberto-commoncrawl-cased-v1"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. CARICAMENTO E MERGE DATI (Logica "Super Merge Cronologico")
    if not Path(EXCEL_FILE).exists():
        print(f"❌ Errore: File Excel {EXCEL_FILE} non trovato.")
        return

    # A. Carica e pulisce Excel
    df_ex = pd.read_csv(EXCEL_FILE)
    df_ex['diag_ex'] = df_ex['Malattie Diagnosticate'].apply(clean_diag)
    df_ex = df_ex[df_ex['diag_ex'].isin(['CTR', 'MCI', 'MILD-AD'])].sort_values('Data Acquisizione').reset_index(drop=True)

    # B. Carica Metadati
    df_me = pd.read_csv(META_FILE)
    df_it = df_me[df_me['Language'] == 'IT'].copy()
    
    # Ordine numerico degli ID (SUBJ_0001, 0002...) per allineamento cronologico
    df_it['ID_num'] = df_it['Subject_ID'].apply(lambda x: int(x.split('_')[1]))
    df_it = df_it.sort_values(['ID_num', 'Subject_ID']).reset_index(drop=True)
    
    unique_ids = sorted(df_it['Subject_ID'].apply(lambda x: x.split('_Task_')[0]).unique(), key=lambda x: int(x.split('_')[1]))

    # C. Algoritmo di Matching (Excel <-> Metadata)
    id_to_score = {}
    ex_ptr = 0
    
    print("🔄 Sincronizzazione Excel-Metadata in corso...")
    
    for subj_id in unique_ids:
        # Trova la diagnosi nel metadata per questo ID
        meta_rows = df_it[df_it['Subject_ID'].str.startswith(subj_id)]
        if len(meta_rows) == 0: continue
        meta_diag = meta_rows.iloc[0]['Diagnosis']
        
        # Sincronizza con l'excel
        found = False
        while ex_ptr < len(df_ex):
            ex_row = df_ex.iloc[ex_ptr]
            if ex_row['diag_ex'] == meta_diag:
                moca = to_f(ex_row['MOCA'])
                mmse = to_f(ex_row['MMSE'])
                
                # Logica priorità: MoCA > MMSE convertito
                score = moca if pd.notna(moca) else mmse_to_moca_trzepacz(mmse)
                
                # Fallback estremo se entrambi mancano nell'Excel
                if pd.isna(score):
                    score = 26.0 if meta_diag == 'CTR' else (22.0 if meta_diag == 'MCI' else 17.0)
                
                id_to_score[subj_id] = score
                ex_ptr += 1
                found = True
                break
            else:
                ex_ptr += 1
        
        # Se finisce l'excel ma ci sono ancora pazienti
        if not found and subj_id not in id_to_score:
             id_to_score[subj_id] = 26.0 if meta_diag == 'CTR' else (22.0 if meta_diag == 'MCI' else 17.0)

    # D. Applica i punteggi al DataFrame
    # Nota: Usiamo la colonna 'mmse_score' perché il Dataloader standard si aspetta quella o 'Score'
    df_it['mmse_score'] = df_it['Subject_ID'].apply(lambda x: id_to_score.get(x.split('_Task_')[0], 0.0))

    # E. Match Folds
    df_folds = pd.read_csv(FOLDS_FILE)
    f_id_col = 'Subject_ID' if 'Subject_ID' in df_folds.columns else 'ID'
    fold_map = df_folds.set_index(f_id_col)['kfold'].to_dict()
    df_it['kfold'] = df_it['Subject_ID'].apply(lambda x: fold_map.get(x.split('_Task_')[0], fold_map.get(x, 0)))

    print(f"✅ DATASET PRONTO: {len(df_it)} campioni audio totali (Task multipli per 139 pazienti).")

    # 3. INFERENCE LOOP
    all_oof_preds, all_oof_targets, all_oof_ids = [], [], []
    unique_folds = sorted(df_it['kfold'].unique())

    for f_id in unique_folds:
        model_path = OUT_DIR / f"model_{int(f_id)}.pt"
        
        if not model_path.exists():
            print(f"⚠️  Modello {model_path} non trovato. Salto Fold {f_id}.")
            continue
            
        print(f"📂 Fold {int(f_id)}: Caricamento {model_path.name}...")
        
        # Prendi solo i dati di validation di questo fold (Out-Of-Fold)
        val_df = df_it[df_it['kfold'] == f_id].reset_index(drop=True)
        
        val_loader = torch.utils.data.DataLoader(
            MultimodalDataset(val_df, config, tokenizer), 
            batch_size=8, 
            shuffle=False, 
            collate_fn=collate_multimodal
        )

        model = build_model(config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            for b in val_loader:
                ids = b.pop('id')
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                targets = batch.pop('labels').float()
                outputs = model(batch).squeeze(-1) # Output è [Batch, 1] -> [Batch]
                
                all_oof_ids.extend(ids)
                all_oof_preds.extend(outputs.cpu().numpy().tolist())
                all_oof_targets.extend(targets.cpu().numpy().tolist())

    # 4. SALVATAGGIO E REPORT
    if not all_oof_ids:
        print("❌ Nessuna predizione generata.")
        return

    res_df = pd.DataFrame({
        'ID': all_oof_ids, 
        'Actual': all_oof_targets, 
        'Predicted': all_oof_preds
    })
    
    # Recupera diagnosi originale
    diag_map = df_it.set_index('Subject_ID')['Diagnosis'].to_dict()
    res_df['Diagnosis'] = res_df['ID'].map(diag_map)
    
    # Salva CSV
    res_df.to_csv(OUT_DIR / "final_results_RECOVERED_139.csv", index=False)

    final_mae = mean_absolute_error(res_df['Actual'], res_df['Predicted'])
    corr, _ = pearsonr(res_df['Actual'], res_df['Predicted'])
    
    print("\n" + "="*50)
    print(f"🏆 RISULTATI RECUPERATI (Campioni processati: {len(res_df)})")
    print(f"   MAE:  {final_mae:.2f}")
    print(f"   Corr: {corr:.3f}")
    print("="*50)

    # Plot
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    palette = {"CTR": "#66c2a5", "MCI": "#8da0cb", "MILD-AD": "#fc8d62"}
    
    # Filtra solo le diagnosi note per il plot
    plot_df = res_df[res_df['Diagnosis'].isin(['CTR', 'MCI', 'MILD-AD'])]
    
    sns.boxplot(data=plot_df, x='Diagnosis', y='Predicted', order=['CTR', 'MCI', 'MILD-AD'], palette=palette, width=0.6, fliersize=0)
    sns.stripplot(data=plot_df, x='Diagnosis', y='Predicted', order=['CTR', 'MCI', 'MILD-AD'], color='black', alpha=0.4, jitter=True)
    
    plt.axhline(y=26, color='red', linestyle='--', label='Cut-off Sano (26)')
    plt.title(f"SOTA IT-Only MoCA Regression (N={len(plot_df)})\nMAE: {final_mae:.2f}")
    
    plt.savefig(PLOT_DIR / "sota_moca_final_RECOVERY_139.png", dpi=300)
    print(f"📈 Grafico salvato in: {PLOT_DIR}")

if __name__ == "__main__":
    main()