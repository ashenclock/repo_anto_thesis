import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal

import warnings
warnings.filterwarnings("ignore")

# === CONFIGURAZIONE PERCORSI ===
MODELS_XLMR_IT = Path("outputs/mmse_ONLY_XLMR_139")      
MODELS_XLMR_EN = Path("outputs/mmse_EN_to_IT_XLMR")     
MODELS_UMB_IT  = Path("outputs/moca_regression_final_139") 

META_FILE = "data/metadata/mmse_experiment_metadata.csv"
EXCEL_FILE = "data/metadata/dataset_10_02_Foglio_1_-_dataset.csv"
SCORES_EN_FILE = "data/metadata/adresso_FULL_mmse.csv"

OUT_DIR = Path("outputs/final_regression_results_6tasks")
PLOT_DIR = Path("plots/final_regression_plots_6tasks")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. FUNZIONI DI SUPPORTO ---

def moca_to_mmse_continuous(moca_float):
    if pd.isna(moca_float): return np.nan
    xp = [0, 5, 10, 15, 18, 20, 22, 24, 26, 28, 30] 
    fp = [6, 12, 15, 20, 24, 26, 27, 28, 29, 30, 30] 
    return float(np.interp(moca_float, xp, fp))

def clean_diag(d):
    d = str(d).strip().upper()
    if 'AD' in d: return 'AD'
    if 'MCI' in d: return 'MCI'
    if 'CTR' in d or 'CONTROLLI' in d: return 'CTR'
    return 'UNK'

def mmse_to_moca_trzepacz(mmse):
    if pd.isna(mmse): return np.nan
    mapping = {30:28, 29:25, 28:23, 27:22, 26:20, 25:19, 24:18, 23:17, 22:17, 21:16, 20:15, 19:14, 18:13, 17:12, 16:11, 15:10, 14:8, 13:6, 12:5, 11:3, 10:2, 9:1}
    try: return float(mapping.get(int(round(float(mmse))), max(1, int(mmse)-2)))
    except: return np.nan

def load_data():
    print("📂 Caricamento e Allineamento Dati...")
    df_meta = pd.read_csv(META_FILE)
    df_it = df_meta[df_meta['Language'] == 'IT'].copy()
    df_ex = pd.read_csv(EXCEL_FILE)
    df_ex['diag_ex'] = df_ex['Malattie Diagnosticate'].apply(clean_diag)
    df_it['ID_num'] = df_it['Subject_ID'].apply(lambda x: int(x.split('_')[1]))
    df_it = df_it.sort_values(['ID_num', 'Subject_ID']).reset_index(drop=True)
    unique_ids = sorted(df_it['Subject_ID'].apply(lambda x: x.split('_Task_')[0]).unique(), key=lambda x: int(x.split('_')[1]))
    id_to_score = {}
    ex_ptr = 0
    for subj_id in unique_ids:
        meta_rows = df_it[df_it['Subject_ID'].str.startswith(subj_id)]
        if len(meta_rows) == 0: continue
        meta_diag = clean_diag(meta_rows.iloc[0]['Diagnosis'])
        found = False
        while ex_ptr < len(df_ex):
            ex_row = df_ex.iloc[ex_ptr]
            if ex_row['diag_ex'] == meta_diag:
                mmse = pd.to_numeric(str(ex_row['MMSE']).replace(',', '.'), errors='coerce')
                id_to_score[subj_id] = mmse if pd.notna(mmse) else (28.0 if meta_diag == 'CTR' else (26.0 if meta_diag == 'MCI' else 22.0))
                ex_ptr += 1
                found = True
                break
            else: ex_ptr += 1
        if not found: id_to_score[subj_id] = 25.0
    df_it['Actual_MMSE'] = df_it['Subject_ID'].apply(lambda x: id_to_score.get(x.split('_Task_')[0], 25.0))
    df_it['Diagnosis'] = df_it['Diagnosis'].apply(clean_diag)
    df_it['mmse_score'] = 0.0
    df_en = df_meta[df_meta['Dataset'] == 'ADReSSo_Test'].copy()
    df_scores = pd.read_csv(SCORES_EN_FILE)
    df_scores['ID'] = df_scores['ID'].astype(str).str.strip().str.replace('"', '')
    en_gt_map = dict(zip(df_scores['ID'], df_scores['Score']))
    df_en['Actual_MMSE'] = df_en['Subject_ID'].apply(lambda x: en_gt_map.get(x.replace("EN_", "").replace("TEST_", "").split("_Task_")[0].strip(), np.nan))
    df_en = df_en.dropna(subset=['Actual_MMSE']).reset_index(drop=True)
    df_en['Diagnosis'] = df_en['Diagnosis'].apply(clean_diag)
    df_en['mmse_score'] = 0.0
    return df_it, df_en

# --- 2. MOTORE INFERENZA ---

def run_ensemble(df_target, model_dir, model_name, config, device, is_moca=False):
    model_files = sorted(list(model_dir.glob("model*.pt")))
    if not model_files: return None
    config.model.text.name = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    loader = torch.utils.data.DataLoader(MultimodalDataset(df_target, config, tokenizer), batch_size=16, shuffle=False, collate_fn=collate_multimodal)
    preds_matrix = np.zeros((len(df_target), len(model_files)))
    for i, m_path in enumerate(model_files):
        config.model.output_dim = 1
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(m_path, map_location=device, weights_only=False))
        model.eval()
        curr_preds = []
        with torch.no_grad():
            for b in loader:
                b_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items() if k != 'id'}
                out = model(b_in).view(-1)
                curr_preds.extend(out.cpu().numpy().flatten().tolist())
        preds_matrix[:, i] = curr_preds
    final = np.mean(preds_matrix, axis=1)
    if is_moca: final = np.array([moca_to_mmse_continuous(p) for p in final])
    return final

# --- 3. MAIN ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("config.yaml")
    df_it_orig, df_en_orig = load_data()
    
    pal_it = {"CTR": "#66c2a5", "MCI": "#8da0cb", "AD": "#fc8d62"}
    pal_en = {"CTR": "#66c2a5", "AD": "#fc8d62"}
    
    tasks = [
        ("XLM_IT_to_IT", df_it_orig.copy(), MODELS_XLMR_IT, "xlm-roberta-base", "XLM-RoBERTa", "IT -> IT", pal_it, False),
        ("XLM_IT_to_EN", df_en_orig.copy(), MODELS_XLMR_IT, "xlm-roberta-base", "XLM-RoBERTa", "IT -> EN (Zero-Shot)", pal_en, False),
        ("XLM_EN_to_EN", df_en_orig.copy(), MODELS_XLMR_EN, "xlm-roberta-base", "XLM-RoBERTa", "EN -> EN", pal_en, False),
        ("XLM_EN_to_IT", df_it_orig.copy(), MODELS_XLMR_EN, "xlm-roberta-base", "XLM-RoBERTa", "EN -> IT (Zero-Shot)", pal_it, False),
        ("UMB_IT_to_IT", df_it_orig.copy(), MODELS_UMB_IT, "Musixmatch/umberto-commoncrawl-cased-v1", "UmBERTo", "IT -> IT", pal_it, True),
        ("UMB_IT_to_EN", df_en_orig.copy(), MODELS_UMB_IT, "Musixmatch/umberto-commoncrawl-cased-v1", "UmBERTo", "IT -> EN (Zero-Shot)", pal_en, True),
    ]

    sns.set_theme(style="whitegrid", font_scale=1.3)
    detailed_summary = []

    for code, df, m_dir, m_hf, m_label, direction, palette, is_moca in tasks:
        if not m_dir.exists(): continue
        print(f"🚀 Processing: {code}")
        preds = run_ensemble(df, m_dir, m_hf, config, device, is_moca=is_moca)
        if preds is None: continue
        df['Pred'] = preds
        
        rmse = np.sqrt(mean_squared_error(df['Actual_MMSE'], df['Pred']))
        mae = mean_absolute_error(df['Actual_MMSE'], df['Pred'])
        r, _ = pearsonr(df['Actual_MMSE'], df['Pred'])

        stats = df.groupby('Diagnosis')['Pred'].agg(['mean', 'std']).to_dict(orient='index')
        
        # --- FIX: AGGIUNTO MCI_Std QUI ---
        entry = {
            "Model": m_label, "Direction": direction, "RMSE": round(rmse, 2), "MAE": round(mae, 2), "r": round(r, 3),
            "CTR_Mean": round(stats['CTR']['mean'], 2), "CTR_Std": round(stats['CTR']['std'], 2),
            "AD_Mean": round(stats['AD']['mean'], 2), "AD_Std": round(stats['AD']['std'], 2),
            "MCI_Mean": round(stats['MCI']['mean'], 2) if 'MCI' in stats else "-",
            "MCI_Std": round(stats['MCI']['std'], 2) if 'MCI' in stats else "-"
        }
        detailed_summary.append(entry)

        plt.figure(figsize=(10, 8))
        order = ['CTR', 'MCI', 'AD'] if 'MCI' in df['Diagnosis'].unique() else ['CTR', 'AD']
        sns.boxplot(data=df, x='Diagnosis', y='Pred', order=order, palette=palette, width=0.6, fliersize=0)
        sns.stripplot(data=df, x='Diagnosis', y='Pred', order=order, color='.2', alpha=0.5, jitter=0.25, size=6)
        plt.title(f"{m_label}: {direction}", fontsize=20, fontweight='bold', pad=20)
        y_min, y_max = df['Pred'].min(), df['Pred'].max()
        margin = (y_max - y_min) * 0.3 if (y_max - y_min) > 0.1 else 1.0
        plt.ylim(y_min - margin, y_max + margin)
        plt.ylabel("Predicted MMSE Score", fontsize=16)
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"PLOT_{code}_Final.png", dpi=300)
        plt.close()

    df_results = pd.DataFrame(detailed_summary)
    # Riordino colonne per avere Mean e Std vicini
    final_cols = ["Model", "Direction", "RMSE", "MAE", "r", "CTR_Mean", "CTR_Std", "MCI_Mean", "MCI_Std", "AD_Mean", "AD_Std"]
    df_results = df_results[final_cols]
    
    df_results.to_csv(OUT_DIR / "regression_master_report_v2.csv", index=False)
    print("\n" + "="*140)
    print("🏆 MASTER TABLE: FULL REGRESSION REPORT (CON MCI STD)")
    print("="*140)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()