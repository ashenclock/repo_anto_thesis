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
from scipy.stats import pearsonr, ttest_ind

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal

# === CONFIGURAZIONE PERCORSI ===
MODEL_DIR = Path("outputs/mmse_EN_to_IT_XLMR")
META_FILE = "data/metadata/mmse_experiment_metadata.csv" 
SCORES_EN_FILE = "data/metadata/adresso_FULL_mmse.csv"
PLOT_DIR = Path("plots/EN_to_ALL_FINAL")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    print("📂 Caricamento Dati...")
    df_full = pd.read_csv(META_FILE)
    
    # 1. EN Test
    df_en = df_full[df_full['Dataset'] == 'ADReSSo_Test'].copy()
    df_scores = pd.read_csv(SCORES_EN_FILE)
    df_scores['ID'] = df_scores['ID'].astype(str).str.strip().str.replace('"', '')
    en_gt_map = dict(zip(df_scores['ID'], df_scores['Score']))
    df_en['Actual_MMSE'] = df_en['Subject_ID'].apply(lambda x: en_gt_map.get(x.replace("EN_", "").replace("TEST_", "").split("_Task_")[0].strip(), np.nan))
    df_en = df_en.dropna(subset=['Actual_MMSE']).reset_index(drop=True)
    df_en['mmse_score'] = 0.0 # Dummy per loader

    # 2. IT Full
    df_it = df_full[df_full['Language'] == 'IT'].copy()
    df_it['mmse_score'] = 0.0 # Dummy per loader
    
    return df_en, df_it

def add_stat_annotation(ax, x1, x2, y, p_val):
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    h = 0.4
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("config.yaml")
    config.task = "regression"; config.model.output_dim = 1
    config.model.text.name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)

    df_en, df_it = load_data()
    
    model_files = sorted(list(MODEL_DIR.glob("model_*.pt")))
    if not model_files:
        print(f"❌ Errore: Nessun modello trovato in {MODEL_DIR}")
        return

    en_preds_matrix = np.zeros((len(df_en), len(model_files)))
    it_preds_matrix = np.zeros((len(df_it), len(model_files)))

    print(f"🚀 Avvio Inferenza Ensemble con {len(model_files)} modelli...")

    for i, m_path in enumerate(model_files):
        print(f"   Modello {m_path.name}...")
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(m_path, map_location=device, weights_only=True))
        model.eval()

        # Inferenza EN
        en_loader = torch.utils.data.DataLoader(MultimodalDataset(df_en, config, tokenizer), batch_size=16, shuffle=False, collate_fn=collate_multimodal)
        with torch.no_grad():
            preds = []
            for b in en_loader:
                b_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items() if k != 'id'}
                preds.extend(model(b_in).view(-1).cpu().numpy().flatten().tolist())
            en_preds_matrix[:, i] = preds

        # Inferenza IT
        it_loader = torch.utils.data.DataLoader(MultimodalDataset(df_it, config, tokenizer), batch_size=16, shuffle=False, collate_fn=collate_multimodal)
        with torch.no_grad():
            preds = []
            for b in it_loader:
                b_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items() if k != 'id'}
                preds.extend(model(b_in).view(-1).cpu().numpy().flatten().tolist())
            it_preds_matrix[:, i] = preds

    # Media delle predizioni (Ensemble)
    df_en['Pred_MMSE'] = np.mean(en_preds_matrix, axis=1)
    df_it['Pred_MMSE'] = np.mean(it_preds_matrix, axis=1)

    # --- CALCOLO METRICHE FINALI ---
    mse_en = mean_squared_error(df_en['Actual_MMSE'], df_en['Pred_MMSE'])
    rmse_en = np.sqrt(mse_en)
    mae_en = mean_absolute_error(df_en['Actual_MMSE'], df_en['Pred_MMSE'])
    corr_en, _ = pearsonr(df_en['Actual_MMSE'], df_en['Pred_MMSE'])

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    pal = {"CTR": "#66c2a5", "MCI": "#8da0cb", "MILD-AD": "#fc8d62", "AD": "#fc8d62"}

    # EN PLOT
    sns.boxplot(data=df_en, x='Diagnosis', y='Pred_MMSE', order=['CTR', 'AD'], palette=pal, ax=ax1, width=0.6, linewidth=2, fliersize=0)
    sns.stripplot(data=df_en, x='Diagnosis', y='Pred_MMSE', order=['CTR', 'AD'], color='.3', alpha=0.4, jitter=0.2, ax=ax1)
    
    p_en = ttest_ind(df_en[df_en['Diagnosis']=='CTR']['Pred_MMSE'], df_en[df_en['Diagnosis']=='AD']['Pred_MMSE'])[1]
    add_stat_annotation(ax1, 0, 1, 28, p_en)
    
    ax1.set_title(f"UK ADReSSo Test Set\nRMSE: {rmse_en:.2f} | MAE: {mae_en:.2f}", fontsize=18, fontweight='bold')
    ax1.set_ylim(10, 32)
    ax1.set_ylabel("Predicted MMSE Score")

    # IT PLOT
    sns.boxplot(data=df_it, x='Diagnosis', y='Pred_MMSE', order=['CTR', 'MCI', 'MILD-AD'], palette=pal, ax=ax2, width=0.7, linewidth=2, fliersize=0)
    sns.stripplot(data=df_it, x='Diagnosis', y='Pred_MMSE', order=['CTR', 'MCI', 'MILD-AD'], color='.3', alpha=0.4, jitter=0.2, ax=ax2)
    
    p_it = ttest_ind(df_it[df_it['Diagnosis']=='CTR']['Pred_MMSE'], df_it[df_it['Diagnosis']=='MILD-AD']['Pred_MMSE'])[1]
    add_stat_annotation(ax2, 0, 2, 29, p_it)
    
    ax2.set_title("ITALY Sicily Full Set\nZero-Shot Cross-Lingual", fontsize=18, fontweight='bold')
    ax2.set_ylim(10, 32)
    ax2.set_ylabel("Predicted MMSE Score")

    plt.suptitle(f"Multilingual Regression: XLM-RoBERTa + XLS-R\nOverall EN-RMSE: {rmse_en:.2f} (Benchmark: 3.85)", fontsize=24, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    save_path = PLOT_DIR / "FINAL_REPORT_RMSE.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*50)
    print(f"🏆 RISULTATI FINALI")
    print("="*50)
    print(f"🇬🇧 ENGLISH TEST RMSE: {rmse_en:.2f}")
    print(f"   (Pappagari Ref:  3.85)")
    print(f"   MAE:             {mae_en:.2f}")
    print(f"   Pearson r:       {corr_en:.3f}")
    print("-" * 50)
    print(f"🇮🇹 ITALY MEDIE PREDETTE:")
    print(df_it.groupby('Diagnosis')['Pred_MMSE'].mean())
    print("="*50)
    print(f"✅ Report salvato in: {save_path}")

if __name__ == "__main__":
    main()