import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, ttest_ind

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal

# --- CONFIG ---
# Modelli addestrati sui 139 italiani
MODEL_DIR = Path("outputs/mmse_ONLY_XLMR_139")
META_FILE = "data/metadata/mmse_experiment_metadata.csv"
SCORES_EN_FILE = "data/metadata/adresso_FULL_mmse.csv"
PLOT_DIR = Path("plots/mmse_results_fancy")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    print("📂 Caricamento dati...")
    df_me = pd.read_csv(META_FILE)
    df_it = df_me[df_me['Language'] == 'IT'].copy()
    df_en = df_me[df_me['Dataset'] == 'ADReSSo_Test'].copy()
    
    # Match punteggi reali inglesi per RMSE
    df_scores = pd.read_csv(SCORES_EN_FILE)
    df_scores['ID'] = df_scores['ID'].astype(str).str.strip().str.replace('"', '')
    en_gt_map = dict(zip(df_scores['ID'], df_scores['Score']))
    
    df_en['Actual_MMSE'] = df_en['Subject_ID'].apply(lambda x: en_gt_map.get(x.replace("EN_", "").replace("TEST_", "").split("_Task_")[0].strip(), np.nan))
    df_en = df_en.dropna(subset=['Actual_MMSE']).reset_index(drop=True)
    
    df_it['mmse_score'] = 0.0 # Dummy
    df_en['mmse_score'] = 0.0 # Dummy
    return df_it, df_en

def add_stat_annotation(ax, x1, x2, y, p_val):
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    h = 0.5
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color='black', fontsize=16, fontweight='bold')

def main():
    device = torch.device("cuda")
    config = load_config("config.yaml")
    config.task = "regression"; config.model.output_dim = 1
    config.model.text.name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    
    df_it, df_en = load_data()
    model_files = sorted(list(MODEL_DIR.glob("model_fold_*.pt")))
    
    it_preds_matrix = np.zeros((len(df_it), len(model_files)))
    en_preds_matrix = np.zeros((len(df_en), len(model_files)))

    print(f"🚀 Running Ensemble Inference with {len(model_files)} models (Source: ITALY)...")
    for i, m_path in enumerate(model_files):
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(m_path, map_location=device, weights_only=True))
        model.eval()
        
        it_loader = torch.utils.data.DataLoader(MultimodalDataset(df_it, config, tokenizer), batch_size=16, collate_fn=collate_multimodal)
        en_loader = torch.utils.data.DataLoader(MultimodalDataset(df_en, config, tokenizer), batch_size=16, collate_fn=collate_multimodal)
        
        with torch.no_grad():
            # IT Inference
            preds_it = []
            for b in it_loader:
                b_in = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items() if k!='id'}
                preds_it.extend(model(b_in).view(-1).cpu().numpy().flatten().tolist())
            it_preds_matrix[:, i] = preds_it
            
            # EN Inference
            preds_en = []
            for b in en_loader:
                b_in = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items() if k!='id'}
                preds_en.extend(model(b_in).view(-1).cpu().numpy().flatten().tolist())
            en_preds_matrix[:, i] = preds_en

    df_it['Predicted_MMSE'] = np.mean(it_preds_matrix, axis=1)
    df_en['Predicted_MMSE'] = np.mean(en_preds_matrix, axis=1)

    # --- CALCOLO METRICHE ---
    # In-Domain (IT) - Nota: qui il MAE è basso per via dei fallback usati nel training
    mae_it = mean_absolute_error(df_it['Diagnosis'].map({'CTR':28,'MCI':26,'MILD-AD':22}), df_it['Predicted_MMSE']) # Stima indicativa
    
    # Zero-Shot (EN) - METRICA REALE
    mse_en = mean_squared_error(df_en['Actual_MMSE'], df_en['Predicted_MMSE'])
    rmse_en = np.sqrt(mse_en)
    mae_en = mean_absolute_error(df_en['Actual_MMSE'], df_en['Predicted_MMSE'])
    r_en, _ = pearsonr(df_en['Actual_MMSE'], df_en['Predicted_MMSE'])

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    custom_pal = {"CTR": "#66c2a5", "MCI": "#8da0cb", "MILD-AD": "#fc8d62", "AD": "#fc8d62"}

    # ITALY
    sns.boxplot(data=df_it, x='Diagnosis', y='Predicted_MMSE', order=['CTR', 'MCI', 'MILD-AD'], palette=custom_pal, ax=ax1, width=0.8, linewidth=2, fliersize=0)
    sns.stripplot(data=df_it, x='Diagnosis', y='Predicted_MMSE', order=['CTR', 'MCI', 'MILD-AD'], color='.3', alpha=0.4, jitter=0.2, ax=ax1)
    p_it = ttest_ind(df_it[df_it['Diagnosis']=='CTR']['Predicted_MMSE'], df_it[df_it['Diagnosis']=='MILD-AD']['Predicted_MMSE'])[1]
    add_stat_annotation(ax1, 0, 2, 29.5, p_it)
    ax1.set_title("ITALY Sicily Full Set (In-Domain)\nModel trained on Italian patients", fontsize=20, fontweight='bold')
    ax1.set_ylim(10, 32)

    # ENGLISH
    sns.boxplot(data=df_en, x='Diagnosis', y='Predicted_MMSE', order=['CTR', 'AD'], palette=custom_pal, ax=ax2, width=0.7, linewidth=2, fliersize=0)
    sns.stripplot(data=df_en, x='Diagnosis', y='Predicted_MMSE', order=['CTR', 'AD'], color='.3', alpha=0.4, jitter=0.2, ax=ax2)
    p_en = ttest_ind(df_en[df_en['Diagnosis']=='CTR']['Predicted_MMSE'], df_en[df_en['Diagnosis']=='AD']['Predicted_MMSE'])[1]
    add_stat_annotation(ax2, 0, 1, 28, p_en)
    ax2.set_title(f"UK ADReSSo Test (Zero-Shot)\nRMSE: {rmse_en:.2f} | r: {r_en:.2f}", fontsize=20, fontweight='bold')
    ax2.set_ylim(10, 32)

    plt.suptitle("Cross-Lingual MMSE Regression (Source: ITALY -> Target: UK)", fontsize=28, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    save_path = PLOT_DIR / "MMSE_RMSE_IT_TO_EN_FINAL.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print("\n" + "="*50)
    print(f"🏆 RISULTATI FINALI (IT -> EN)")
    print("="*50)
    print(f"🇬🇧 ENGLISH TEST RMSE: {rmse_en:.2f}")
    print(f"   (Pappagari Ref:  3.85)")
    print(f"   MAE:             {mae_en:.2f}")
    print(f"   Pearson r:       {r_en:.3f}")
    print("-" * 50)
    print(f"✅ Report salvato in: {save_path}")

if __name__ == "__main__":
    main()