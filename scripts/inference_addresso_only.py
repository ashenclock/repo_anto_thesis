import sys
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import pearsonr

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal
from src.utils import set_seed

# === CONFIGURAZIONE ===
# Dove sono i modelli appena addestrati
MODEL_DIR = Path("outputs/moca_regression_XLMR_139") 
# Dove salvare i risultati inglesi
OUT_DIR = Path("outputs/adresso_inference_XLMR")
PLOT_DIR = Path("plots/moca_sota_XLMR")

METADATA_FILE = "data/metadata/mmse_experiment_metadata.csv" 
SCORES_FILE = "data/metadata/adresso_FULL_mmse.csv"

def mmse_to_moca_trzepacz(mmse):
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

def match_scores_and_convert(df_meta, path_scores):
    df_scores = pd.read_csv(path_scores)
    df_scores['ID'] = df_scores['ID'].astype(str).str.strip().str.replace('"', '')
    id_map = dict(zip(df_scores['ID'], df_scores['Score']))
    
    def get_gt_moca(full_id):
        clean = full_id.replace("EN_", "").replace("TEST_", "").split("_Task_")[0].strip()
        mmse = id_map.get(clean, np.nan)
        return mmse_to_moca_trzepacz(mmse)

    df_meta['moca_ground_truth'] = df_meta['Subject_ID'].apply(get_gt_moca)
    df_meta = df_meta.dropna(subset=['moca_ground_truth']).reset_index(drop=True)
    df_meta['mmse_score'] = 0.0 # Dummy
    return df_meta

def main():
    print(f"\n{'='*60}")
    print(f"🌍 XLM-RoBERTa ZERO-SHOT INFERENCE (Solo Inglese)")
    print(f"{'='*60}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Configurazione (deve matchare quella del training)
    config = load_config("config.yaml")
    config.task = "regression"
    config.model.output_dim = 1
    config.modality = "multimodal_cross_attention"
    config.model.text.name = "xlm-roberta-base" # IMPORTANTE
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)

    # Carica Dati Inglesi
    df_full = pd.read_csv(METADATA_FILE)
    df_en_raw = df_full[df_full['Dataset'] == 'ADReSSo_Test'].copy()
    
    try:
        df_en = match_scores_and_convert(df_en_raw, SCORES_FILE)
    except Exception as e:
        print(f"❌ Errore caricamento dati: {e}")
        return

    print(f"📊 Dataset ADReSSo Test: {len(df_en)} pazienti.")

    test_loader = torch.utils.data.DataLoader(
        MultimodalDataset(df_en, config, tokenizer), 
        batch_size=16, shuffle=False, collate_fn=collate_multimodal
    )

    # Ensemble dei modelli salvati
    model_files = sorted(list(MODEL_DIR.glob("model_*.pt")))
    if not model_files:
        print(f"❌ Nessun modello trovato in {MODEL_DIR}")
        return
    
    print(f"🔄 Ensemble su {len(model_files)} modelli XLM-R...")
    preds_matrix = np.zeros((len(df_en), len(model_files)))
    
    for i, model_path in enumerate(model_files):
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        curr_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch).squeeze(-1)
                curr_preds.extend(out.cpu().numpy())
        preds_matrix[:, i] = curr_preds

    # Risultati Finali
    df_en['Predicted_MoCA'] = np.mean(preds_matrix, axis=1)
    
    mae = mean_absolute_error(df_en['moca_ground_truth'], df_en['Predicted_MoCA'])
    corr, p_val = pearsonr(df_en['moca_ground_truth'], df_en['Predicted_MoCA'])
    
    print("\n" + "="*60)
    print(f"🏆 RISULTATI XLM-R su ADReSSo Test")
    print("-" * 60)
    print(f"   MAE:  {mae:.2f} (Target Pappagari: 3.85)")
    print(f"   Corr: {corr:.3f} (p={p_val:.3e})")
    print("="*60)
    
    # Salvataggio
    df_en.to_csv(OUT_DIR / "adresso_xlmr_results.csv", index=False)
    
    # Plot Scatter
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    palette = {"CTR": "green", "AD": "orange"}
    
    sns.scatterplot(data=df_en, x='moca_ground_truth', y='Predicted_MoCA', hue='Diagnosis', palette=palette, s=80)
    sns.regplot(data=df_en, x='moca_ground_truth', y='Predicted_MoCA', scatter=False, color='red')
    plt.plot([0, 30], [0, 30], ls="--", c=".3")
    
    plt.title(f"XLM-R Zero-Shot (IT->EN)\nMAE: {mae:.2f} | r: {corr:.2f}")
    plt.xlabel("Ground Truth MoCA (Converted)")
    plt.ylabel("Predicted MoCA")
    
    plt.savefig(PLOT_DIR / "xlmr_adresso_final_scatter.png", dpi=300)
    print(f"✅ Grafico salvato: {PLOT_DIR}/xlmr_adresso_final_scatter.png")

if __name__ == "__main__":
    main()