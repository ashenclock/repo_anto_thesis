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
from sklearn.model_selection import KFold
from scipy.stats import ttest_ind

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal
from src.utils import set_seed, clear_memory

# === CONFIGURAZIONE PERCORSI ===
META_FILE = "data/metadata/mmse_experiment_metadata.csv" 
SCORES_EN_FILE = "data/metadata/adresso_FULL_mmse.csv"
EXCEL_IT_FILE = "data/metadata/dataset_10_02_Foglio_1_-_dataset.csv"

OUT_DIR = Path("outputs/mmse_EN_to_IT_XLMR")
PLOT_DIR = Path("plots/mmse_results_EN_to_IT")

def load_data():
    df_full = pd.read_csv(META_FILE)
    df_en = df_full[(df_full['Language'] == 'EN') & (df_full['Dataset'] == 'ADReSSo_Train')].copy()
    df_scores_en = pd.read_csv(SCORES_EN_FILE)
    df_scores_en['ID'] = df_scores_en['ID'].astype(str).str.strip().str.replace('"', '')
    en_map = dict(zip(df_scores_en['ID'], df_scores_en['Score']))
    df_en['mmse_score'] = df_en['Subject_ID'].apply(lambda x: en_map.get(x.replace("EN_", "").split("_Task_")[0].strip(), np.nan))
    df_en = df_en.dropna(subset=['mmse_score']).reset_index(drop=True)
    df_it = df_full[df_full['Language'] == 'IT'].copy()
    return df_en, df_it

def add_stat_annotation(ax, x1, x2, y, p_val):
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    h = 0.4
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')

def main():
    set_seed(42)
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("⏳ Caricamento configurazione...")
    config = load_config("config.yaml")
    config.task = "regression"
    config.model.output_dim = 1
    config.model.text.name = "xlm-roberta-base"

    print(f"⏳ Pre-scaricamento modelli ({config.model.text.name})...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    _ = AutoModel.from_pretrained(config.model.text.name)
    print("✅ Modelli caricati in cache.")

    df_en, df_it = load_data()
    print(f"🚀 Training su INGLESE (N={len(df_en)}) -> Test su ITALIANO (N={len(df_it)})")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    it_preds_matrix = np.zeros((len(df_it), 5))

    for f_id, (train_idx, val_idx) in enumerate(kf.split(df_en)):
        print(f"\n🔹 Inizio Fold {f_id}...")
        train_df = df_en.iloc[train_idx].reset_index(drop=True)
        val_df = df_en.iloc[val_idx].reset_index(drop=True)
        
        train_loader = torch.utils.data.DataLoader(MultimodalDataset(train_df, config, tokenizer), batch_size=16, shuffle=True, collate_fn=collate_multimodal, num_workers=0)
        val_loader = torch.utils.data.DataLoader(MultimodalDataset(val_df, config, tokenizer), batch_size=16, shuffle=False, collate_fn=collate_multimodal, num_workers=0)

        clear_memory()
        print("   🏗️  Costruzione modello SOTA...")
        model = build_model(config).to(device)
        nn.init.constant_(model.classifier[-1].bias, 24.0) 
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=len(train_loader)*20)
        criterion = nn.L1Loss()

        best_mae = 100.0
        pbar = tqdm(range(20), desc=f"   Fold {f_id} Training")
        for epoch in pbar:
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                targets = batch.pop('labels').float()
                outputs = model(batch).view(-1) # FIX: view(-1) invece di squeeze
                loss = criterion(outputs, targets)
                optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()

            model.eval()
            v_preds = []
            v_targets = []
            with torch.no_grad():
                for b in val_loader:
                    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                    target_b = b.pop('labels').float()
                    output_b = model(b).view(-1) # FIX: view(-1)
                    
                    # Convertiamo in liste per evitare errori di iterazione su scalari
                    v_preds.extend(output_b.cpu().numpy().flatten().tolist())
                    v_targets.extend(target_b.cpu().numpy().flatten().tolist())
            
            m = mean_absolute_error(v_targets, v_preds)
            pbar.set_postfix({'MAE': f"{m:.2f}"})
            if m < best_mae:
                best_mae = m
                torch.save(model.state_dict(), OUT_DIR / f"model_en_fold_{f_id}.pt")

        # --- ZERO-SHOT INFERENCE ON ITALIAN ---
        print(f"   🧪 Inferenza Zero-Shot su Italia (Modello Fold {f_id})...")
        model.load_state_dict(torch.load(OUT_DIR / f"model_en_fold_{f_id}.pt", weights_only=True))
        model.eval()
        it_loader = torch.utils.data.DataLoader(MultimodalDataset(df_it, config, tokenizer), batch_size=16, shuffle=False, collate_fn=collate_multimodal, num_workers=0)
        
        curr_it_preds = []
        with torch.no_grad():
            for b in it_loader:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                out = model(b).view(-1) # FIX: view(-1)
                curr_it_preds.extend(out.cpu().numpy().flatten().tolist())
        it_preds_matrix[:, f_id] = curr_it_preds

    # Aggregazione finale
    df_it['Predicted_MMSE'] = np.mean(it_preds_matrix, axis=1)

    # Salva risultati in CSV
    df_it.to_csv(OUT_DIR / "italy_zeroshot_results.csv", index=False)

    # Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    custom_pal = {"CTR": "#66c2a5", "MCI": "#8da0cb", "MILD-AD": "#fc8d62"}
    order = ['CTR', 'MCI', 'MILD-AD']
    ax = sns.boxplot(data=df_it, x='Diagnosis', y='Predicted_MMSE', order=order, palette=custom_pal, width=0.6, fliersize=0)
    sns.stripplot(data=df_it, x='Diagnosis', y='Predicted_MMSE', order=order, color='.3', alpha=0.4, jitter=0.2, size=5)
    
    ctr_vals = df_it[df_it['Diagnosis']=='CTR']['Predicted_MMSE']
    ad_vals = df_it[df_it['Diagnosis']=='MILD-AD']['Predicted_MMSE']
    p_val = ttest_ind(ctr_vals, ad_vals)[1]
    add_stat_annotation(ax, 0, 2, 29, p_val)

    plt.title("ZERO-SHOT: IT Diagnosis from EN-trained Model\n(XLM-RoBERTa + XLS-R)", fontsize=18, fontweight='bold')
    plt.axhline(y=24, color='red', linestyle='--', alpha=0.6)
    plt.ylim(10, 32)
    plt.ylabel("Predicted MMSE Score", fontsize=14)
    
    save_path = PLOT_DIR / "ITALY_ZERO_SHOT_FROM_EN.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ ESPERIMENTO COMPLETATO!")
    print(f"📊 Medie IT predette: \n{df_it.groupby('Diagnosis')['Predicted_MMSE'].mean()}")
    print(f"🖼️  Grafico salvato in: {save_path}")

if __name__ == "__main__":
    main()