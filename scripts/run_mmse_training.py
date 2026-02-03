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
from sklearn.model_selection import KFold

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal
from src.utils import set_seed, clear_memory

# ==========================================
# CONFIGURAZIONE HARDCODED
# ==========================================
# Usiamo il file NUOVO creato nello Step 1
METADATA_FILE = "data/metadata/mmse_experiment_metadata.csv"
# Usiamo il file dei punteggi UNITO (creato in precedenza)
SCORES_FILE = "data/metadata/adresso_FULL_mmse.csv"

def force_regression_config(config):
    """Forza il config per la regressione."""
    config.task = "regression"
    if not hasattr(config, 'model'): config.model = type('obj', (object,), {})()
    config.model.output_dim = 1       
    config.modality = "multimodal_cross_attention"
    # Sovrascriviamo il metadata file nel config object in memoria
    config.data.metadata_file = METADATA_FILE 
    return config

def manual_bias_init(model, init_value=25.0):
    """Inizializza il bias finale a 25 per stabilità."""
    last_layer = model.classifier[-1]
    if isinstance(last_layer, nn.Linear):
        nn.init.constant_(last_layer.bias, init_value)
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.01)

def match_scores(metadata_df, scores_file):
    """Match ID Metadata <-> ID Punteggi"""
    scores_df = pd.read_csv(scores_file)
    scores_df['ID'] = scores_df['ID'].astype(str).str.strip().str.replace('"', '')
    id_to_score = dict(zip(scores_df['ID'], scores_df['Score']))
    
    def get_score(full_id):
        # Pulisce: EN_TEST_adrsdt44_Task_01 -> adrsdt44
        clean = full_id.replace("EN_", "").split("_Task_")[0]
        clean = clean.replace("TEST_", "").strip()
        return id_to_score.get(clean, None)

    metadata_df['mmse_score'] = metadata_df['Subject_ID'].apply(get_score)
    # Ritorna solo chi ha il punteggio (Inglesi)
    return metadata_df.dropna(subset=['mmse_score']).copy().reset_index(drop=True)

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion = nn.L1Loss() # MAE Loss
    
    pbar = tqdm(loader, desc="    Train", leave=False, unit="batch")
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        targets = batch.pop('labels').float()
        
        outputs = model(batch)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'MAE': f"{loss.item():.2f}"})
        
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            targets = batch.pop('labels').float()
            outputs = model(batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    mse = np.mean((np.array(all_preds) - np.array(all_targets))**2)
    return np.sqrt(mse) # RMSE

def predict(model, loader, device):
    model.eval()
    all_preds, ids = [], []
    with torch.no_grad():
        for batch in loader:
            ids.extend(batch.pop('id'))
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            all_preds.extend(outputs.cpu().numpy())
    return ids, np.array(all_preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_regression.yaml")
    args = parser.parse_args()
    
    # Carica e Forza Config
    config = load_config(args.config)
    config = force_regression_config(config) # <--- Qui usa il nuovo file metadata
    
    out_dir = Path("outputs/mmse_regression_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device(config.device)
    
    print(f"\n{'='*60}")
    print("🚀 AVVIO REGRESSIONE MMSE (Metadata Dedicato)")
    print(f"   Metadata: {METADATA_FILE}")
    print(f"{'='*60}")

    # 1. CARICAMENTO DATI
    if not Path(METADATA_FILE).exists():
        print(f"❌ Errore: File {METADATA_FILE} non trovato. Esegui 'create_mmse_metadata.py'.")
        return

    df_full = pd.read_csv(METADATA_FILE)
    
    # Train: Inglese
    df_en = df_full[df_full['Language'] == 'EN'].copy()
    df_train = match_scores(df_en, SCORES_FILE)
    print(f"📊 Dataset Inglese (Train+Test): {len(df_train)} campioni.")

    # Test: Italiano (Zero-Shot)
    df_it = df_full[df_full['Language'] == 'IT'].copy()
    df_it['mmse_score'] = -1.0 # Dummy
    
    # Debug Diagnosi
    print(f"🇮🇹 Dataset Italiano: {len(df_it)} campioni.")
    print(f"   Diagnosi: {list(df_it['Diagnosis'].unique())}") # Dovrebbe includere MCI ora!

    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    test_loader = torch.utils.data.DataLoader(MultimodalDataset(df_it, config, tokenizer), batch_size=32, shuffle=False, collate_fn=collate_multimodal)

    # 2. ENSEMBLE TRAINING
    kf = KFold(n_splits=10, shuffle=True, random_state=config.seed)
    ensemble_preds = np.zeros((len(df_it), 10))
    cv_scores = []

    fold_pbar = tqdm(enumerate(kf.split(df_train)), total=10, desc="Overall Progress")

    for fold_idx, (train_idx, val_idx) in fold_pbar:
        fold_pbar.set_description(f"Fold {fold_idx+1}/10")
        
        train_sub = df_train.iloc[train_idx].reset_index(drop=True)
        val_sub = df_train.iloc[val_idx].reset_index(drop=True)
        
        train_loader = torch.utils.data.DataLoader(MultimodalDataset(train_sub, config, tokenizer), batch_size=16, shuffle=True, collate_fn=collate_multimodal, drop_last=True)
        val_loader = torch.utils.data.DataLoader(MultimodalDataset(val_sub, config, tokenizer), batch_size=16, shuffle=False, collate_fn=collate_multimodal)

        # Init & Bias Fix
        clear_memory()
        model = build_model(config).to(device)
        manual_bias_init(model, init_value=25.0) 
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=len(train_loader)*15)

        best_rmse = float('inf')
        patience = 0
        
        print(f"\n   Training Fold {fold_idx+1}...")
        for epoch in range(12): 
            train_mae = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_rmse = validate(model, val_loader, device)
            
            print(f"   Ep {epoch+1:02d} | Train MAE: {train_mae:.2f} | Val RMSE: {val_rmse:.2f}", end="")
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), out_dir / f"model_fold_{fold_idx}.pt")
                patience = 0
                print(" 🌟")
            else:
                patience += 1
                print("")
                if patience >= 4:
                    print("   🛑 Early Stopping")
                    break
        cv_scores.append(best_rmse)
        
        # Predict
        model.load_state_dict(torch.load(out_dir / f"model_fold_{fold_idx}.pt", weights_only=True))
        _, preds = predict(model, test_loader, device)
        ensemble_preds[:, fold_idx] = preds

    # 3. PLOT FINALE
    print(f"\n📊 CV RMSE Medio: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

    final_preds = np.mean(ensemble_preds, axis=1)
    ids, _ = predict(model, test_loader, device) 
    
    results = []
    for i, pid in enumerate(ids):
        row = df_it[df_it['Subject_ID'] == pid].iloc[0]
        results.append({'Subject_ID': pid, 'Diagnosis': row['Diagnosis'], 'Predicted_MMSE': final_preds[i]})
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "italian_mmse_final.csv", index=False)
    
    print("\n🏆 Medie Predette per Gruppo:")
    print(res_df.groupby('Diagnosis')['Predicted_MMSE'].agg(['mean', 'std', 'count']))

    plt.figure(figsize=(10, 6))
    order = ['CTR', 'MCI', 'MILD-AD', 'AD']
    plot_order = [d for d in order if d in res_df['Diagnosis'].unique()]
    
    sns.boxplot(data=res_df, x='Diagnosis', y='Predicted_MMSE', order=plot_order, palette="Set2")
    sns.stripplot(data=res_df, x='Diagnosis', y='Predicted_MMSE', order=plot_order, color='black', alpha=0.4)
    
    plt.title(f"MMSE Prediction (Ensemble 10-Fold)\nTrain: EN (ADReSSo Full) -> Test: IT (Sicily)\nCV RMSE: {np.mean(cv_scores):.2f}")
    plt.ylabel("Predicted MMSE Score")
    plt.savefig(out_dir / "final_plot.png", dpi=300)
    print(f"\n✅ Grafico salvato: {out_dir / 'final_plot.png'}")

if __name__ == "__main__":
    main()
    # MCI 25 < MMSE < 27
    # AD <= 24
    # CTR >= 27
    # crossdataset zero-shot regression