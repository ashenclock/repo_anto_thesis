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

# === CONFIGURAZIONE ===
METADATA_FILE = "data/metadata/mmse_experiment_metadata.csv"
SCORES_FILE = "data/metadata/adresso_FULL_MOCA_converted.csv"

def force_regression_config(config):
    config.task = "regression"
    if not hasattr(config, 'model'): config.model = type('obj', (object,), {})()
    config.model.output_dim = 1       
    config.modality = "multimodal_cross_attention"
    config.data.metadata_file = METADATA_FILE 
    config.model.audio.trainable_encoder = False # Usiamo i vettori pre-computati
    return config

def manual_bias_init(model, init_value=22.0): 
    """Inizializza il bias finale per aiutare la convergenza su scala MoCA."""
    last_layer = model.classifier[-1]
    if isinstance(last_layer, nn.Linear):
        print(f"      ⚖️  Bias Init: {init_value}")
        nn.init.constant_(last_layer.bias, init_value)
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.01)

def match_scores(metadata_df, scores_file):
    scores_df = pd.read_csv(scores_file)
    scores_df['ID'] = scores_df['ID'].astype(str).str.strip().str.replace('"', '')
    id_to_score = dict(zip(scores_df['ID'], scores_df['Score']))
    
    def get_score(full_id):
        clean = full_id.replace("EN_", "").split("_Task_")[0].replace("TEST_", "").strip()
        return id_to_score.get(clean, None)

    metadata_df['mmse_score'] = metadata_df['Subject_ID'].apply(get_score)
    return metadata_df.dropna(subset=['mmse_score']).copy().reset_index(drop=True)

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
    return np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets))**2))

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
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    config = force_regression_config(config)
    
    out_dir = Path("outputs/moca_regression_final")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device(config.device)
    
    print(f"\n{'='*60}")
    print("🚀 AVVIO REGRESSIONE MoCA (Huber Loss + Diff LR)")
    print(f"{'='*60}")

    # 1. Caricamento Dati
    df_full = pd.read_csv(METADATA_FILE)
    df_en = df_full[df_full['Language'] == 'EN'].copy()
    df_train = match_scores(df_en, SCORES_FILE)
    df_it = df_full[df_full['Language'] == 'IT'].copy()

    print(f"📊 Train Set (EN): {len(df_train)} | Test Set (IT): {len(df_it)}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    it_test_loader = torch.utils.data.DataLoader(
        MultimodalDataset(df_it, config, tokenizer), 
        batch_size=32, shuffle=False, collate_fn=collate_multimodal
    )

    # 2. Cross-Validation 10-Fold
    kf = KFold(n_splits=10, shuffle=True, random_state=config.seed)
    ensemble_preds = np.zeros((len(df_it), 10))
    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        print(f"\n🔄 FOLD {fold_idx+1}/10")
        
        train_sub = df_train.iloc[train_idx].reset_index(drop=True)
        val_sub = df_train.iloc[val_idx].reset_index(drop=True)
        
        t_loader = torch.utils.data.DataLoader(
            MultimodalDataset(train_sub, config, tokenizer), 
            batch_size=16, shuffle=True, collate_fn=collate_multimodal, drop_last=True
        )
        v_loader = torch.utils.data.DataLoader(
            MultimodalDataset(val_sub, config, tokenizer), 
            batch_size=16, shuffle=False, collate_fn=collate_multimodal
        )

        clear_memory()
        model = build_model(config).to(device)
        manual_bias_init(model, init_value=0.0) 

        # Optimizer con Learning Rate differenziato
        optimizer = torch.optim.AdamW([
            {'params': model.text_encoder.parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 5e-4},
            {'params': model.cross_attn_T_A.parameters(), 'lr': 1e-4},
            {'params': model.cross_attn_A_T.parameters(), 'lr': 1e-4},
        ], weight_decay=0.01)

        criterion = nn.HuberLoss(delta=1.0) 
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=40, num_training_steps=len(t_loader)*25)

        best_rmse = float('inf')
        patience = 0
        
        for epoch in range(25):
            model.train()
            train_loss = 0
            pbar = tqdm(t_loader, desc=f"      Ep {epoch+1:02d}", leave=False)
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                targets = batch.pop('labels').float()
                outputs = model(batch)
                
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            val_rmse = validate(model, v_loader, device)
            avg_train_loss = train_loss / len(t_loader)
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), out_dir / f"model_fold_{fold_idx}.pt")
                patience = 0
                print(f"      Ep {epoch+1:02d} | Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.2f} 🌟")
            else:
                patience += 1
                if patience >= 5: 
                    print(f"      🛑 Early Stop fold {fold_idx+1}")
                    break
        
        cv_scores.append(best_rmse)
        
        # Inference su IT con il miglior modello di questo fold
        model.load_state_dict(torch.load(out_dir / f"model_fold_{fold_idx}.pt", weights_only=True))
        _, preds = predict(model, it_test_loader, device)
        ensemble_preds[:, fold_idx] = preds

    # 3. Aggregazione Finale e Plot
    print(f"\n📊 CV RMSE Medio (MoCA): {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

    final_preds = np.mean(ensemble_preds, axis=1)
    ids, _ = predict(model, it_test_loader, device)
    
    results = []
    for i, pid in enumerate(ids):
        row = df_it[df_it['Subject_ID'] == pid].iloc[0]
        results.append({
            'Subject_ID': pid, 
            'Diagnosis': row['Diagnosis'], 
            'Predicted_MoCA': final_preds[i]
        })
    
    res_df = pd.DataFrame(results)
    # Rinomina per il grafico se vuoi vedere MCI
    res_df['Diagnosis'] = res_df['Diagnosis'].replace({'MILD-AD': 'AD'})
    res_df.to_csv(out_dir / "italian_moca_final.csv", index=False)
    
    print("\n🏆 Medie MoCA Predette per Gruppo:")
    print(res_df.groupby('Diagnosis')['Predicted_MoCA'].agg(['mean', 'std', 'count']))

    plt.figure(figsize=(10, 6))
    order = ['CTR', 'MCI', 'AD']
    plot_order = [d for d in order if d in res_df['Diagnosis'].unique()]
    
    sns.boxplot(data=res_df, x='Diagnosis', y='Predicted_MoCA', order=plot_order, palette="magma")
    sns.stripplot(data=res_df, x='Diagnosis', y='Predicted_MoCA', order=plot_order, color='black', alpha=0.4)
    
    plt.title(f"MoCA Zero-Shot Regression (10-Fold Ensemble)\nCV RMSE: {np.mean(cv_scores):.2f}")
    plt.ylabel("Predicted MoCA Score")
    plt.savefig(out_dir / "final_moca_plot.png", dpi=300)
    print(f"\n✅ Grafico salvato in: {out_dir / 'final_moca_plot.png'}")

if __name__ == "__main__":
    main()