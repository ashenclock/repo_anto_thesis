import argparse
import sys
import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# Setup Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils import set_seed, clear_memory
from src.data import MultimodalDataset, collate_multimodal
from src.engine import train_epoch, evaluate
from src.models import build_model
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

def get_datasets(config, mode):
    df = pd.read_csv(config.data.metadata_file)
    
    # Definizioni Dataset
    df_it = df[df['Dataset'] == 'Sicily'].copy()
    # Train inglese: ADReSSo Train
    df_en_train = df[df['Dataset'].isin(['ADReSSo_Train', 'ADReSSo'])].copy()
    # Test inglese: ADReSSo Test
    df_en_test = df[df['Dataset'] == 'ADReSSo_Test'].copy()

    if mode == "EN_to_IT":
        print(f"🌍 ENSEMBLE: Train su INGLESE (10 Fold) -> Test su ITALIANO (Mean Prob)")
        return df_en_train, df_it # Train Full, Target Full
    elif mode == "IT_to_EN":
        print(f"🌍 ENSEMBLE: Train su ITALIANO (10 Fold) -> Test su INGLESE (Mean Prob)")
        return df_it, df_en_test
    else:
        raise ValueError("Mode non supportato")

def train_single_fold(config, train_df, val_df, fold_idx, device):
    """Addestra un singolo modello su uno split dei dati"""
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    
    train_ds = MultimodalDataset(train_df, config, tokenizer)
    val_ds = MultimodalDataset(val_df, config, tokenizer)
    
    # drop_last=True per evitare crash su batch singoli
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.training.batch_size, shuffle=True, 
        collate_fn=collate_multimodal, num_workers=4, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.training.batch_size, shuffle=False, 
        collate_fn=collate_multimodal, num_workers=4
    )
    
    model = build_model(config).to(device)
    
    # --- WEIGHTED LOSS (1:3) ---
    # Fondamentale per alzare la Sensitivity
    weights = torch.tensor([1.0, 3.0]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config.training.learning_rate), 
        weight_decay=config.training.weight_decay
    )
    
    total_steps = len(train_loader) * config.training.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    
    best_f1 = 0
    patience = 0
    best_state = None
    
    print(f"\n🔹 START FOLD {fold_idx+1}/10")
    
    for epoch in range(config.training.epochs):
        t_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        _, metrics, _ = evaluate(model, val_loader, device, torch.nn.CrossEntropyLoss())
        
        # --- STAMPA PROGRESSI (Corretto) ---
        print(f"   Ep {epoch+1:02d} | Loss: {t_loss:.4f} | Val F1: {metrics['f1']:.4f} | Sens: {metrics['sensitivity']:.4f} | Spec: {metrics['specificity']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_state = model.state_dict()
            patience = 0
            print(f"      🌟 New Best Model found! (F1: {best_f1:.4f})") 
        else:
            patience += 1
            if patience >= config.training.early_stopping_patience:
                print("   🛑 Early Stopping")
                break
                
    # Ritorna il modello migliore di questo fold
    if best_state is None: best_state = model.state_dict() # Fallback
    model.load_state_dict(best_state)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", type=str, default="EN_to_IT")
    args = parser.parse_args()

    # Config Setup
    with open(args.config) as f: raw_config = yaml.safe_load(f)
    raw_config['output_dir'] = f"outputs/ensemble_cross_{args.mode}"
    config = Config(raw_config)
    
    # Creazione cartella output
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(config.seed)
    device = torch.device(config.device)
    
    # 1. Carica Dati
    df_train_full, df_target = get_datasets(config, args.mode)
    
    # Dataset Target (Italiano) su cui faremo l'ensemble
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    target_ds = MultimodalDataset(df_target, config, tokenizer)
    target_loader = torch.utils.data.DataLoader(
        target_ds, batch_size=config.training.batch_size, shuffle=False, 
        collate_fn=collate_multimodal, num_workers=4
    )
    
    # Matrice per accumulare le probabilità: [N_Samples, N_Folds]
    all_folds_probs = np.zeros((len(df_target), 10))
    
    # 2. Loop 10-Fold CV sul Training Set
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.seed)
    
    labels_target = [] 
    ids_target = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train_full, df_train_full['Diagnosis'])):
        clear_memory()
        
        # Split Data
        train_sub = df_train_full.iloc[train_idx]
        val_sub = df_train_full.iloc[val_idx]
        
        # Train Model
        model = train_single_fold(config, train_sub, val_sub, fold_idx, device)
        
        # Save Model
        torch.save(model.state_dict(), out_dir / f"model_fold_{fold_idx}.pt")
        
        # Predict on Target (IT)
        model.eval()
        _, _, details = evaluate(model, target_loader, device, torch.nn.CrossEntropyLoss())
        
        # Salva probabilità (Classe 1)
        probs = np.array(details['probs']) 
        all_folds_probs[:, fold_idx] = probs
        
        # Al primo giro salviamo ID e Label
        if fold_idx == 0:
            labels_target = np.array(details['labels'])
            ids_target = details['ids']
            
        print(f"   ✅ Fold {fold_idx+1} Completato.")

    # 3. AGGREGAZIONE (Mean Probability)
    print("\n" + "="*60)
    print("📊 CALCOLO ENSEMBLE (Mean Probability 10 Folds)")
    print("="*60)
    
    final_probs = np.mean(all_folds_probs, axis=1) # Media sulle colonne
    final_preds = (final_probs >= 0.5).astype(int)
    
    # Metriche
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    acc = accuracy_score(labels_target, final_preds)
    f1 = f1_score(labels_target, final_preds, average='weighted')
    
    tn, fp, fn, tp = confusion_matrix(labels_target, final_preds, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp+fn)>0 else 0
    spec = tn / (tn + fp) if (tn+fp)>0 else 0
    
    print(f"🏆 RISULTATO FINALE ENSEMBLE:")
    print(f"   Accuracy:    {acc:.4f}")
    print(f"   F1-Score:    {f1:.4f}")
    print(f"   Sensitivity: {sens:.4f}")
    print(f"   Specificity: {spec:.4f}")
    print("="*60)
    
    # Salva CSV
    df_res = pd.DataFrame({
        "ID": ids_target,
        "Label": labels_target,
        "Ensemble_Prob": final_probs,
        "Ensemble_Pred": final_preds
    })
    
    for i in range(10):
        df_res[f"Prob_Fold_{i}"] = all_folds_probs[:, i]
        
    out_file = out_dir / "ensemble_results.csv"
    df_res.to_csv(out_file, index=False)
    print(f"💾 File risultati salvato: {out_file}")

if __name__ == "__main__":
    main()