import argparse
import sys
import yaml
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Setup Path per importare i moduli src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, load_config
from src.utils import set_seed, clear_memory
from src.data import MultimodalDataset, collate_multimodal
from src.engine import evaluate
from src.models import build_model

def load_data_splits(config):
    """
    Replica ESATTAMENTE la logica di split usata in run_balanced_mix.py
    per garantire che il Test Set sia quello 'non visto'.
    """
    df = pd.read_csv(config.data.metadata_file)
    
    # 1. Recupera la parte Italiana
    df_sicily = df[df['Dataset'] == 'Sicily'].copy()
    
    # REPLICA LO SPLIT DEL TRAINING (Usando lo stesso Seed 42)
    # In run_balanced_mix.py avevi usato test_size=0.20
    _, it_test = train_test_split(
        df_sicily, 
        test_size=0.20, 
        stratify=df_sicily['Diagnosis'], 
        random_state=42
    )
    
    # 2. Recupera la parte Inglese (Test Set Ufficiale)
    # In ADReSSo, il test set è separato alla fonte nel file metadata
    en_test = df[df['Dataset'] == 'ADReSSo_Test'].copy()
    
    return it_test, en_test

def run_evaluation(model, df, config, tokenizer, dataset_name):
    print(f"\n{'='*50}")
    print(f"📊 VALUTAZIONE SU: {dataset_name} ({len(df)} samples)")
    print(f"{'='*50}")
    
    if len(df) == 0:
        print("⚠️ Dataset vuoto!")
        return

    # Crea Dataloader
    ds = MultimodalDataset(df, config, tokenizer)
    loader = torch.utils.data.DataLoader(
        ds, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        collate_fn=collate_multimodal,
        num_workers=4
    )
    
    device = torch.device(config.device)
    # Usiamo CrossEntropy perché stiamo valutando un classificatore
    criterion = torch.nn.CrossEntropyLoss()
    
    # Esegue Evaluate
    _, metrics, details = evaluate(model, loader, device, criterion)
    
    print("-" * 30)
    print(f"✅ Accuracy:    {metrics['accuracy']:.4f}")
    print(f"✅ F1-Score:    {metrics['f1']:.4f}")
    print(f"✅ Sensitivity: {metrics['sensitivity']:.4f} (Recall AD)")
    print(f"✅ Specificity: {metrics['specificity']:.4f} (Recall CTR)")
    print("-" * 30)
    
    # Salva CSV delle predizioni per analisi errori
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True) # Assicura che la cartella esista
    
    out_csv = out_dir / f"predictions_{dataset_name}.csv"
    
    res_df = pd.DataFrame({
        "ID": details['ids'],
        "Label": details['labels'],
        "Prob_AD": np.array(details['probs']), # Probabilità classe 1
        "Pred": details['preds']
    })
    res_df.to_csv(out_csv, index=False)
    print(f"💾 Predizioni salvate in: {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml") 
    parser.add_argument("--model_path", default="outputs/balanced_mix/best_model.pt", help="Path del modello addestrato")
    args = parser.parse_args()

    # 1. Setup
    config = load_config(args.config)
    
    # --- FIX CRUCIALE: FORZA CLASSIFICAZIONE ---
    # Questo blocco risolve l'errore "RuntimeError: ... not implemented for Float"
    # Sovrascrive qualsiasi cosa ci sia nel config.yaml
    print("🔧 Forzatura Configurazione: CLASSIFICATION MODE")
    
    config.task = "classification"       # Forza il Dataloader a restituire Long (0/1)
    config.modality = "multimodal_cross_attention" # Assicura modello SOTA
    
    # Forza il Modello ad avere 2 neuroni finali (compatibile con i pesi salvati)
    if not hasattr(config, 'model'): config.model = type('obj', (object,), {})()
    config.model.output_dim = 2          
    
    # Imposta output dir predefinita se non c'è, per salvare i CSV
    if not hasattr(config, 'output_dir'):
        config.output_dir = "outputs/balanced_mix_evaluation"
    # -------------------------------------------

    set_seed(config.seed)
    device = torch.device(config.device)
    
    print(f"📂 Caricamento pesi da: {args.model_path}")
    if not Path(args.model_path).exists():
        print(f"❌ Errore: Modello non trovato in {args.model_path}")
        print("   Hai eseguito 'run_balanced_mix.py' prima?")
        return

    # 2. Ricostruzione Modello
    model = build_model(config).to(device)
    # Caricamento pesi sicuro
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    except:
        # Fallback per vecchie versioni di pytorch o pickle
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)

    # 3. Recupero Dati (Split coerente col training)
    it_test_df, en_test_df = load_data_splits(config)
    
    # 4. Inferenza
    # Test su ITALIANO (Sicily - Subset non visto)
    run_evaluation(model, it_test_df, config, tokenizer, "ITALIAN_TEST_SET")
    
    # Test su INGLESE (ADReSSo - Test Ufficiale)
    run_evaluation(model, en_test_df, config, tokenizer, "ENGLISH_TEST_SET")

if __name__ == "__main__":
    main()