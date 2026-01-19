import argparse
import sys
import yaml
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup Path per importare i moduli src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils import set_seed, clear_memory
from src.data import MultimodalDataset, collate_multimodal
from src.engine import Trainer, evaluate
from src.models import build_model
from transformers import AutoTokenizer

def get_dataset_splits(config, mode):
    """
    Seleziona i dati dal CSV Multimodale con rigore sulle partizioni.
    """
    df = pd.read_csv(config.data.metadata_file)
    
    # 1. Definisci i DataFrame per le singole partizioni
    df_it_sicily = df[df['Dataset'] == 'Sicily'].copy()
    
    # Separiamo nettamente Train e Test per l'inglese
    df_en_train = df[df['Dataset'].isin(['ADReSSo_Train', 'ADReSSo'])].copy()
    df_en_test = df[df['Dataset'] == 'ADReSSo_Test'].copy() # <--- SOLO IL TEST UFFICIALE
    
    print(f"   📊 Dati Disponibili:")
    print(f"      - IT (Sicily):       {len(df_it_sicily)}")
    print(f"      - EN (ADReSSo Train): {len(df_en_train)}")
    print(f"      - EN (ADReSSo Test):  {len(df_en_test)}")

    # 3. Logica Scambio
    if mode == "IT_to_EN":
        print(f"\n🌍 MODALITÀ: Train su ITALIANO -> Test su INGLESE (Solo Test Set Ufficiale)")
        # Train: Tutto il dataset Sicily
        train_full = df_it_sicily
        # Test: SOLO la parte di test di ADReSSo
        test_df = df_en_test 
        folder_name = "cross_dataset_IT_to_EN"
        
    elif mode == "EN_to_IT":
        print(f"\n🌍 MODALITÀ: Train su INGLESE (Solo Train Set) -> Test su ITALIANO")
        # Train: Usiamo ADReSSo_Train (NON tocchiamo il test set per il training)
        train_full = df_en_train
        # Test: Tutto Sicily
        test_df = df_it_sicily
        folder_name = "cross_dataset_EN_to_IT"
    else:
        raise ValueError("Mode deve essere IT_to_EN o EN_to_IT")

    # 4. Suddividi il Train in Train/Val (15% per validation interna)
    train_df, val_df = train_test_split(
        train_full, 
        test_size=0.15, 
        stratify=train_full['Diagnosis'], 
        random_state=config.seed
    )
    
    return train_df, val_df, test_df, folder_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    # Due modalità: IT_to_EN (allena IT, testa EN) o EN_to_IT (viceversa)
    parser.add_argument("--mode", type=str, choices=["IT_to_EN", "EN_to_IT"], required=True)
    args = parser.parse_args()

    # Carica configurazione base
    with open(args.config) as f: raw_config = yaml.safe_load(f)
    
    # Prepariamo i dati
    # (Lo facciamo prima di creare Config per impostare l'output_dir corretto)
    temp_conf = Config(raw_config) # Config temporaneo per leggere i path
    train_df, val_df, test_df, exp_name = get_dataset_splits(temp_conf, args.mode)
    
    # --- CONFIGURAZIONE OUTPUT ---
    # Questo assicura che NON sovrascrivi nulla.
    # Cartella: outputs/cross_dataset_IT_to_EN (o EN_to_IT)
    raw_config['output_dir'] = f"outputs/{exp_name}"
    
    config = Config(raw_config)
    set_seed(config.seed)
    clear_memory()
    
    print(f"   📁 Output Dir: {config.output_dir}")

    # --- DATALOADERS ---
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    
    train_ds = MultimodalDataset(train_df, config, tokenizer)
    val_ds = MultimodalDataset(val_df, config, tokenizer)
    test_ds = MultimodalDataset(test_df, config, tokenizer)
    
    # --- FIX CRUCIALE: drop_last=True ---
    # Questo impedisce il crash quando l'ultimo batch ha dimensione 1
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        collate_fn=collate_multimodal, 
        num_workers=4,
        drop_last=True  # <--- AGGIUNTO QUESTO
    )
    
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_multimodal, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_multimodal, num_workers=4)

    # --- TRAINING (Dataset A) ---
    print(f"\n🚀 AVVIO TRAINING su {args.mode.split('_')[0]}...")
    # Usiamo "fold" nel nome solo per compatibilità con il Trainer, qui è simbolico
    trainer = Trainer(config, train_loader, val_loader, fold="CrossDataset")
    trainer.train()

    # --- TESTING (Dataset B) ---
    print(f"\n🔬 AVVIO TEST su {args.mode.split('_')[-1]} (Target Domain)...")
    
    device = torch.device(config.device)
    model = build_model(config).to(device)
    
    # Carica il miglior modello appena addestrato
    best_model_path = Path(config.output_dir) / "best_model.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    # Valutazione
    loss, metrics, details = evaluate(model, test_loader, device, torch.nn.CrossEntropyLoss())
    
    print("\n" + "="*60)
    print(f"🏆 RISULTATI CROSS-DATASET ({args.mode})")
    print("="*60)
    print(f"   Dataset Train: {len(train_df)+len(val_df)} samples")
    print(f"   Dataset Test:  {len(test_df)} samples")
    print("-" * 30)
    print(f"   Accuracy:    {metrics['accuracy']:.4f}")
    print(f"   F1-Score:    {metrics['f1']:.4f}")
    print(f"   Sensitivity: {metrics['sensitivity']:.4f} (Capacità di trovare AD)")
    print(f"   Specificity: {metrics['specificity']:.4f} (Capacità di trovare Sani)")
    print("="*60)

    # Salva il report
    results_csv = Path(config.output_dir) / "cross_dataset_results.csv"
    res_df = pd.DataFrame({
        "ID": details['ids'],
        "Label": details['labels'],
        "Prob": details['probs'],
        "Pred": details['preds'],
        "Experiment": args.mode
    })
    res_df.to_csv(results_csv, index=False)
    print(f"💾 Dettagli predizioni salvati in: {results_csv}")

if __name__ == "__main__":
    main()