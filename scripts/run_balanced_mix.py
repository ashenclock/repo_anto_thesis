import argparse
import sys
import yaml
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch.utils.data import WeightedRandomSampler # <--- NUOVO

# Setup Path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from src.config import Config
from src.utils import set_seed
from src.data import get_dataloaders, MultimodalDataset, collate_multimodal
from src.engine import Trainer, evaluate
from src.models import build_model
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore") # Zittiamo i warning sui pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_balanced.yaml")
    parser.add_argument("--use_all_english", action='store_true')
    # Nuovo flag per attivare il sampler
    parser.add_argument("--weighted_sampler", action='store_true', help="Usa campionamento pesato per bilanciare le lingue")
    args = parser.parse_args()

    with open(args.config) as f: raw_config = yaml.safe_load(f)
    if 'output_dir' not in raw_config: raw_config['output_dir'] = "outputs/balanced_mix"
    config = Config(raw_config)
    set_seed(config.seed)

    print("\n🔮 PREPARAZIONE DATASET (Weighted Mix)")
    
    # 1. Carica Tutto
    df = pd.read_csv(config.data.metadata_file)
    df_sicily = df[df['Dataset'] == 'Sicily'].copy()
    df_adresso_train = df[df['Dataset'].isin(['ADReSSo_Train', 'ADReSSo'])].copy()
    df_adresso_test_official = df[df['Dataset'] == 'ADReSSo_Test'].copy()

    # 2. Split IT
    it_train, it_test = train_test_split(df_sicily, test_size=0.20, stratify=df_sicily['Diagnosis'], random_state=config.seed)
    
    # 3. EN
    en_train = df_adresso_train # Usiamo tutto se use_all_english è True (default logica precedente)
    
    # 4. Training Set
    train_final = pd.concat([it_train, en_train]).reset_index(drop=True)
    
    # Split Val
    train_split, val_split = train_test_split(train_final, test_size=0.1, stratify=train_final['Diagnosis'], random_state=config.seed)
    
    print(f"   Train: {len(train_split)} (IT: {len(train_split[train_split['Language']=='IT'])}, EN: {len(train_split[train_split['Language']=='EN'])})")

    # --- SAMPLER PER BILANCIARE ---
    sampler = None
    shuffle = True
    
    if args.weighted_sampler:
        print("⚖️  Attivazione Weighted Sampler per bilanciare IT/EN...")
        # Calcola pesi
        count_it = len(train_split[train_split['Language']=='IT'])
        count_en = len(train_split[train_split['Language']=='EN'])
        weight_it = 1.0 / count_it
        weight_en = 1.0 / count_en
        
        # Assegna peso a ogni sample
        sample_weights = train_split['Language'].map({'IT': weight_it, 'EN': weight_en}).values
        sample_weights = torch.DoubleTensor(sample_weights)
        
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False # Con sampler, shuffle deve essere False

    # Dataloaders
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    train_ds = MultimodalDataset(train_split, config, tokenizer)
    val_ds = MultimodalDataset(val_split, config, tokenizer)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=config.training.batch_size, 
        sampler=sampler, # <--- QUI
        shuffle=shuffle, 
        collate_fn=collate_multimodal
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_multimodal)

    # Train
    trainer = Trainer(config, train_loader, val_loader, fold="Weighted_Mix")
    trainer.train()

    # Test
    print("\n🔬 VALUTAZIONE FINALE")
    device = torch.device(config.device)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(trainer.output_dir / "best_model.pt", weights_only=True)) # Fix warning
    model.eval()

    def run_test(name, dataframe):
        if len(dataframe) == 0: return
        ds = MultimodalDataset(dataframe, config, tokenizer)
        loader = torch.utils.data.DataLoader(ds, batch_size=config.training.batch_size, collate_fn=collate_multimodal)
        _, metrics, _ = evaluate(model, loader, device, torch.nn.CrossEntropyLoss())
        print(f"\n👉 {name}")
        print(f"   Accuracy:    {metrics['accuracy']:.4f}")
        print(f"   F1-Score:    {metrics['f1']:.4f}")
        print(f"   Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"   Specificity: {metrics['specificity']:.4f}")

    run_test("TEST SET ITALIANO", it_test)
    run_test("TEST SET INGLESE", df_adresso_test_official)

if __name__ == "__main__":
    main()