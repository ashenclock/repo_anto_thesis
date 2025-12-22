import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import set_seed
from src.data import get_data_splits, get_dataloaders
from src.engine import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    # Ciclo sui 5 Fold pre-calcolati
    for fold, train_df, val_df in get_data_splits(config):
        print(f"\n--- Training FOLD {fold} ---")
        train_loader, val_loader = get_dataloaders(config, train_df, val_df)
        
        # Il tuo Trainer originale
        trainer = Trainer(config, train_loader, val_loader, fold)
        trainer.train()

if __name__ == "__main__":
    main()