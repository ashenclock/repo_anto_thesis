import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config
from src.data import get_data_splits, get_dataloaders
from src.engine import Trainer
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    # Loop over the 5 pre-computed folds
    for fold, train_df, val_df in get_data_splits(config):
        print(f"\n--- Training FOLD {fold} ---")
        train_loader, val_loader = get_dataloaders(config, train_df, val_df)

        # Initialize and run the trainer
        trainer = Trainer(config, train_loader, val_loader, fold)
        trainer.train()


if __name__ == "__main__":
    main()