import argparse
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils import set_seed
from src.data import get_data_splits_combined, get_dataloaders
from src.engine import Trainer

def load_raw_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train on COMBINED tasks (Data Fusion)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--tasks", type=str, nargs='+', default=["ALL"], help="Lista task da combinare")
    parser.add_argument("--tag", type=str, default="", help="Suffisso nome esperimento")
    
    args = parser.parse_args()

    base_config = load_raw_config(args.config)
    
    tasks_str = "ALL" if "ALL" in args.tasks else "-".join(args.tasks)
    exp_name = f"COMBINED_{tasks_str}"
    if args.tag: exp_name = f"{exp_name}_{args.tag}"
    
    base_config['output_dir'] = f"outputs/combined_experiments/{exp_name}"
    
    config = Config(base_config)
    set_seed(config.seed)
    
    print(f"\n🚀 AVVIO TRAINING COMBINATO: {tasks_str}")
    print(f"🏷️  Nome Esperimento: {exp_name}")
    
    results_csv_dir = Path("results_csv_combined")
    results_csv_dir.mkdir(exist_ok=True)
    
    fold_metrics_list = []
    all_ids, all_probs, all_labels, all_folds = [], [], [], [] # <--- Aggiunto all_folds

    try:
        for fold, train_df, val_df in get_data_splits_combined(config, args.tasks):
            print(f"\n  [Fold {int(fold)}] Train samples: {len(train_df)} | Val samples: {len(val_df)}")
            
            train_loader, val_loader = get_dataloaders(config, train_df, val_df)
            
            trainer = Trainer(config, train_loader, val_loader, fold)
            best_metrics, fold_details = trainer.train()

            if best_metrics:
                fold_metrics_list.append(best_metrics)
                target = config.training.get("eval_metric", "f1")
                print(f"  -> Fold {int(fold)} Best {target.upper()}: {best_metrics.get(target, 0):.4f}")
            
            if fold_details:
                all_ids.extend(fold_details['ids'])
                all_probs.extend(fold_details['probs'])
                all_labels.extend(fold_details['labels'])
                
                # --- FIX FONDAMENTALE: SALVIAMO IL FOLD ---
                # Aggiungiamo il numero del fold ripetuto per ogni sample di questo batch
                all_folds.extend([int(fold)] * len(fold_details['ids']))

    except Exception as e:
        print(f"❌ Errore critico: {e}")
        import traceback; traceback.print_exc()
        return

    if fold_metrics_list:
        df_metrics = pd.DataFrame(fold_metrics_list)
        means = df_metrics.mean()
        stds = df_metrics.std()
        
        print(f"\n🏆 RISULTATI AGGREGATI ({exp_name}):")
        print("-" * 50)
        for metric in means.index:
            print(f"{metric.capitalize():<15}: {means[metric]:.4f} ± {stds[metric]:.4f}")
        print("-" * 50)
        
        # Salva Predizioni con colonna kfold
        df_preds = pd.DataFrame({
            "ID": all_ids, 
            "Label": all_labels, 
            "Prob": all_probs,
            "kfold": all_folds # <--- Ora c'è!
        })
        out_file = results_csv_dir / f"preds_{exp_name}.csv"
        df_preds.to_csv(out_file, index=False)
        print(f"💾 Predizioni salvate in: {out_file}")

if __name__ == "__main__":
    main()