import argparse
import sys
import yaml
import pandas as pd
import numpy as np
import re
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils import set_seed
from src.tabular_engine import TabularTrainer

def load_raw_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def find_feature_tasks(features_root, feature_set):
    """
    Trova i file train_FEATURESET_Task_XX.csv
    """
    root = Path(features_root)
    # Pattern: train_egemaps_Task_01.csv
    files = list(root.glob(f"train_{feature_set}_Task_*.csv"))
    
    tasks = []
    for f in files:
        # Estrai "Task_XX" dal nome del file
        match = re.search(r"(Task_\d+)", f.name)
        if match:
            tasks.append(match.group(1))
            
    return sorted(list(set(tasks)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    base_config_dict = load_raw_config(args.config)
    features_root = base_config_dict['data']['features_root']
    feature_set = base_config_dict['feature_extraction']['feature_set']
    model_name = base_config_dict['tabular_model']['name']
    
    # 1. Trova i task dalle features estratte
    available_tasks = find_feature_tasks(features_root, feature_set)
    print(f"🔍 Feature Set: {feature_set}")
    print(f"🔍 Task trovati: {available_tasks}")
    
    if not available_tasks:
        print("❌ Nessun file di feature trovato. Hai eseguito extract_features.py?")
        return

    # Cartella per i CSV di predizione (per ensemble)
    # Es: outputs/tabular_xgboost_egemaps/results_csv
    base_out_dir = Path(f"outputs/tabular_{model_name}_{feature_set}")
    results_csv_dir = base_out_dir / "results_csv"
    results_csv_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = []

    # 2. Loop Training
    for task_name in available_tasks:
        print(f"\n{'='*40}")
        print(f"🚀 TABULAR TRAINING: {task_name}")
        print(f"{'='*40}")
        
        current_config = deepcopy(base_config_dict)
        
        # Imposta il task target per permettere a tabular_engine di caricare il file giusto
        current_config['data']['target_task'] = task_name
        
        # Output specifico per i modelli .pkl di questo task
        current_config['output_dir'] = str(base_out_dir / task_name)
        
        config = Config(current_config)
        set_seed(config.seed)
        
        trainer = TabularTrainer(config)
        
        # Nota: abbiamo modificato TabularTrainer per salvare il CSV nella cartella parent (results_csv_dir)
        # Passiamo temporaneamente una path fittizia e gestiamo il salvataggio CSV qui sotto o nell'engine
        # Per pulizia, sovrascriviamo l'output_dir dell'engine per puntare a results_csv_dir per il salvataggio CSV
        trainer.output_dir = Path(current_config['output_dir']) # Ripristina path modelli
        
        # Hack: Passiamo la cartella csv all'engine tramite attributo dinamico o lasciamo che salvi nel suo folder
        # Nel codice engine sopra, salva in self.output_dir.parent. 
        # Facciamo in modo che engine salvi i modelli in output_dir e il csv lo spostiamo noi o lo lasciamo lì.
        
        # Modifica al volo per far salvare il CSV in results_csv_dir dentro l'engine
        # (Richiede che l'engine usi trainer.output_dir come base per il CSV)
        # La modifica all'engine sopra fa: out_path = self.output_dir.parent / f"preds_{task_name}.csv"
        # Quindi se output_dir è "outputs/.../Task_01", parent è "outputs/..." -> perfetto.
        
        acc, f1 = trainer.train()
        
        print(f"✅ {task_name} -> Acc: {acc:.4f} | F1: {f1:.4f}")
        
        final_results.append({
            "Task": task_name,
            "Accuracy": acc,
            "F1": f1
        })
        
        # Sposta il file CSV generato dall'engine nella cartella ordinata 'results_csv'
        # L'engine lo ha salvato in base_out_dir / preds_Task_XX.csv
        src_csv = base_out_dir / f"preds_{task_name}.csv"
        dst_csv = results_csv_dir / f"preds_{task_name}.csv"
        if src_csv.exists():
            src_csv.rename(dst_csv)

    # 3. Classifica
    print(f"\n\n🏆 CLASSIFICA TABULAR ({model_name} + {feature_set}) 🏆")
    final_results.sort(key=lambda x: x['F1'], reverse=True)
    
    for res in final_results:
        print(f"{res['Task']:<15} | Acc: {res['Accuracy']:.4f} | F1: {res['F1']:.4f}")
        
    pd.DataFrame(final_results).to_csv(base_out_dir / "summary_results.csv", index=False)
    print(f"\nRisultati salvati in: {base_out_dir}")

if __name__ == "__main__":
    main()