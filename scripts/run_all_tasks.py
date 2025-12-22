import argparse
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy

# Aggiunge la root al path per importare i moduli src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils import set_seed
from src.data import get_data_splits, get_dataloaders
from src.engine import Trainer

def load_raw_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def find_available_tasks(dataset_root):
    """
    Scansiona il dataset per trovare quali Task_XX sono disponibili.
    Cerca dentro la cartella audio del primo soggetto che trova.
    """
    root = Path(dataset_root)
    # Prendi un soggetto a caso (es. il primo folder dentro la prima diagnosi)
    try:
        first_diag = next(root.iterdir())
        if not first_diag.is_dir(): return []
        first_subj = next(first_diag.iterdir())
        
        audio_dir = first_subj / "Audio"
        if not audio_dir.exists():
            return []
    except StopIteration:
        return []
    
    # Trova tutti i file wav e estrai la parte "Task_XX"
    tasks = set()
    for f in audio_dir.glob("*.wav"):
        # Esempio nome: SUBJ_0001_Task_01_Mic_Shure.wav
        parts = f.name.split('_')
        if "Task" in parts:
            idx = parts.index("Task")
            if idx + 1 < len(parts):
                task_name = f"Task_{parts[idx+1]}"
                tasks.add(task_name)
    
    return sorted(list(tasks))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # 1. Carica la configurazione base
    base_config_dict = load_raw_config(args.config)
    dataset_root = base_config_dict['data']['dataset_root']
    
    # 2. Trova i task disponibili
    available_tasks = find_available_tasks(dataset_root)
    print(f"🔍 Task trovati nel dataset: {available_tasks}")
    
    if not available_tasks:
        print("❌ Nessun task trovato (controlla i nomi dei file audio). Uso il default dal config.")
        available_tasks = [base_config_dict['data']['audio_file_pattern']]

    # Dizionario per salvare i risultati finali
    final_results = []
    
    # Cartella dove salvare i CSV delle predizioni per l'ensemble
    results_csv_dir = Path("results_csv")
    results_csv_dir.mkdir(exist_ok=True)

    # 3. Loop su ogni Task
    for task_name in available_tasks:
        print(f"\n{'='*40}")
        print(f"🚀 AVVIO TRAINING PER: {task_name}")
        print(f"{'='*40}")
        
        # Clona e modifica la configurazione per questo task
        current_config_dict = deepcopy(base_config_dict)
        
        # A. Imposta il pattern audio corretto
        current_config_dict['data']['audio_file_pattern'] = task_name
        
        # --- FIX NOMI CARTELLE OUTPUT ---
        # Creiamo un nome cartella basato sulla modalità e sul modello usato
        mode = current_config_dict.get('modality', 'unknown')
        
        if mode == 'text':
            # Prende l'ultima parte del nome modello (es. bert-base-italian...)
            model_name = current_config_dict['model']['text']['name'].split('/')[-1]
            exp_name = f"text_{model_name}"
            
        elif mode == 'audio':
            # Controlla se usa pretrained o nome custom
            if 'pretrained' in current_config_dict['model']['audio']:
                 model_name = current_config_dict['model']['audio']['pretrained'].split('/')[-1]
            else:
                 model_name = current_config_dict['model']['audio']['name']
            exp_name = f"audio_{model_name}"
            
        elif mode == 'multimodal':
            text_name = current_config_dict['model']['text']['name'].split('/')[-1]
            audio_name = current_config_dict['model']['audio']['pretrained'].split('/')[-1]
            exp_name = f"multimodal_{text_name}_{audio_name}"
        else:
            exp_name = f"{mode}_experiment"

        # Nuova struttura: outputs/nome_esperimento/Task_XX
        current_config_dict['output_dir'] = f"outputs/{exp_name}/{task_name}"
        # --------------------------------
        
        # B. FIX PATH TRASCRIZIONI
        # Se stiamo usando testo/multimodale, puntiamo alla sottocartella del task
        if 'transcripts_root' in current_config_dict['data']:
            base_transcripts = Path(current_config_dict['data']['transcripts_root'])
            task_transcript_dir = base_transcripts / task_name
            current_config_dict['data']['transcripts_root'] = str(task_transcript_dir)
            print(f"📂 Transcripts Dir: {task_transcript_dir}")
        
        print(f"📂 Output Dir:      {current_config_dict['output_dir']}")
        
        # Crea l'oggetto Config
        config = Config(current_config_dict)
        set_seed(config.seed)
        
        # Variabili per accumulare le metriche dei 5 fold
        fold_scores = []
        
        # Variabili per salvare le predizioni (per Ensemble/Stacking)
        all_task_ids = []
        all_task_probs = []
        all_task_labels = []
        
        # Loop sui Fold (Cross Validation)
        for fold, train_df, val_df in get_data_splits(config):
            print(f"  > Training Fold {fold}...")
            
            # Prepara i loader specifici per questo task
            train_loader, val_loader = get_dataloaders(config, train_df, val_df)
            
            # Inizializza e allena
            trainer = Trainer(config, train_loader, val_loader, fold)
            
            # --- FIX GESTIONE RETURN VALUE ---
            result = trainer.train()
            
            if isinstance(result, tuple):
                best_metric, fold_details = result
            else:
                best_metric = result
                fold_details = None

            if best_metric is None: best_metric = 0.0
            
            fold_scores.append(best_metric)
            print(f"  -> Fold {fold} Best Score: {best_metric:.4f}")
            
            # Accumula predizioni se disponibili
            if fold_details is not None:
                all_task_ids.extend(fold_details['ids'])
                all_task_probs.extend(fold_details['probs'])
                all_task_labels.extend(fold_details['labels'])

        # Calcola la media per questo Task
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"\n✅ Risultato {task_name}: {avg_score:.4f} ± {std_score:.4f}")
        
        final_results.append({
            "Task": task_name,
            "Avg_Score": avg_score,
            "Std_Dev": std_score
        })
        
        # Salva le predizioni per l'ensemble
        if all_task_ids:
            df_preds = pd.DataFrame({
                "ID": all_task_ids,
                "Label": all_task_labels,
                "Prob": all_task_probs
            })
            # Il nome del file include anche il modello per non sovrascrivere
            pred_filename = f"preds_{exp_name}_{task_name}.csv"
            pred_file = results_csv_dir / pred_filename
            df_preds.to_csv(pred_file, index=False)
            print(f"💾 Predizioni salvate in: {pred_file}")

    # 4. Stampa la classifica finale
    print("\n\n🏆 CLASSIFICA MIGLIORI TASK 🏆")
    print(f"{'Task':<15} | {'Score Medio':<12} | {'Std Dev'}")
    print("-" * 40)
    
    # Ordina dal migliore al peggiore
    final_results.sort(key=lambda x: x['Avg_Score'], reverse=True)
    
    for res in final_results:
        print(f"{res['Task']:<15} | {res['Avg_Score']:.4f}       | {res['Std_Dev']:.4f}")
    
    # Salva su CSV
    out_csv_name = f"results_{exp_name}.csv"
    pd.DataFrame(final_results).to_csv(out_csv_name, index=False)
    print(f"\nRisultati salvati in '{out_csv_name}'")

if __name__ == "__main__":
    main()