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

def get_asr_folder_name(config_dict):
    """Helper per ottenere il nome della cartella del modello ASR dal config."""
    try:
        engine = config_dict['transcription']['engine'].lower()
        if engine == "whisperx":
            return f"WhisperX_{config_dict['transcription']['whisperx']['model_name']}"
        elif engine == "nemo":
            return config_dict['transcription']['nemo']['model_name'].split('/')[-1]
        elif engine == "crisperwhisper":
            return config_dict['transcription']['crisperwhisper']['model_id'].split('/')[-1]
    except KeyError:
        # Se la sezione transcription non c'è, ritorna un default
        return "UnknownASR"
    return "UnknownASR"


def find_available_tasks(dataset_root):
    """Scansiona il dataset per trovare quali Task_XX sono disponibili."""
    root = Path(dataset_root)
    tasks = set()
    try:
        # Cerca i task in tutte le cartelle audio
        for wav_file in root.rglob("Audio/*.wav"):
            parts = wav_file.name.split('_')
            if "Task" in parts:
                idx = parts.index("Task")
                if idx + 1 < len(parts):
                    tasks.add(f"Task_{parts[idx+1]}")
    except Exception:
        pass
    
    return sorted(list(tasks)) if tasks else ["Task_01"] # Fallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    base_config_dict = load_raw_config(args.config)
    dataset_root = base_config_dict['data']['dataset_root']
    
    available_tasks = find_available_tasks(dataset_root)
    print(f"🔍 Task trovati nel dataset: {available_tasks}")
    
    final_results = []
    results_csv_dir = Path("results_csv")
    results_csv_dir.mkdir(exist_ok=True)

    # Determina il nome del modello ASR usato per le trascrizioni
    asr_name = get_asr_folder_name(base_config_dict)
    
    for task_name in available_tasks:
        print(f"\n{'='*40}")
        print(f"🚀 AVVIO TRAINING PER: {task_name}")
        print(f"{'='*40}")
        
        current_config_dict = deepcopy(base_config_dict)
        current_config_dict['data']['audio_file_pattern'] = task_name
        
        mode = current_config_dict.get('modality', 'unknown')
        
        # --- FIX NOMI CARTELLE OUTPUT E PATH TRASCRIZIONI ---
        
        # Determina il nome dell'esperimento per la cartella di output
        if mode == 'text':
            model_name = current_config_dict['model']['text']['name'].split('/')[-1]
            # Se usiamo XPhoneBERT, specifichiamolo nel nome
            if 'xphonebert' in model_name:
                exp_name = f"phonetic_{model_name}_ASR-{asr_name}"
            else:
                exp_name = f"text_{model_name}_ASR-{asr_name}"
        # ... (aggiungi logica per 'audio' e 'multimodal' se ti serve)
        else:
            exp_name = f"{mode}_experiment"

        # Path di output: outputs/nome_esperimento/Task_XX
        current_config_dict['output_dir'] = f"outputs/{exp_name}/{task_name}"
        
        # --- FIX PATH TRASCRIZIONI ---
        if mode == 'text' or mode == 'multimodal':
            # Se usiamo XPhoneBERT, dobbiamo puntare alla cartella _phonemes
            if "xphonebert" in current_config_dict['model']['text']['name'].lower():
                # Path: data/transcripts/WhisperX_large-v3_phonemes/Task_01
                transcripts_dir = Path(current_config_dict['data']['transcripts_root']) / f"{asr_name}_phonemes" / task_name
            else:
                # Path normale: data/transcripts/WhisperX_large-v3/Task_01
                transcripts_dir = Path(current_config_dict['data']['transcripts_root']) / asr_name / task_name

            current_config_dict['data']['transcripts_root'] = str(transcripts_dir)
            print(f"📂 Transcripts Dir: {transcripts_dir}")
        
        print(f"📂 Output Dir:      {current_config_dict['output_dir']}")
        
        config = Config(current_config_dict)
        set_seed(config.seed)
        
        fold_scores = []
        all_task_ids, all_task_probs, all_task_labels = [], [], []
        
        for fold, train_df, val_df in get_data_splits(config):
            print(f"  > Training Fold {fold}...")
            train_loader, val_loader = get_dataloaders(config, train_df, val_df)
            trainer = Trainer(config, train_loader, val_loader, fold)
            
            result = trainer.train()
            
            if isinstance(result, tuple):
                best_metric, fold_details = result
            else:
                best_metric, fold_details = result, None

            if best_metric is None: best_metric = 0.0
            fold_scores.append(best_metric)
            print(f"  -> Fold {fold} Best Score: {best_metric:.4f}")
            
            if fold_details:
                all_task_ids.extend(fold_details['ids'])
                all_task_probs.extend(fold_details['probs'])
                all_task_labels.extend(fold_details['labels'])

        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f"\n✅ Risultato {task_name}: {avg_score:.4f} ± {std_score:.4f}")
        
        final_results.append({"Task": task_name, "Avg_Score": avg_score, "Std_Dev": std_score})
        
        if all_task_ids:
            df_preds = pd.DataFrame({"ID": all_task_ids, "Label": all_task_labels, "Prob": all_task_probs})
            pred_filename = f"preds_{exp_name}_{task_name}.csv"
            pred_file = results_csv_dir / pred_filename
            df_preds.to_csv(pred_file, index=False)
            print(f"💾 Predizioni salvate in: {pred_file}")

    print(f"\n\n🏆 CLASSIFICA FINALE - Esperimento: {exp_name} 🏆")
    final_results.sort(key=lambda x: x['Avg_Score'], reverse=True)
    
    for res in final_results:
        print(f"{res['Task']:<15} | {res['Avg_Score']:.4f}       | {res['Std_Dev']:.4f}")
    
    out_csv_name = f"results_{exp_name}.csv"
    pd.DataFrame(final_results).to_csv(out_csv_name, index=False)
    print(f"\nRiassunto salvato in: '{out_csv_name}'")

if __name__ == "__main__":
    main()