import argparse
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy

# Aggiungi root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils import set_seed
from src.data import get_data_splits, get_dataloaders
from src.engine import Trainer

def load_raw_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_asr_folder_name(config_dict):
    try:
        engine = config_dict['transcription']['engine'].lower()
        if engine == "whisperx":
            return f"WhisperX_{config_dict['transcription']['whisperx']['model_name']}"
        elif engine == "nemo":
            return config_dict['transcription']['nemo']['model_name'].split('/')[-1]
        elif engine == "crisperwhisper":
            return config_dict['transcription']['crisperwhisper']['model_id'].split('/')[-1]
    except KeyError:
        return "UnknownASR"
    return "UnknownASR"

def find_available_tasks(dataset_root):
    root = Path(dataset_root)
    tasks = set()
    try:
        for wav_file in root.rglob("Audio/*.wav"):
            parts = wav_file.name.split('_')
            if "Task" in parts:
                idx = parts.index("Task")
                if idx + 1 < len(parts):
                    tasks.add(f"Task_{parts[idx+1]}")
    except Exception: pass
    return sorted(list(tasks)) if tasks else ["Task_01"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    base_config_dict = load_raw_config(args.config)
    dataset_root = base_config_dict['data']['dataset_root']
    
    available_tasks = find_available_tasks(dataset_root)
    print(f"🔍 Task trovati: {available_tasks}")
    
    results_csv_dir = Path("results_csv")
    results_csv_dir.mkdir(exist_ok=True)
    
    final_summary = []
    
    # Nome esperimento
    modality = base_config_dict.get('modality', 'unknown')
    if modality == 'text' or 'multimodal' in modality:
        if "xphonebert" in base_config_dict['model']['text']['name']:
             exp_name = f"phonetic_xphonebert"
        else:
             exp_name = f"text_{base_config_dict['model']['text']['name'].split('/')[-1]}"
    else:
        exp_name = f"audio_{base_config_dict['model']['audio']['pretrained'].split('/')[-1]}"
    
    if 'multimodal' in modality:
        exp_name = f"multimodal_{exp_name}"

    asr_base_name = get_asr_folder_name(base_config_dict)

    for task_name in available_tasks:
        print(f"\n{'='*60}")
        print(f"🚀 AVVIO TRAINING: {task_name} | Exp: {exp_name}")
        print(f"{'='*60}")
        
        current_config_dict = deepcopy(base_config_dict)
        current_config_dict['data']['audio_file_pattern'] = task_name
        
        # --- PATH TRASCRIZIONI ---
        if 'text' in modality or 'multimodal' in modality:
            base_transcripts_root = Path(base_config_dict['data']['transcripts_root'])
            # Se è path completo, resetta alla base
            if "Task_" in str(base_transcripts_root):
                 base_transcripts_root = base_transcripts_root.parent.parent

            # Folder name (es. WhisperX_large-v3 o WhisperX_large-v3_phonemes)
            if "xphonebert" in base_config_dict['model']['text']['name']:
                folder_name = f"{asr_base_name}_phonemes"
            else:
                folder_name = asr_base_name
            
            target_transcript_dir = base_transcripts_root / folder_name / task_name
            
            if not target_transcript_dir.exists():
                # Fallback: prova a cercarlo dentro transcripts_root se l'utente ha messo il path parziale
                fallback = Path("data/transcripts") / folder_name / task_name
                if fallback.exists():
                    target_transcript_dir = fallback
                else:
                    print(f"❌ Trascrizioni non trovate: {target_transcript_dir}")
                    continue
            
            current_config_dict['data']['transcripts_root'] = str(target_transcript_dir)
        
        current_config_dict['output_dir'] = f"outputs/{exp_name}/{task_name}"
        
        config = Config(current_config_dict)
        set_seed(config.seed)
        
        fold_metrics_list = []
        all_ids, all_probs, all_labels = [], [], []

        try:
            for fold, train_df, val_df in get_data_splits(config):
                print(f"\n  [Fold {fold}] Training...")
                train_loader, val_loader = get_dataloaders(config, train_df, val_df)
                
                if len(train_loader) == 0: continue

                trainer = Trainer(config, train_loader, val_loader, fold)
                best_metrics, fold_details = trainer.train()

                if best_metrics:
                    fold_metrics_list.append(best_metrics)
                
                if fold_details:
                    all_ids.extend(fold_details['ids'])
                    all_probs.extend(fold_details['probs'])
                    all_labels.extend(fold_details['labels'])
        
        except Exception as e:
            print(f"❌ Errore critico nel task {task_name}: {e}")
            import traceback; traceback.print_exc()
            continue

        # --- CALCOLO STATISTICHE FINALI (Mean ± Std) ---
        if fold_metrics_list:
            df_metrics = pd.DataFrame(fold_metrics_list)
            means = df_metrics.mean()
            stds = df_metrics.std()
            
            print(f"\n📊 RISULTATI AGGREGATI {task_name} (5 Folds):")
            print("-" * 50)
            for metric in means.index:
                print(f"{metric.capitalize():<15}: {means[metric]:.4f} ± {stds[metric]:.4f}")
            print("-" * 50)
            
            res_entry = {"Task": task_name}
            for metric in means.index:
                res_entry[f"{metric}_mean"] = means[metric]
                res_entry[f"{metric}_std"] = stds[metric]
            final_summary.append(res_entry)
        
        # Salvataggio CSV Predizioni
        if all_ids:
            df_preds = pd.DataFrame({"ID": all_ids, "Label": all_labels, "Prob": all_probs})
            out_file = results_csv_dir / f"preds_{exp_name}_{task_name}.csv"
            df_preds.to_csv(out_file, index=False)
            print(f"💾 Predizioni salvate in: {out_file}")

    # --- SALVATAGGIO SUMMARY GENERALE ---
    if final_summary:
        df_summary = pd.DataFrame(final_summary)
        summary_path = results_csv_dir / f"summary_results_{exp_name}.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\n🏆 Summary completo salvato in: {summary_path}")

if __name__ == "__main__":
    main()