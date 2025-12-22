import argparse
import sys
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import opensmile
import warnings
from multiprocessing import Pool, cpu_count
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

warnings.filterwarnings("ignore")

extractor = None

def initialize_worker(feature_set_name):
    """Inizializza l'estrattore OpenSMILE su ogni processo figlio."""
    global extractor
    feature_set = opensmile.FeatureSet.eGeMAPSv02 if feature_set_name == 'egemaps' else opensmile.FeatureSet.ComParE_2016
    extractor = opensmile.Smile(
        feature_set=feature_set,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

def process_single_file(audio_path):
    """Funzione eseguita in parallelo."""
    global extractor
    if not audio_path or pd.isna(audio_path) or not Path(audio_path).exists():
        return None
    try:
        df_feat = extractor.process_file(audio_path)
        df_feat.reset_index(inplace=True)
        # Estrai l'ID soggetto (es. SUBJ_0109) dal nome del file
        subj_id_match = re.search(r'(SUBJ_\d+)', Path(audio_path).name)
        if subj_id_match:
            df_feat['ID'] = subj_id_match.group(1)
        else:
            return None # Salta se non trova l'ID
        return df_feat
    except Exception as e:
        print(f"Errore su {audio_path}: {e}")
        return None

def main():
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Estrae feature acustiche per TUTTI i task.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    # 1. Carica il file CSV principale
    metadata_path = Path(config.data.metadata_file)
    if not metadata_path.exists():
        sys.exit(f"ERRORE: File metadati non trovato: {metadata_path}")
    
    df_meta = pd.read_csv(metadata_path)
    df_meta = df_meta[df_meta['has_audio'] == True].copy()

    features_root = Path(config.data.features_root)
    features_root.mkdir(parents=True, exist_ok=True)
    
    feature_set_name = config.feature_extraction.feature_set
    overwrite = config.feature_extraction.overwrite
    n_cpu = cpu_count()
    
    # --- FIX CRUCIALE: ESTRAI I TASK DA 'audio_path' ---
    # Invece di 'Subject_ID', usiamo 'audio_path' che contiene "Task_XX"
    df_meta['task'] = df_meta['audio_path'].str.extract(r'(Task_\d+)', expand=False)
    df_meta.dropna(subset=['task'], inplace=True)
    tasks = sorted(df_meta['task'].unique())
    # ----------------------------------------------------
    
    print(f"--- Estrazione {feature_set_name} per i task: {tasks} ---")

    if not tasks:
        sys.exit("Nessun task trovato. Controlla la colonna 'audio_path' in speech_metadata.csv")

    for task in tasks:
        print(f"\nProcessing {task}...")
        
        task_df = df_meta[df_meta['task'] == task]
        audio_paths = task_df['audio_path'].tolist()
        
        out_name = f"train_{feature_set_name}_{task}.csv"
        out_path = features_root / out_name

        if out_path.exists() and not overwrite:
            print(f"File {out_name} esistente, salto.")
            continue
            
        results = []
        with Pool(processes=n_cpu, initializer=initialize_worker, initargs=(feature_set_name,)) as pool:
            for res in tqdm(pool.imap_unordered(process_single_file, audio_paths), total=len(audio_paths), desc=f"Extracting {task}"):
                if res is not None:
                    results.append(res)
        
        if not results:
            print(f"Nessuna feature estratta per {task}, salto.")
            continue

        full_df = pd.concat(results, ignore_index=True)
        cols_to_drop = ['file', 'start', 'end']
        full_df = full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns], errors='ignore')
        cols = ['ID'] + [c for c in full_df.columns if c != 'ID']
        full_df = full_df[cols]
        full_df.to_csv(out_path, index=False)
        print(f"✅ Salvato: {out_path} ({len(full_df)} campioni)")

if __name__ == "__main__":
    main()