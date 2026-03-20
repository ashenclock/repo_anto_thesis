import os
import shutil
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

# --- CONFIGURAZIONE DESTINAZIONE ---
DEST_DATASET_ROOT = Path("data/dataset")
DEST_TRANSCRIPTS_ROOT = Path("data/transcripts")
OUTPUT_CSV = Path("data/metadata/multilingual_meta_full.csv")
SICILY_META = Path("data/metadata/speech_metadata.csv")

def copy_transcripts(src_root, filename_stem, new_id, dest_root):
    """
    Copia tutte le varianti di trascrizione (large-v3, parakeet...)
    """
    for model_dir in src_root.iterdir():
        if not model_dir.is_dir(): continue
        
        # Cerca il file txt corrispondente
        src_txt = model_dir / f"{filename_stem}.txt"
        
        if src_txt.exists():
            # Destinazione: data/transcripts/{Modello}/Task_01/{NewID}.txt
            dest_folder = dest_root / model_dir.name / "Task_01"
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            dest_file = dest_folder / f"{new_id}.txt"
            shutil.copy2(src_txt, dest_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_repo", required=True, help="Path root del vecchio repository")
    args = parser.parse_args()

    old_repo = Path(args.old_repo)
    
    # --- PATHS SORGENTI (Ricavati dal tuo tree) ---
    src_train_audio = old_repo / "Adresso21/ADReSSo21-diagnosis-train/ADReSSo21/diagnosis/train/audio"
    src_test_audio  = old_repo / "Adresso21/ADReSSo21-diagnosis-test/ADReSSo21/diagnosis/test-dist/audio"
    
    src_train_transcripts = old_repo / "transcripts"
    src_test_transcripts  = old_repo / "transcripts_test"
    
    # Label File per il Test Set
    src_test_labels = old_repo / "Adresso21/label_test_task1.csv"

    if not src_test_labels.exists():
        print(f"❌ Errore: File label test non trovato: {src_test_labels}")
        return

    print("🚀 INIZIO IMPORTAZIONE TOTALE...")
    all_records = []

    # =================================================================
    # 1. SICILY (Fold 0)
    # =================================================================
    print("\n🔹 [1/3] Caricamento Sicily...")
    if SICILY_META.exists():
        df_sicily = pd.read_csv(SICILY_META)
        df_sicily = df_sicily[df_sicily['Subject_ID'].str.contains("Task_01")]
        df_sicily = df_sicily[df_sicily['Diagnosis'].isin(['CTR', 'MILD-AD'])]
        df_sicily['kfold'] = 0
        df_sicily['Language'] = 'IT'
        df_sicily['Dataset'] = 'Sicily'
        all_records.append(df_sicily)
        print(f"   -> Sicily: {len(df_sicily)}")
    else:
        print("⚠️  Metadata Sicily non trovato. Salto.")

    # =================================================================
    # 2. ADReSSo TRAIN (Fold 1)
    # =================================================================
    print("\n🔹 [2/3] Importazione ADReSSo TRAIN...")
    dest_train_audio = DEST_DATASET_ROOT / "ADReSSo_Train"
    dest_train_audio.mkdir(parents=True, exist_ok=True)
    
    train_records = []
    for label_dir in ["ad", "cn"]:
        src_dir = src_train_audio / label_dir
        if not src_dir.exists(): continue
        
        diagnosis = "AD" if label_dir == "ad" else "CTR"
        
        for wav in tqdm(list(src_dir.glob("*.wav")), desc=f"Copia {label_dir}"):
            clean_name = wav.stem
            new_id = f"EN_{clean_name}_Task_01"
            
            # Copia Audio
            dst_wav = dest_train_audio / f"{new_id}.wav"
            shutil.copy2(wav, dst_wav)
            
            # Copia Trascrizioni
            copy_transcripts(src_train_transcripts, clean_name, new_id.replace("_Task_01", ""), DEST_TRANSCRIPTS_ROOT)
            
            train_records.append({
                "Subject_ID": new_id,
                "Diagnosis": diagnosis,
                "audio_path": str(dst_wav.resolve()),
                "Language": "EN",
                "Dataset": "ADReSSo_Train",
                "kfold": 1,
                "has_audio": True
            })
    all_records.append(pd.DataFrame(train_records))

    # =================================================================
    # 3. ADReSSo TEST (Fold 2)
    # =================================================================
    print("\n🔹 [3/3] Importazione ADReSSo TEST...")
    
    # Carica Labels Test
    try:
        # Tenta separatori diversi
        df_lbl = pd.read_csv(src_test_labels)
        if len(df_lbl.columns) < 2: 
            df_lbl = pd.read_csv(src_test_labels, sep=';')
        
        # Mappa ID -> Label
        # Cerca colonne
        id_col = next((c for c in df_lbl.columns if 'id' in c.lower()), df_lbl.columns[0])
        lbl_col = next((c for c in df_lbl.columns if 'dx' in c.lower() or 'label' in c.lower()), df_lbl.columns[1])
        
        def norm_lbl(v):
            s = str(v).upper()
            return 'AD' if ('1' in s or 'AD' in s) else 'CTR'
            
        test_labels_map = dict(zip(df_lbl[id_col].astype(str), df_lbl[lbl_col].apply(norm_lbl)))
        print(f"   -> Caricate {len(test_labels_map)} etichette di test.")
        
    except Exception as e:
        print(f"❌ Errore lettura CSV Test Labels: {e}")
        return

    dest_test_audio = DEST_DATASET_ROOT / "ADReSSo_Test"
    dest_test_audio.mkdir(parents=True, exist_ok=True)
    
    test_records = []
    wav_files = list(src_test_audio.glob("*.wav"))
    
    for wav in tqdm(wav_files, desc="Copia Test"):
        clean_name = wav.stem
        # Gestione nomi file strani (es. adrsdt10.wav)
        if clean_name not in test_labels_map:
            # Prova a cercare nel CSV se c'è un ID parziale
            print(f"⚠️ Label mancante per {clean_name}, salto.")
            continue
            
        new_id = f"EN_TEST_{clean_name}_Task_01"
        
        # Copia Audio
        dst_wav = dest_test_audio / f"{new_id}.wav"
        shutil.copy2(wav, dst_wav)
        
        # Copia Trascrizioni (da transcripts_test)
        copy_transcripts(src_test_transcripts, clean_name, new_id.replace("_Task_01", ""), DEST_TRANSCRIPTS_ROOT)
        
        test_records.append({
            "Subject_ID": new_id,
            "Diagnosis": test_labels_map[clean_name],
            "audio_path": str(dst_wav.resolve()),
            "Language": "EN",
            "Dataset": "ADReSSo_Test",
            "kfold": 2, # Fold 2 = Test Ufficiale
            "has_audio": True
        })
        
    all_records.append(pd.DataFrame(test_records))

    # =================================================================
    # SALVATAGGIO
    # =================================================================
    df_final = pd.concat(all_records, ignore_index=True)
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*60)
    print(f"✅ DATASET UNIFICATO PRONTO!")
    print(f"   CSV: {OUTPUT_CSV}")
    print(f"   Totale File: {len(df_final)}")
    print(f"   - Sicily (IT):          {len(df_sicily)}")
    print(f"   - ADReSSo Train (EN):   {len(train_records)}")
    print(f"   - ADReSSo Test  (EN):   {len(test_records)}")
    print(f"   Cartelle popolate in data/dataset e data/transcripts")
    print("="*60)

if __name__ == "__main__":
    main()