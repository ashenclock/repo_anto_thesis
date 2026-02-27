import pandas as pd
import shutil
import numpy as np
from pathlib import Path
import sys

# --- CONFIGURAZIONE ---
META_FILE = Path("data/metadata/mmse_experiment_metadata.csv")
SCORES_EN = Path("data/metadata/adresso_FULL_mmse.csv")
EXCEL_IT  = Path("data/metadata/dataset_10_02_Foglio_1_-_dataset.csv")

# Cartella di OUTPUT
DEST_ROOT = Path("data/qualitative_samples")

# Sorgenti (Ibride)
MODEL_DIRS = {
    "Whisper_V3": {
        "IT": Path("data/transcripts/WhisperX_large-v3/Task_01"),
        "EN": Path("/home/speechlab/Desktop/pappagari2021-interspeech-replication/transcripts/large-v3")
    },
    "Parakeet": {
        "IT": Path("data/transcripts/Parakeet_Unified/Task_01"),
        "EN": Path("/home/speechlab/Desktop/pappagari2021-interspeech-replication/transcripts/parakeet-tdt-0.6b-v2")
    },
    "CrisperWhisper": {
        "IT": Path("data/transcripts/WhisperX_nyrahealth/faster_CrisperWhisper/Task_01"), 
        "EN": Path("/home/speechlab/Desktop/pappagari2021-interspeech-replication/transcripts/WhisperX_nyrahealth/faster_CrisperWhisper")
    }
}

def get_clean_filename(subject_id, lang):
    """Calcola il nome file sorgente probabile"""
    if lang == 'IT':
        return subject_id.split('_Task_')[0] + ".txt"
    else:
        base = subject_id.split('_Task_')[0]
        return base.replace("EN_TEST_", "").replace("EN_", "") + ".txt"

def load_italian_scores():
    """Carica i punteggi reali dall'Excel italiano"""
    print("⏳ Caricamento punteggi reali Excel Italia...")
    if not EXCEL_IT.exists():
        print(f"⚠️  Excel non trovato: {EXCEL_IT}. Userò fallback.")
        return {}

    df = pd.read_csv(EXCEL_IT)
    
    # Filtra solo le righe valide
    valid_diags = ['AD', 'AD-LIEVE', 'MCI', 'CTR', 'CONTROLLI']
    df = df[df['Malattie Diagnosticate'].isin(valid_diags)].reset_index(drop=True)
    
    scores_map = {}
    for idx, row in df.iterrows():
        # L'ID SUBJ_0001 corrisponde all'indice 0 + 1 dell'excel pulito
        subj_num = idx + 1
        
        try:
            val = str(row['MMSE']).replace(',', '.').strip()
            if val == '' or val.lower() == 'nan':
                score = 99.0 # Assegna un valore alto se mancante per escluderlo dai "peggiori"
            else:
                score = float(val)
        except:
            score = 99.0
            
        scores_map[subj_num] = score
        
    return scores_map

def find_and_copy(src_folder, filename, dest_folder, new_name):
    """Cerca e copia il file"""
    if not src_folder.exists(): return False
    
    candidates = [filename, filename.replace(".txt", ".wav.txt")]
    
    for cand in candidates:
        # 1. Cerca diretto
        fpath = src_folder / cand
        if fpath.exists() and fpath.stat().st_size > 5:
            shutil.copy2(fpath, dest_folder / new_name)
            return True
            
        # 2. Cerca ricorsivo
        matches = list(src_folder.rglob(cand))
        valid_match = next((m for m in matches if m.stat().st_size > 5), None)
        if valid_match:
            shutil.copy2(valid_match, dest_folder / new_name)
            return True
                
    return False

def main():
    print(f"📦 SALVATAGGIO CAMPIONI QUALITATIVI (Casi Reali Peggiori)")
    
    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Carica Mappe Punteggi
    it_scores_map = load_italian_scores()
    
    df_en_scores = pd.read_csv(SCORES_EN)
    df_en_scores['ID'] = df_en_scores['ID'].astype(str).str.strip().str.replace('"', '')
    en_scores_map = df_en_scores.set_index('ID')['Score'].to_dict()

    # 2. Carica Metadata
    df = pd.read_csv(META_FILE)

    # 3. Funzione di Sorting Intelligente
    def get_real_mmse(row):
        sid = row['Subject_ID']
        
        if row['Language'] == 'EN':
            clean = sid.split('_Task_')[0].replace("EN_TEST_", "").replace("EN_", "")
            return en_scores_map.get(clean, 99.0)
        else:
            try:
                num = int(sid.split('_')[1])
                return it_scores_map.get(num, 99.0)
            except:
                return 99.0

    df['real_mmse'] = df.apply(get_real_mmse, axis=1)

    # Filtra solo AD (o Mild-AD)
    df_ad = df[df['Diagnosis'].astype(str).str.contains('AD')].copy()
    
    # Ordina per MMSE crescente (i più bassi in cima)
    it_sorted = df_ad[df_ad['Language'] == 'IT'].sort_values('real_mmse')
    en_sorted = df_ad[df_ad['Language'] == 'EN'].sort_values('real_mmse')

    datasets = [("IT", it_sorted), ("EN", en_sorted)]

    for lang, group in datasets:
        print(f"\n🌍 Lingua: {lang}")
        
        for model_name, paths in MODEL_DIRS.items():
            model_out_dir = DEST_ROOT / lang / model_name
            model_out_dir.mkdir(parents=True, exist_ok=True)
            
            found_count = 0
            for _, row in group.iterrows():
                if found_count >= 3: break
                
                sid = row['Subject_ID']
                score = row['real_mmse']
                
                # --- FIX CRUCIALE PER IL CRASH ---
                if pd.isna(score) or score == 99.0: 
                    continue # Salta se NaN o valore fittizio alto
                
                # Prendiamo solo i casi "interessanti" (sotto 26)
                if score > 26: 
                    continue

                src_fname = get_clean_filename(sid, lang)
                clean_id = src_fname.replace(".txt", "")
                
                # Nome file: MMSE_12_SUBJ_0045.txt
                try:
                    dest_fname = f"MMSE_{int(score):02d}_{clean_id}.txt"
                except ValueError:
                    print(f"⚠️ Errore conversione score {score} per {sid}. Salto.")
                    continue
                
                if find_and_copy(paths[lang], src_fname, model_out_dir, dest_fname):
                    print(f"   ✅ Copiato ({model_name}): {dest_fname} (MMSE: {score})")
                    found_count += 1
            
            if found_count == 0:
                print(f"   ❌ Nessun file trovato per {model_name} (o nessun paziente grave ha la trascrizione)!")

    print(f"\n✅ Fatto. Controlla: {DEST_ROOT}")

if __name__ == "__main__":
    main()