import pandas as pd
from pathlib import Path

# INPUT 1: Il file "Master" attuale (Lo usiamo SOLO per prendere gli inglesi)
path_master = "data/metadata/multilingual_meta_full.csv"

# INPUT 2: Il file "Sicily Originale" (Lo usiamo per prendere TUTTI gli italiani, inclusi MCI)
path_sicily = "data/metadata/speech_metadata.csv"

# OUTPUT: Il nuovo file dedicato per questo esperimento
path_output = "data/metadata/mmse_experiment_metadata.csv"

def main():
    print("🛠️  Creazione Metadata Dedicato per Regressione MMSE...")
    
    # 1. Recuperiamo gli INGLESI dal Master
    df_master = pd.read_csv(path_master)
    df_en = df_master[df_master['Language'] == 'EN'].copy()
    print(f"   -> Recuperati {len(df_en)} campioni Inglesi (ADReSSo Full).")
    
    # 2. Recuperiamo gli ITALIANI dal file Sicily (che ha gli MCI)
    df_sicily = pd.read_csv(path_sicily)
    
    # Filtriamo:
    # - Solo Task 01 (Descrizione Immagine)
    df_it = df_sicily[df_sicily['Subject_ID'].str.contains("Task_01")].copy()
    
    # Standardizziamo le colonne mancanti
    df_it['Language'] = 'IT'
    df_it['Dataset'] = 'Sicily'
    
    # Diagnosi trovate
    diags = df_it['Diagnosis'].unique()
    print(f"   -> Recuperati {len(df_it)} campioni Italiani.")
    print(f"      Diagnosi incluse: {diags}") # Qui dovresti vedere MCI!

    # 3. Uniamo tutto
    # Concateniamo EN e IT
    df_new = pd.concat([df_en, df_it], ignore_index=True)
    
    # 4. Salviamo il nuovo file
    df_new.to_csv(path_output, index=False)
    print("\n" + "="*50)
    print(f"✅ FILE CREATO: {path_output}")
    print(f"   Totale righe: {len(df_new)}")
    print("="*50)

if __name__ == "__main__":
    main()