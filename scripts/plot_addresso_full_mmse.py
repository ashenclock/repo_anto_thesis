import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# PERCORSI ASSOLUTI
PATH_META = "data/metadata/multilingual_meta_full.csv"
PATH_TRAIN_SCORES = "/home/speechlab/Desktop/repo_anto_thesis/data/metadata/adresso-train-mmse-scores.csv"
PATH_TEST_SCORES = "/home/speechlab/Desktop/repo_anto_thesis/data/metadata/label_test_task2.csv"
OUT_DIR = Path("outputs/descriptive_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_id(val):
    return str(val).strip().replace('"', '')

def main():
    print("📊 Generazione Statistiche Descrittive ADReSSo (Full MMSE)...")

    # 1. Carica il file Master per recuperare le diagnosi ufficiali
    df_meta = pd.read_csv(PATH_META)
    df_meta = df_meta[df_meta['Language'] == 'EN'].copy()
    
    # Creiamo una colonna Clean_ID per il match (es. adrso024 o adrsdt15)
    def extract_id(full_id):
        # Toglie EN_, TEST_ e _Task_01
        clean = full_id.replace("EN_", "").replace("TEST_", "").split("_Task_")[0]
        return clean.strip()
    
    df_meta['Clean_ID'] = df_meta['Subject_ID'].apply(extract_id)
    id_to_diag = dict(zip(df_meta['Clean_ID'], df_meta['Diagnosis']))

    # 2. Carica Punteggi TRAIN
    # Header: "","adressfname","mmse","dx"
    df_tr = pd.read_csv(PATH_TRAIN_SCORES)
    df_tr = df_tr[['adressfname', 'mmse']].rename(columns={'adressfname': 'ID', 'mmse': 'MMSE'})
    df_tr['Set'] = 'Train'

    # 3. Carica Punteggi TEST
    # Header: "ID","MMSE"
    df_ts = pd.read_csv(PATH_TEST_SCORES)
    df_ts = df_ts[['ID', 'MMSE']]
    df_ts['Set'] = 'Test'

    # 4. Unione
    df_all = pd.concat([df_tr, df_ts], ignore_index=True)
    df_all['ID'] = df_all['ID'].apply(clean_id)
    
    # Associa Diagnosi dal metadata (per avere CTR/AD standard)
    df_all['Diagnosis'] = df_all['ID'].map(id_to_diag)
    
    # Rimuovi eventuali mismatch
    df_all = df_all.dropna(subset=['Diagnosis'])
    
    # Standardizzazione etichette
    df_all['Diagnosis'] = df_all['Diagnosis'].replace({'cn': 'CTR', 'ad': 'AD'})

    # 5. STATISTICHE
    print("\n📈 STATISTICHE MMSE ADRESSO (TRAIN + TEST):")
    stats = df_all.groupby('Diagnosis')['MMSE'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(stats)

    # 6. PLOT
    plt.figure(figsize=(10, 7))
    
    # Creiamo una palette coerente (Sani = Verde/Blu, Malati = Rosso/Arancio)
    palette = {"CTR": "#66c2a5", "AD": "#fc8d62"}
    order = ["CTR", "AD"]

    # Boxplot
    sns.boxplot(
        data=df_all, x='Diagnosis', y='MMSE', 
        order=order, palette=palette, width=0.5,
        linewidth=2, fliersize=0 # fliersize=0 perché usiamo stripplot
    )
    
    # Aggiungiamo i singoli punti colorati per Set (Train/Test) per vedere la densità
    sns.stripplot(
        data=df_all, x='Diagnosis', y='MMSE', 
        order=order, hue='Set', palette={"Train": "black", "Test": "gray"},
        alpha=0.4, jitter=True, dodge=True
    )

    plt.title("ADReSSo Dataset: Clinical MMSE Distribution (Ground Truth)\nFull Dataset (N=237)", fontsize=15, pad=15)
    plt.ylabel("Clinical MMSE Score (0-30)", fontsize=13)
    plt.xlabel("Clinical Diagnosis", fontsize=13)
    plt.ylim(-1, 31) # Range MMSE
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Data Partition", loc='lower left')

    save_path = OUT_DIR / "adresso_full_mmse_boxplot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Grafico salvato in: {save_path}")

if __name__ == "__main__":
    main()