import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURAZIONE ---
# Assumo che tu abbia salvato il contenuto in questo file
PATH_INPUT = "data/metadata/italy_clinical_complete.csv" 
OUT_DIR = Path("plots/clinical_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🧹 Pulizia dati clinici in corso...")
    df = pd.read_csv(PATH_INPUT)

    # 1. Rimuoviamo le righe "fantasma" (quelle che hanno quasi tutto NaN)
    # Teniamo solo le righe dove NOME o COGNOME sono presenti
    df = df.dropna(subset=['NOME', 'COGNOME'], how='all')
    
    # 2. Pulizia colonne numeriche
    cols_to_fix = ['MMSE', 'MOCA', 'ETA\'', 'FLUIDITA\' CATEGORIE', 'FLUIDITA\' LETTERE']
    for col in cols_to_fix:
        if col in df.columns:
            # Rimuove eventuali stringhe tipo "N.A." o "N.C." e converte in numero
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    # 3. Standardizzazione Diagnosi
    df['Diagnosis'] = df['Diagnosis'].str.strip().str.upper()
    df['Diagnosis'] = df['Diagnosis'].replace({'AD LIEVE': 'MILD-AD', 'CONTROLLI': 'CTR'})

    # 4. Salvataggio CSV pulito per Gemini CLI
    df.to_csv("data/metadata/italy_clinical_cleaned.csv", index=False)
    print(f"✅ CSV pulito salvato: data/metadata/italy_clinical_cleaned.csv")

    # --- GENERAZIONE GRAFICI ---
    sns.set_theme(style="whitegrid")
    diag_order = ['CTR', 'MCI', 'MILD-AD']
    palette = {"CTR": "#66c2a5", "MCI": "#8da0cb", "MILD-AD": "#fc8d62"}

    # Grafico MMSE
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Diagnosis', y='MMSE', order=diag_order, palette=palette, hue='Diagnosis', legend=False)
    sns.stripplot(data=df, x='Diagnosis', y='MMSE', order=diag_order, color='black', alpha=0.3)
    plt.title("Distribuzione MMSE Reale (Dataset Italy)", fontsize=14)
    plt.ylim(0, 32)
    plt.savefig(OUT_DIR / "boxplot_mmse_real.png")

    # Grafico MoCA
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Diagnosis', y='MOCA', order=diag_order, palette=palette, hue='Diagnosis', legend=False)
    sns.stripplot(data=df, x='Diagnosis', y='MOCA', order=diag_order, color='black', alpha=0.3)
    plt.title("Distribuzione MoCA Reale (Dataset Italy)", fontsize=14)
    plt.ylim(0, 32)
    plt.savefig(OUT_DIR / "boxplot_moca_real.png")

    # --- REPORT STATISTICO ---
    stats = df.groupby('Diagnosis')[['MMSE', 'MOCA', 'ETA\'']].agg(['mean', 'std', 'count']).round(2)
    print("\n📊 STATISTICHE DESCRITTIVE REALI:")
    print(stats)
    stats.to_csv(OUT_DIR / "summary_stats_real.csv")

if __name__ == "__main__":
    main()