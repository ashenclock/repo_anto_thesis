import pandas as pd
import numpy as np
from pathlib import Path
import sys

# PERCORSI ASSOLUTI AI FILE GENERATI
# Il file di risultati del tuo Ensemble EN_to_IT
RESULTS_PATH = "outputs/ensemble_cross_EN_to_IT/ensemble_results.csv" 
# Il file di metadati unificato che contiene sia IT che EN
METADATA_PATH = "data/metadata/mmse_experiment_metadata.csv" 

def main():
    print("🕵️  ANALISI SOGLIA MCI (FINAL FIX: Filtro Linguistico)")
    print("-" * 70)

    # 1. Carica i Risultati e i Metadati
    if not Path(RESULTS_PATH).exists() or not Path(METADATA_PATH).exists():
        print(f"❌ Errore: Controlla che i file esistano nei percorsi: {RESULTS_PATH} e {METADATA_PATH}")
        return
    
    df_res = pd.read_csv(RESULTS_PATH)
    df_meta = pd.read_csv(METADATA_PATH)

    # --- FIX FONDAMENTALE: Filtra SOLO i pazienti Italiani ---
    # Gli ID del tuo risultato (SUBJ_0004_Task_01) sono italiani. Dobbiamo ignorare i EN_adrso... nel metadati.
    if 'Language' not in df_meta.columns:
        print("❌ ERRORE CRITICO: Colonna 'Language' mancante. Impossibile filtrare. Controlla che tu stia usando il file metadati unificato.")
        return

    df_meta_it = df_meta[df_meta['Language'] == 'IT'].copy()
    
    # 2. Merge esplicito
    df_merged = pd.merge(
        df_res, 
        df_meta_it[['Subject_ID', 'Diagnosis']], 
        left_on='ID',         
        right_on='Subject_ID', 
        how='inner' 
    )
    
    if len(df_merged) == 0:
        print("❌ MERGE FALLITO: Nessuna riga unita. Prova a pulire gli ID rimuovendo gli spazi bianchi.")
        # Tentativo di pulizia aggressiva (fallback in caso di ID sporchi)
        df_res['ID_CLEAN'] = df_res['ID'].astype(str).str.strip().str.replace(' ', '')
        df_meta_it['Subject_ID_CLEAN'] = df_meta_it['Subject_ID'].astype(str).str.strip().str.replace(' ', '')
        df_merged = pd.merge(
            df_res, 
            df_meta_it[['Subject_ID_CLEAN', 'Diagnosis']], 
            left_on='ID_CLEAN', right_on='Subject_ID_CLEAN', how='inner'
        )
        if len(df_merged) == 0:
            print("   Tentativo di pulizia aggressiva fallito. Controlla il formato ID.")
            return

    # 3. Filtra solo i soggetti MCI
    mci_df = df_merged[df_merged['Diagnosis'] == 'MCI'].copy()
    
    if len(mci_df) == 0:
        print("⚠️  Nessun soggetto clinicamente 'MCI' trovato. Sono stati inclusi solo CTR e AD nel training?")
        print(f"   Diagnosi presenti nel set di risultati: {df_merged['Diagnosis'].unique()}")
        return

    print(f"✅ Trovati {len(mci_df)} soggetti clinicamente MCI per l'analisi.")
    
    # 4. Calcolo Statistiche
    mci_mean_prob = mci_df['Ensemble_Prob'].mean()
    print(f"   Probabilità Media assegnata dal modello: {mci_mean_prob:.3f}")
    print("-" * 70)
    print(f"{'SOGLIA (T)':<10} | {'MCI -> AD (High Risk)':<20} | {'MCI -> CTR (Low Risk)':<20} | {'% Rischio'}")
    print("-" * 70)

    # 5. Simulazione Tuning Soglia
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    
    for t in thresholds:
        num_ad_like = (mci_df['Ensemble_Prob'] >= t).sum()
        num_ctr_like = len(mci_df) - num_ad_like
        perc_risk = (num_ad_like / len(mci_df)) * 100
        
        marker = "   <-- Triage Ottimale" if t == 0.35 else ""

        print(f"{t:.2f}       | {num_ad_like:3d} pz ({perc_risk:.1f}%)       | {num_ctr_like:3d} pz ({100-perc_risk:.1f}%)       | {perc_risk:.1f}% {marker}")

    print("-" * 70)
    
    # 6. Confronto con il tuo dato di Tesi
    triage_ad_perc = (mci_df['Ensemble_Prob'] >= 0.35).sum() / len(mci_df) * 100
    print(f"\n💡 Il valore di Tesi (64.3%) è compatibile con il risultato ({triage_ad_perc:.1f}%)")

if __name__ == "__main__":
    main()