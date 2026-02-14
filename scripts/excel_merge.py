import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURAZIONE PERCORSI ---
PATH_EXCEL = "data/metadata/punteggi_italy.xlsx"
OUT_CSV = "data/metadata/italy_clinical_final.csv"
OUT_PLOT_DIR = Path("plots/clinical_analysis_fixed")
OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

def to_num(val):
    """Converte in numero gestendo virgole e sporcizia testuale"""
    try:
        if pd.isna(val): return np.nan
        # Gestisce il caso di stringhe con virgola (es. "24,5")
        s = str(val).replace(',', '.').strip()
        # Rimuove eventuali caratteri non numerici residui
        return float(s)
    except:
        return np.nan

def main():
    if not Path(PATH_EXCEL).exists():
        print(f"❌ Errore: File {PATH_EXCEL} non trovato!")
        return

    print(f"📂 Lettura Excel: {PATH_EXCEL}")
    xls = pd.ExcelFile(PATH_EXCEL)
    all_data = []

    # --- 1. ESTRAZIONE DATI PER FOGLIO (LOGICA INDICIZZATA) ---
    
    # Foglio MCI
    if "MCI" in xls.sheet_names:
        df_mci = pd.read_excel(xls, sheet_name="MCI")
        print(f"   -> Processing MCI ({len(df_mci)} righe)")
        for _, row in df_mci.iterrows():
            if pd.isna(row.iloc[1]): continue
            all_data.append({
                'ID': f"{row.iloc[1]} {row.iloc[2]}".strip(),
                'Age': to_num(row.iloc[5]),
                'MMSE': to_num(row.iloc[16]), 
                'MoCA': to_num(row.iloc[38]),
                'Diagnosis': 'MCI'
            })

    # Foglio AD lieve
    if "AD lieve" in xls.sheet_names:
        df_ad = pd.read_excel(xls, sheet_name="AD lieve")
        print(f"   -> Processing AD lieve ({len(df_ad)} righe)")
        for _, row in df_ad.iterrows():
            if pd.isna(row.iloc[1]): continue
            all_data.append({
                'ID': f"{row.iloc[1]} {row.iloc[2]}".strip(),
                'Age': to_num(row.iloc[5]),
                'MMSE': to_num(row.iloc[20]), # MMSE spostato in questo foglio
                'MoCA': to_num(row.iloc[46]), # MoCA spostato in questo foglio
                'Diagnosis': 'MILD-AD'
            })

    # Foglio Controlli
    if "Controlli" in xls.sheet_names:
        df_ctr = pd.read_excel(xls, sheet_name="Controlli")
        print(f"   -> Processing Controlli ({len(df_ctr)} righe)")
        for _, row in df_ctr.iterrows():
            if pd.isna(row.iloc[1]): continue
            # FIX: In questo foglio l'MMSE reale è sotto l'etichetta CDR (colonna 16)
            all_data.append({
                'ID': f"{row.iloc[1]} {row.iloc[2]}".strip(),
                'Age': to_num(row.iloc[5]),
                'MMSE': to_num(row.iloc[16]), 
                'MoCA': to_num(row.iloc[38]),
                'Diagnosis': 'CTR'
            })

    # Creazione DataFrame Finale
    df_final = pd.DataFrame(all_data)
    df_final = df_final.dropna(subset=['MMSE', 'MoCA'], how='all')
    
    # Salvataggio CSV
    df_final.to_csv(OUT_CSV, index=False)
    print(f"\n✅ CSV unificato creato: {OUT_CSV}")

    # --- 2. GENERAZIONE GRAFICI RISCALATI (0-30) ---
    
    sns.set_theme(style="whitegrid")
    diag_order = ['CTR', 'MCI', 'MILD-AD']
    palette = {"CTR": "#66c2a5", "MCI": "#8da0cb", "MILD-AD": "#fc8d62"}
    
    # Soglie cliniche di riferimento (linee tratteggiate)
    thresholds = {'MMSE': 24, 'MoCA': 26}

    for test in ['MMSE', 'MoCA']:
        plt.figure(figsize=(10, 7))
        
        # Boxplot
        sns.boxplot(
            data=df_final, x='Diagnosis', y=test, order=diag_order, 
            palette=palette, hue='Diagnosis', legend=False,
            width=0.6, linewidth=2, fliersize=0
        )
        
        # Sovrapposizione punti individuali
        sns.stripplot(
            data=df_final, x='Diagnosis', y=test, order=diag_order, 
            color='black', alpha=0.4, jitter=True, size=5
        )
        
        # Linea soglia clinica
        plt.axhline(y=thresholds[test], color='red', linestyle='--', linewidth=1.5, 
                    label=f'Cut-off Clinico ({thresholds[test]})')
        
        # --- RISCALAMENTO ASSE Y ---
        plt.ylim(-0.5, 31) 
        plt.yticks(np.arange(0, 31, 2)) # Un segno ogni 2 punti
        
        # Estetica
        plt.title(f"Italy Clinical Dataset: Distribuzione {test}\n(Scala Clinica Completa 0-30)", fontsize=15, pad=15)
        plt.ylabel(f"Punteggio {test}", fontsize=12)
        plt.xlabel("Diagnosi", fontsize=12)
        plt.legend(loc='lower left')
        
        # Salva
        plot_path = OUT_PLOT_DIR / f"final_boxplot_{test.lower()}_0_30.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Grafico {test} salvato: {plot_path}")

    # --- 3. STATISTICHE DI RIEPILOGO ---
    print("\n📊 MEDIE FINALI PER GRUPPO:")
    summary = df_final.groupby('Diagnosis')[['MMSE', 'MoCA']].agg(['mean', 'std', 'count']).round(2)
    print(summary)

if __name__ == "__main__":
    main()