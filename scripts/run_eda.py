import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Configurazione stile e output
sns.set_theme(style="whitegrid")
OUT_DIR = Path("plots/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Percorso del file (Assicurati che il nome sia corretto)
PATH_ITALY = "data/metadata/dataset_metadata.csv" 

def simplify_occupation(job):
    """Raggruppa le professioni raw in categorie clinico-sociali per la tesi"""
    job = str(job).lower().strip()
    if job in ['nan', '', 'none', 'pensionato']: return 'Unknown/Retired'
    
    if any(x in job for x in ['impiegat', 'funzionario', 'dirigente', 'banca', 'contabile', 'ragioniere', 'vendite', 'commessa', 'consulenza', 'postale', 'regional', 'comunale', 'telecom', 'enel', 'mps', 'amministrazione']): 
        return 'Office & Admin'
    if any(x in job for x in ['insegnante', 'maestra', 'professore', 'scolastico', 'conservatorio']): 
        return 'Education'
    if any(x in job for x in ['operaio', 'meccanico', 'tecnico', 'autista', 'pescatore', 'agricoltore', 'contadino', 'sottoufficiale', 'geometra', 'ingegniere', 'operaio acquedotto']): 
        return 'Technical & Manual'
    if any(x in job for x in ['medico', 'infermier', 'farmacia', 'estetista']): 
        return 'Healthcare'
    if any(x in job for x in ['casaling', 'sarta', 'assistente']): 
        return 'Home Management'
    
    return 'Other'

def main():
    if not Path(PATH_ITALY).exists():
        print(f"❌ Errore: File {PATH_ITALY} non trovato.")
        return

    print("📊 AVVIO EDA COMPLETA: Generazione Grafici e Report CSV per Italy...")
    df = pd.read_csv(PATH_ITALY)
    
    # Pre-processing
    df['Macro_Occupation'] = df['Occupation'].apply(simplify_occupation)
    edu_order = ['Istruzione Primaria', 'Istruzione Secondaria', 'Istruzione Superiore', 'Laurea', 'Laurea Magistrale']

    # --- 1. GENERAZIONE GRAFICI (PNG) ---
    
    # Grafico 1: Diagnosi
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Diagnosis', hue='Diagnosis', palette="viridis", order=['CTR', 'MCI', 'MILD-AD'], legend=False)
    plt.title("Clinical Class Distribution - Italy Dataset")
    plt.savefig(OUT_DIR / "italy_1_class_distribution.png")

    # Grafico 2: Età
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Diagnosis', y='Age', hue='Diagnosis', palette="Set2", order=['CTR', 'MCI', 'MILD-AD'], legend=False)
    sns.stripplot(data=df, x='Diagnosis', y='Age', color='black', alpha=0.3, order=['CTR', 'MCI', 'MILD-AD'])
    plt.title("Age Distribution by Diagnosis (Italy)")
    plt.savefig(OUT_DIR / "italy_2_age_distribution.png")

    # Grafico 3: Istruzione
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='Education', hue='Education', order=[e for e in edu_order if e in df['Education'].unique()], palette="mako", legend=False)
    plt.title("Education Level (Italy Dataset)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "italy_3_education_distribution.png")

    # Grafico 4: Occupazione
    plt.figure(figsize=(12, 7))
    job_order = df['Macro_Occupation'].value_counts().index
    sns.countplot(data=df, y='Macro_Occupation', hue='Macro_Occupation', order=job_order, palette="rocket", legend=False)
    plt.title("Professional Background Distribution (Italy Dataset)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "italy_4_occupation_distribution.png")

    # --- 2. GENERAZIONE REPORT STATISTICO (CSV) ---
    
    print("📝 Generazione report CSV per analisi testuale...")
    
    # Creiamo un file di riepilogo con diverse sezioni
    with open(OUT_DIR / "italy_eda_statistics.csv", "w", encoding="utf-8") as f:
        f.write("=== ITALY DATASET EDA SUMMARY ===\n\n")
        
        f.write("--- 1. Diagnosis Distribution ---\n")
        df['Diagnosis'].value_counts().to_csv(f, header=["Count"])
        f.write("\n")
        
        f.write("--- 2. Age Statistics per Diagnosis ---\n")
        df.groupby('Diagnosis')['Age'].agg(['mean', 'std', 'min', 'max', 'count']).to_csv(f)
        f.write("\n")
        
        f.write("--- 3. Education Levels ---\n")
        df['Education'].value_counts().to_csv(f, header=["Count"])
        f.write("\n")
        
        f.write("--- 4. Professional Macro-Categories ---\n")
        df['Macro_Occupation'].value_counts().to_csv(f, header=["Count"])
        f.write("\n")
        
        f.write("--- 5. Gender Distribution ---\n")
        df.groupby('Diagnosis')['Gender'].value_counts(normalize=True).unstack().to_csv(f)

    print("\n" + "="*40)
    print(f"🎯 EDA COMPLETATA")
    print(f"Grafici: {OUT_DIR}/*.png")
    print(f"Dati per Gemini: {OUT_DIR / 'italy_eda_statistics.csv'}")
    print("="*40)

if __name__ == "__main__":
    main()