import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- CONFIGURAZIONE ---
INPUT_FILE = "outputs/zero_shot_napoli_sota/REFERTO_NAPOLI_SOTA.csv"

def load_and_clean(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    
    df = pd.read_csv(path)
    
    # Pulizia Etichette
    df['GT'] = df['GT'].astype(str).str.strip()

    # Pulizia Probabilità: alcuni CSV vecchi le salvavano come "28.7%", gestiamo sia i numeri che le stringhe
    for col in ['P1', 'P2']:
        if df[col].dtype == object:
            df[col] = df[col].str.replace('%', '').str.strip().astype(float) / 100.0
            
    # Pulizia del MoCA
    if 'Predicted_MoCA' in df.columns and df['Predicted_MoCA'].dtype == object:
        df['Predicted_MoCA'] = pd.to_numeric(df['Predicted_MoCA'], errors='coerce')
        
    return df

def calculate_metrics(y_true, y_pred, strategy_name):
    # Calcolo Metriche Base
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Sensitivity (Recall Classe 1 - Malati)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (Recall Classe 0 - Sani)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "Strategia": strategy_name,
        "Acc": f"{acc:.2f}",
        "Sens": f"{sens:.2f}",
        "Spec": f"{spec:.2f}",
        "F1": f"{f1:.2f}"
    }

def run_classification_analysis(df, scenario_name, target_mapping):
    print(f"\n{'='*80}")
    print(f"📊 SCENARIO CLASSIFICAZIONE: {scenario_name}")
    print(f"{'='*80}")
    
    df_sub = df[df['GT'].isin(target_mapping.keys())].copy()
    df_sub['Target'] = df_sub['GT'].map(target_mapping)
    y_true = df_sub['Target'].values
    
    if len(df_sub) == 0:
        print("   ⚠️ Nessun dato per questo scenario.")
        return

    print(f"   Campioni Totali: {len(df_sub)}")
    print(f"   Distribuzione Diagnosi: {df_sub['GT'].value_counts().to_dict()}")
    print("-" * 80)

    results = []

    # --- 1. MEAN VOTING (Soft Voting) ---
    probs_mean = (df_sub['P1'] + df_sub['P2']) / 2
    preds_mean = (probs_mean >= 0.5).astype(int)
    results.append(calculate_metrics(y_true, preds_mean, "Mean Voting (Soft)"))

    # --- 2. OR VOTING CLASSICO (Max Prob >= 0.5) ---
    probs_max = df_sub[['P1', 'P2']].max(axis=1)
    preds_or_50 = (probs_max >= 0.5).astype(int)
    results.append(calculate_metrics(y_true, preds_or_50, "OR Voting (T=0.50)"))
    
    # --- 3. OR VOTING TRIAGE (La tua proposta di Tesi: T=0.35) ---
    preds_or_35 = (probs_max >= 0.35).astype(int)
    results.append(calculate_metrics(y_true, preds_or_35, "OR Triage (T=0.35)"))

    # --- 4. AND VOTING (Min Probability) ---
    probs_min = df_sub[['P1', 'P2']].min(axis=1)
    preds_and = (probs_min >= 0.5).astype(int)
    results.append(calculate_metrics(y_true, preds_and, "AND Voting (Min Prob)"))

    # --- 5. MAX CONFIDENCE VOTING ---
    def get_conf_pred(row):
        conf1 = abs(row['P1'] - 0.5)
        conf2 = abs(row['P2'] - 0.5)
        selected_p = row['P1'] if conf1 >= conf2 else row['P2']
        return 1 if selected_p >= 0.5 else 0
    
    preds_conf = df_sub.apply(get_conf_pred, axis=1)
    results.append(calculate_metrics(y_true, preds_conf, "Max Confidence"))

    # Stampa Tabella
    res_df = pd.DataFrame(results)
    cols = ["Strategia", "Acc", "Sens", "Spec", "F1"]
    print(res_df[cols].to_string(index=False))


def run_regression_analysis(df):
    print(f"\n{'='*80}")
    print(f"🧠 SCENARIO REGRESSIONE: Punteggi Cognitivi (MoCA/MMSE)")
    print(f"{'='*80}")
    
    if 'Predicted_MoCA' not in df.columns:
        print("   ⚠️ Colonna 'Predicted_MoCA' non trovata. Hai eseguito la versione aggiornata di napoli.py?")
        return
        
    # Ordiniamo le righe in modo logico per la lettura: dal Sano al Malato grave
    order = ['CTR', 'MCI', 'MILD-AD']
    
    # Raggruppiamo
    stats = df.groupby('GT')['Predicted_MoCA'].agg(['mean', 'std', 'count']).round(2)
    stats.columns = ['Score Stimato (Media)', 'Deviazione Std', 'N. Pazienti']
    
    # Riorganizziamo l'indice
    stats = stats.reindex([x for x in order if x in stats.index])
    
    print(stats.to_string())
    print("-" * 80)
    print("💡 ANALISI: Il modello ha indovinato la progressione se: CTR > MCI > MILD-AD")


def main():
    try:
        df = load_and_clean(INPUT_FILE)
    except Exception as e:
        print(f"❌ Errore: {e}")
        return

    # --- SCENARIO CLASSIFICAZIONE A: PURE AD ---
    map_a = {'MILD-AD': 1, 'CTR': 0}
    run_classification_analysis(df, "PURE AD vs CTR (Esclusi MCI)", map_a)

    # --- SCENARIO CLASSIFICAZIONE B: SCREENING ---
    map_b = {'MILD-AD': 1, 'MCI': 1, 'CTR': 0}
    run_classification_analysis(df, "SCREENING: IMPAIRED (AD + MCI) vs CTR", map_b)
    
    # --- SCENARIO REGRESSIONE ---
    run_regression_analysis(df)

if __name__ == "__main__":
    main()