import pandas as pd
import numpy as np
from pathlib import Path
import torch
import warnings
import yaml
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score

# Setup per importare i moduli dalla cartella src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import build_model
from src.config import load_config

warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE ---
PATH_T1 = "results_csv/preds_multimodal_text_bert-base-italian-xxl-cased_Task_01.csv"
PATH_T2 = "results_csv/preds_multimodal_text_bert-base-italian-xxl-cased_Task_02.csv"
FOLDS_FILE = "data/metadata/dataset_folds.csv"
CONFIG_FILE = "config.yaml"

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'Acc': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'Sens': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Spec': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'Prec': precision_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def print_model_summary():
    """Stampa il dettaglio dei parametri del modello multimodale SOTA."""
    print("\n" + "="*50)
    print(f"{'ARCHITETTURA MODELLO SOTA (best_model.pt)':^50}")
    print("="*50)
    
    try:
        # Carica configurazione
        config = load_config(CONFIG_FILE)
        # Forziamo il backbone corretto per il riepilogo (BERT XXL)
        config.model.text.name = "dbmdz/bert-base-italian-xxl-cased"
        config.model.output_dim = 2

        # Inizializza il modello (senza caricare i pesi, ci serve solo la struttura)
        model = build_model(config)

        total_params = 0
        print(f"{'Blocco Funzionale':<25} | {'Parametri':>15}")
        print("-" * 50)
        
        for name, child in model.named_children():
            params = sum(p.numel() for p in child.parameters())
            print(f"{name:<25} | {params:>15,}")
            total_params += params
            
        print("-" * 50)
        print(f"{'TOTALE PARAMETRI':<25} | {total_params:>15,}")
        print("="*50 + "\n")
    except Exception as e:
        print(f"⚠️ Impossibile generare il summary: {e}")

def main():
    # 1. Stampa il Summary Tecnico
    print_model_summary()

    print("🧪 CONFRONTO FINALE STRATEGIE DI FUSIONE (CLASSE IMPAIRED N=139)")
    print("="*100)

    # 2. Caricamento e Allineamento Dati
    if not (Path(PATH_T1).exists() and Path(PATH_T2).exists()):
        print("❌ File delle predizioni non trovati. Verifica i percorsi.")
        return

    df1 = pd.read_csv(PATH_T1)
    df2 = pd.read_csv(PATH_T2)
    df_folds = pd.read_csv(FOLDS_FILE).rename(columns={'Subject_ID': 'PID', 'ID': 'PID'})

    df1['PID'] = df1['ID'].apply(lambda x: str(x).split('_Task_')[0])
    df2['PID'] = df2['ID'].apply(lambda x: str(x).split('_Task_')[0])

    m1 = df1.groupby('PID')['Prob'].max().reset_index().rename(columns={'Prob': 'P1'})
    m2 = df2.groupby('PID')['Prob'].max().reset_index().rename(columns={'Prob': 'P2'})
    
    meta_df = pd.merge(m1, m2, on='PID')
    meta_df = pd.merge(meta_df, df_folds[['PID', 'Diagnosis', 'kfold']], on='PID')

    # Target: 0 = CTR, 1 = MCI+AD
    meta_df['Target'] = meta_df['Diagnosis'].apply(lambda x: 1 if x in ['MCI', 'AD', 'MILD-AD'] else 0)

    X = meta_df[['P1', 'P2']].values
    y = meta_df['Target'].values
    folds = meta_df['kfold'].values

    stats = {
        'Soft Voting (Mean)': [],
        'OR Voting (Max Prob)': [],
        'OR Triage (T=0.35)': [],
        'Meta-Classifier (XGBoost)': []
    }

    # 3. Cross-Validation
    unique_folds = sorted(np.unique(folds))
    for f_id in unique_folds:
        train_idx = np.where(folds != f_id)[0]
        val_idx = np.where(folds == f_id)[0]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]

        # A. Soft Voting
        p_mean = np.mean(X_val, axis=1)
        stats['Soft Voting (Mean)'].append(calculate_metrics(y_test, (p_mean >= 0.5).astype(int)))

        # B. OR Voting
        p_max = np.max(X_val, axis=1)
        stats['OR Voting (Max Prob)'].append(calculate_metrics(y_test, (p_max >= 0.5).astype(int)))

        # C. OR Triage
        stats['OR Triage (T=0.35)'].append(calculate_metrics(y_test, (p_max >= 0.35).astype(int)))

        # D. XGBoost
        clf = XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.05, eval_metric='logloss')
        clf.fit(X_train, y_train)
        stats['Meta-Classifier (XGBoost)'].append(calculate_metrics(y_test, clf.predict(X_val)))

    # 4. Formattazione Tabella Finale
    final_rows = []
    for name, metrics_list in stats.items():
        df_res = pd.DataFrame(metrics_list)
        final_rows.append({
            'Method': name,
            'Accuracy': f"{df_res['Acc'].mean():.2f} ± {df_res['Acc'].std():.2f}",
            'F1-Score': f"{df_res['F1'].mean():.2f} ± {df_res['F1'].std():.2f}",
            'Sens (Recall)': f"{df_res['Sens'].mean():.2f} ± {df_res['Sens'].std():.2f}",
            'Specificity': f"{df_res['Spec'].mean():.2f} ± {df_res['Spec'].std():.2f}",
            '_sort': df_res['Acc'].mean()
        })

    report_df = pd.DataFrame(final_rows).sort_values('_sort', ascending=False).drop(columns=['_sort'])
    print(report_df.to_string(index=False))
    print("="*100)
    
    report_df.to_csv("TABELLA_TESI_FINALE_CON_SUMMARY.csv", index=False)

if __name__ == "__main__":
    main()