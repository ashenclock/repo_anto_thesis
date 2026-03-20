import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import warnings
import yaml

warnings.filterwarnings("ignore")

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
    }

def print_summary(title, means, stds):
    print("\n" + "-"*65)
    print(f"🔹 {title}")
    print("-"*65)
    print(f"{'Metric':<15} | {'Mean':<10} | {'Std Dev':<10}")
    print("-"*65)
    for metric in means.index:
        print(f"{metric.capitalize():<15} | {means[metric]:.4f}     | ± {stds[metric]:.4f}")

def load_folds_map(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    folds_path = config['data']['folds_file']
    df_folds = pd.read_csv(folds_path)
    id_col = 'Subject_ID' if 'Subject_ID' in df_folds.columns else 'ID'
    return df_folds.set_index(id_col)['kfold'].to_dict()

def main():
    parser = argparse.ArgumentParser(description="Calcolo Metriche Paziente Avanzate")
    parser.add_argument("--csv", type=str, required=True, help="File preds_COMBINED_....csv")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    file_path = Path(args.csv)
    if not file_path.exists():
        print(f"❌ File non trovato: {file_path}")
        return

    df = pd.read_csv(file_path)

    # 1. Recupero Fold se mancanti
    def extract_patient(full_id): return full_id.split('_Task_')[0]
    df['Patient_ID'] = df['ID'].apply(extract_patient)
    
    if 'kfold' not in df.columns:
        print("⚠️  Recupero kfold da config...")
        try:
            folds_map = load_folds_map(args.config)
            df['kfold'] = df['Patient_ID'].map(folds_map)
            df = df.dropna(subset=['kfold'])
        except: return

    # 2. Logica di Aggregazione Avanzata
    # Funzione custom per Max Confidence
    def max_confidence_voting(probs):
        # probs è una serie di probabilità (es. [0.6, 0.9])
        # Calcoliamo la "fiducia": distanza da 0.5
        confidences = abs(probs - 0.5)
        # Indice della più fiduciosa
        idx_max = confidences.idxmax()
        # Restituiamo la probabilità originale più fiduciosa
        return probs[idx_max]

    patient_df = df.groupby('Patient_ID').agg({
        'Prob': ['mean', max_confidence_voting, lambda x: list(x)],
        'Label': 'first',
        'kfold': 'first'
    }).reset_index()
    
    # Appiattiamo le colonne
    patient_df.columns = ['Patient_ID', 'Soft_Prob', 'MaxConf_Prob', 'All_Probs', 'Label', 'kfold']

    # --- STRATEGIE DI PREDICITON ---
    
    # A. SOFT VOTING (Classica media)
    patient_df['Pred_Soft'] = (patient_df['Soft_Prob'] >= 0.5).astype(int)
    
    # B. MAX CONFIDENCE (Fidati di chi è più sicuro)
    patient_df['Pred_MaxConf'] = (patient_df['MaxConf_Prob'] >= 0.5).astype(int)
    
    # C. OR VOTING (Se almeno uno dice AD -> AD). Utile per Sensitivity.
    # Controlliamo se nella lista delle probabilità ce n'è almeno una > 0.5
    patient_df['Pred_OR'] = patient_df['All_Probs'].apply(lambda x: 1 if any(p >= 0.5 for p in x) else 0)

    # 3. Calcolo Metriche CV
    soft_res, maxconf_res, or_res = [], [], []
    folds = sorted(patient_df['kfold'].unique())

    print("\n📊 ANALISI PER FOLD:")
    for fold in folds:
        data = patient_df[patient_df['kfold'] == fold]
        y = data['Label']
        
        s_m = calculate_metrics(y, data['Pred_Soft'])
        m_m = calculate_metrics(y, data['Pred_MaxConf'])
        o_m = calculate_metrics(y, data['Pred_OR'])
        
        soft_res.append(s_m)
        maxconf_res.append(m_m)
        or_res.append(o_m)
        
        print(f"   Fold {int(fold)}: Soft F1={s_m['f1']:.2f} | MaxConf F1={m_m['f1']:.2f} | OR F1={o_m['f1']:.2f}")

    # 4. Stampa Report
    print("\n" + "="*65)
    print(f"🏆 DIAGNOSI CLINICA PAZIENTE ({len(folds)} Folds)")
    print("="*65)
    
    soft_df = pd.DataFrame(soft_res)
    max_df = pd.DataFrame(maxconf_res)
    or_df = pd.DataFrame(or_res)
    
    print_summary("1. SOFT VOTING (Consigliato per bilanciamento)", soft_df.mean(), soft_df.std())
    print_summary("2. MAX CONFIDENCE (Fiducia massima)", max_df.mean(), max_df.std())
    print_summary("3. OR VOTING (Ottimizzato per Sensitivity)", or_df.mean(), or_df.std())
    print("="*65)

    # Salva
    out_file = file_path.parent / f"report_advanced_{file_path.name}"
    patient_df.drop(columns=['All_Probs']).to_csv(out_file, index=False)
    print(f"\n💾 Report avanzato salvato: {out_file}")

if __name__ == "__main__":
    main()