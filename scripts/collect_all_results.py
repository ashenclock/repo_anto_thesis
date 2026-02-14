import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
import warnings

warnings.filterwarnings("ignore")

# Configurazione cartelle di ricerca
SEARCH_PATHS = [Path("results_csv"), Path("results_csv_combined"), Path("outputs")]
OUTPUT_FILE = "final_report_summary.csv"

def calculate_metrics_raw(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except: sens, spec = 0.0, 0.0
    return acc, f1, prec, sens, spec

def format_ms(mean, std):
    if pd.isna(std) or std == 0: return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"

def main():
    print("🔍 Scansione globale per generare final_report_summary.csv...")
    all_rows = []
    found_files = []
    for base_dir in SEARCH_PATHS:
        if base_dir.exists(): found_files.extend(list(base_dir.rglob("*.csv")))
    
    # Filtriamo i file che contengono predizioni (evitiamo i summary stessi)
    valid_files = [f for f in found_files if ("preds" in f.name or "results" in f.name) and "summary" not in f.name]
    
    for file_path in valid_files:
        try:
            df = pd.read_csv(file_path)
            # Normalizzazione nomi colonne
            cols_map = {'Target': 'Label', 'target': 'Label', 'id': 'ID', 'Subject_ID': 'ID', 'Ensemble_Prob': 'Prob', 'prob': 'Prob'}
            df.rename(columns=cols_map, inplace=True)
            
            if 'Label' not in df.columns: continue
            
            # Determina la predizione
            if 'Prob' in df.columns: y_pred = (df['Prob'] >= 0.5).astype(int)
            elif 'Pred' in df.columns: y_pred = df['Pred'].astype(int)
            else: continue

            # Estrazione Info dal Path
            path_str = str(file_path).lower()
            model_name = file_path.stem.replace("preds_", "").replace("results_", "")
            
            # Categoria
            if "tabular" in path_str: group = "1_Baseline_Tabular"
            elif "xphonebert" in path_str: group = "2_Phonetic"
            elif "cross" in path_str: group = "4_SOTA_CrossDataset"
            elif "combined" in path_str: group = "3_SOTA_Combined"
            else: group = "0_Text_Baseline"

            # Se ci sono i fold, calcoliamo la std reale
            if 'kfold' in df.columns:
                fold_stats = []
                for fold in sorted(df['kfold'].unique()):
                    sub = df[df['kfold'] == fold]
                    p = (sub['Prob'] >= 0.5).astype(int) if 'Prob' in sub.columns else sub['Pred']
                    fold_stats.append(calculate_metrics_raw(sub['Label'], p))
                f_df = pd.DataFrame(fold_stats, columns=['acc', 'f1', 'prec', 'sens', 'spec'])
                
                all_rows.append({
                    "Group": group, "Model": model_name,
                    "Accuracy": f_df['acc'].mean(), "Acc_std": f_df['acc'].std(),
                    "F1": f_df['f1'].mean(), "F1_std": f_df['f1'].std(),
                    "Sens": f_df['sens'].mean(), "Sens_std": f_df['sens'].std()
                })
            else:
                acc, f1, prec, sens, spec = calculate_metrics_raw(df['Label'], y_pred)
                all_rows.append({
                    "Group": group, "Model": model_name,
                    "Accuracy": acc, "Acc_std": 0, "F1": f1, "F1_std": 0, "Sens": sens, "Sens_std": 0
                })
        except: continue

    if not all_rows:
        print("❌ Nessun dato trovato."); return

    df_final = pd.DataFrame(all_rows)
    
    # Formattazione finale mu +- sigma
    df_final['Accuracy (avg)'] = df_final.apply(lambda x: format_ms(x['Accuracy'], x['Acc_std']), axis=1)
    df_final['F1-Score (avg)'] = df_final.apply(lambda x: format_ms(x['F1'], x['F1_std']), axis=1)
    df_final['Sensitivity (avg)'] = df_final.apply(lambda x: format_ms(x['Sens'], x['Sens_std']), axis=1)

    # Pulizia e ordinamento
    df_report = df_final[['Group', 'Model', 'Accuracy (avg)', 'F1-Score (avg)', 'Sensitivity (avg)']]
    df_report = df_report.sort_values(['Group', 'Accuracy (avg)'], ascending=[True, False])

    df_report.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ FILE GENERATO: {OUTPUT_FILE}")
    print("-" * 80)
    print(df_report.to_string(index=False))

if __name__ == "__main__":
    main()