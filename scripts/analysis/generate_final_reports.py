import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE PERCORSI ---
SEARCH_DIRS = [Path("results_csv"), Path("results_csv_combined"), Path("outputs")]
OUTPUT_DIR = Path("outputs/thesis_reports_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def calculate_metrics_raw(y_true, y_pred):
    if len(y_true) == 0: return 0, 0, 0, 0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, f1, sens, spec

def format_mean_std(mean, std):
    if pd.isna(std) or std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"

def get_model_and_task(file_path):
    """Estrae Model Name e Task Name dal nome del file"""
    fname = file_path.stem
    # Esempio: preds_multimodal_text_bert_Task_01
    task = "Global"
    if "Task_" in fname:
        parts = fname.split("Task_")
        task = "Task_" + parts[-1]
        model = parts[0].replace("preds_", "").replace("results_", "").strip("_")
    else:
        model = fname.replace("preds_", "").replace("results_", "")
    
    # Categorizzazione rapida per il report
    cat = "Text-Only"
    if "multimodal" in model.lower(): cat = "Multimodal SOTA"
    elif "tabular" in model.lower() or "xgboost" in model.lower() or "svm" in model.lower(): cat = "Baseline Tabular"
    elif "xphonebert" in model.lower() or "phonetic" in model.lower(): cat = "Phonetic"
    
    return cat, model, task

def main():
    print("🔎 Analisi globale dei file di predizione...")
    
    found_files = []
    for d in SEARCH_DIRS:
        if d.exists():
            found_files.extend(list(d.rglob("*.csv")))
    
    # Filtriamo solo i file di predizione validi
    valid_files = [f for f in found_files if ("preds" in f.name or "results" in f.name) and "summary" not in f.name]
    
    if not valid_files:
        print("❌ Nessun file CSV di predizione trovato. Controlla le cartelle results_csv o outputs.")
        return

    all_task_rows = []

    for f in valid_files:
        try:
            df = pd.read_csv(f)
            # Normalizzazione colonne
            cols_map = {
                'target': 'Label', 'Target': 'Label', 'label': 'Label',
                'id': 'ID', 'subject_id': 'ID', 'Subject_ID': 'ID',
                'Prob': 'Prob', 'Ensemble_Prob': 'Prob', 'prob': 'Prob',
                'Pred': 'Pred', 'Ensemble_Pred': 'Pred', 'pred': 'Pred'
            }
            df = df.rename(columns=cols_map)

            if 'Label' not in df.columns: continue
            
            # Gestione colonna predizione
            if 'Prob' in df.columns:
                df['Final_Pred'] = (df['Prob'] >= 0.5).astype(int)
            elif 'Pred' in df.columns:
                df['Final_Pred'] = df['Pred'].astype(int)
            else: continue

            cat, model, task = get_model_and_task(f)

            # Analisi sui Fold (se presenti)
            if 'kfold' in df.columns:
                fold_metrics = []
                for fold in sorted(df['kfold'].unique()):
                    sub = df[df['kfold'] == fold]
                    fold_metrics.append(calculate_metrics_raw(sub['Label'], sub['Final_Pred']))
                
                f_df = pd.DataFrame(fold_metrics, columns=['acc', 'f1', 'sens', 'spec'])
                row = {
                    "Category": cat, "Model": model, "Task": task,
                    "Acc": f_df['acc'].mean(), "Acc_std": f_df['acc'].std(),
                    "F1": f_df['f1'].mean(), "F1_std": f_df['f1'].std(),
                    "Sens": f_df['sens'].mean(), "Sens_std": f_df['sens'].std(),
                    "raw_df": df
                }
            else:
                acc, f1, sens, spec = calculate_metrics_raw(df['Label'], df['Final_Pred'])
                row = {
                    "Category": cat, "Model": model, "Task": task,
                    "Acc": acc, "Acc_std": 0.0, "F1": f1, "F1_std": 0.0,
                    "Sens": sens, "Sens_std": 0.0, "raw_df": df
                }
            all_task_rows.append(row)
        except Exception as e:
            continue

    if not all_task_rows:
        print("❌ Nessun dato estratto dai file. Verifica il formato dei CSV.")
        return

    res_df = pd.DataFrame(all_task_rows)

    # --- REPORT 1: DETTAGLIO PER OGNI TASK (5 FOLD STD) ---
    r1 = res_df.copy()
    r1['Accuracy'] = r1.apply(lambda x: format_mean_std(x['Acc'], x['Acc_std']), axis=1)
    r1['F1-Score'] = r1.apply(lambda x: format_mean_std(x['F1'], x['F1_std']), axis=1)
    r1['Sensitivity'] = r1.apply(lambda x: format_mean_std(x['Sens'], x['Sens_std']), axis=1)
    r1[['Category', 'Model', 'Task', 'Accuracy', 'F1-Score', 'Sensitivity']].to_csv(OUTPUT_DIR / "1_all_tasks_detail.csv", index=False)

    # --- REPORT 2: MIGLIOR TASK PER MODELLO ---
    r2_list = []
    for (cat, model), group in res_df.groupby(['Category', 'Model']):
        best = group.loc[group['F1'].idxmax()]
        r2_list.append({
            "Category": cat, "Model": model, "Best Task": best['Task'],
            "Accuracy": format_mean_std(best['Acc'], best['Acc_std']),
            "F1-Score": format_mean_std(best['F1'], best['F1_std']),
            "Sensitivity": format_mean_std(best['Sens'], best['Sens_std'])
        })
    pd.DataFrame(r2_list).to_csv(OUTPUT_DIR / "2_best_task_per_model.csv", index=False)

    # --- REPORT 3: MEDIA DEI TASK (Varianza tra i task) ---
    r3_list = []
    for (cat, model), group in res_df.groupby(['Category', 'Model']):
        if len(group) < 2: continue
        r3_list.append({
            "Category": cat, "Model": model,
            "Avg Accuracy": format_mean_std(group['Acc'].mean(), group['Acc'].std()),
            "Avg F1-Score": format_mean_std(group['F1'].mean(), group['F1'].std()),
            "N_Tasks": len(group)
        })
    pd.DataFrame(r3_list).to_csv(OUTPUT_DIR / "3_average_across_tasks.csv", index=False)

    # --- REPORT 4: PATIENT-LEVEL OR VOTING (Sicily Italy) ---
    r4_list = []
    for (cat, model), group in res_df.groupby(['Category', 'Model']):
        if len(group) < 2: continue # Serve più di un task per votare
        
        full_df = pd.concat(group['raw_df'].tolist())
        full_df['Patient_ID'] = full_df['ID'].astype(str).apply(lambda x: x.split('_Task_')[0])
        
        # Logica Clinica OR: AD se almeno un task è positivo
        patient_logic = full_df.groupby('Patient_ID').agg({
            'Label': 'first',
            'Final_Pred': lambda x: 1 if any(p == 1 for p in x) else 0
        })
        
        acc, f1, sens, spec = calculate_metrics_raw(patient_logic['Label'], patient_logic['Final_Pred'])
        r4_list.append({
            "Category": cat, "Model": model,
            "Accuracy": f"{acc:.3f}", "F1-Score": f"{f1:.3f}", 
            "Sensitivity": f"{sens:.3f}", "Specificity": f"{spec:.3f}"
        })
    pd.DataFrame(r4_list).to_csv(OUTPUT_DIR / "4_patient_or_voting.csv", index=False)

    print("\n" + "="*80)
    print(f"✅ REPORT GENERATI CON SUCCESSO IN: {OUTPUT_DIR}")
    print("="*80)
    if r2_list:
        print("\nTOP PERFORMANCE (Best Task per Modello):")
        print(pd.DataFrame(r2_list)[['Model', 'Best Task', 'F1-Score']].to_string(index=False))

if __name__ == "__main__":
    main()