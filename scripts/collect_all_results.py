import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
import sys
import warnings

warnings.filterwarnings("ignore")

SEARCH_PATHS = [
    Path("results_csv"),
    Path("results_csv_combined"),
    Path("outputs") 
]

def get_asr_info(file_path):
    path_str = str(file_path).lower()
    if "parakeet" in path_str: return " (ASR: Parakeet)"
    if "crisper" in path_str: return " (ASR: Crisper)"
    if "whisper" in path_str and "large" in path_str: return " (ASR: Whisper Large v3)"
    if "whisperx" in path_str: return " (ASR: WhisperX)"
    return ""

def prettify_model_name(raw_name, category):
    name = raw_name.lower()
    if category == "Baseline Tabular":
        clf = "Unknown"
        if "xgboost" in name: clf = "XGBoost"
        elif "svm" in name: clf = "SVM"
        elif "lr" in name or "logistic" in name: clf = "LogReg"
        feats = "Unknown"
        if "egemaps" in name: feats = "eGeMAPS"
        elif "compare" in name: feats = "ComParE"
        elif "whisper" in name: feats = "WhisperEmb"
        return f"{clf} [{feats}]"
    if category == "Baseline Text-Only":
        if "umberto" in name: return "UmBERTo"
        if "bert-base-italian" in name: return "BERT-IT"
        if "xphonebert" in name: return "XPhoneBERT"
        return raw_name.replace("text_", "")
    if "cross" in name or "ensemble" in name:
        if "it_to_en" in name: return "Cross: IT->EN"
        if "en_to_it" in name: return "Cross: EN->IT"
    if "combined" in name: return "Combined (IT)"
    return raw_name

def calculate_metrics_raw(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except: sens, spec = 0.0, 0.0
    return acc, f1, prec, sens, spec

def get_patient_metrics(df, model_name, asr_tag):
    if 'ID' not in df.columns: return None
    df['Patient_ID'] = df['ID'].astype(str).apply(lambda x: x.split('_Task_')[0])
    grouped = df.groupby('Patient_ID').agg({'Label': 'first', 'Prob': list}).reset_index()
    y_true = grouped['Label'].values.astype(int)
    preds = [1 if any(p >= 0.5 for p in probs) else 0 for probs in grouped['Prob']]
    acc, f1, prec, sens, spec = calculate_metrics_raw(y_true, preds)
    return {
        "Group": "SOTA (Combined)",
        "Model": f"{model_name}{asr_tag} [Patient OR Voting]",
        "Task": "All Tasks",
        "Accuracy": acc, "F1-Score": f1, "Precision": prec, "Sensitivity": sens, "Specificity": spec, "Samples": len(y_true)
    }

def extract_info_from_path(file_path):
    filename = file_path.stem
    path_parts = file_path.parts
    task = "N/A"
    if "_Task_" in filename: task = f"Task_{filename.split('_Task_')[1]}"
    else:
        for part in path_parts:
            if "Task_" in part: task = part; break
    
    relevant_part = ""
    if "outputs" in path_parts:
        idx = path_parts.index("outputs")
        if len(path_parts) > idx + 1: relevant_part = path_parts[idx+1]
    elif "results_csv" in path_parts: relevant_part = filename.replace("preds_", "").replace(f"_{task}", "")
    elif "results_csv_combined" in path_parts: relevant_part = filename.replace("preds_", "")
    
    model_name = relevant_part if relevant_part else filename
    lower_name = model_name.lower()
    
    if "combined" in lower_name: category = "SOTA (Combined)"
    elif "tabular" in lower_name: category = "Baseline Tabular"
    elif "text" in lower_name and "multimodal" not in lower_name: category = "Baseline Text-Only"
    elif "cross" in lower_name or "ensemble" in lower_name: category = "SOTA (Cross-Dataset)"
    elif "multimodal" in lower_name: category = "Baseline Multimodal"
    elif "fusion" in lower_name: category = "Manual Fusion"
    else: category = "Other"
    
    return category, model_name, task

def assign_macro_category(row):
    """Assegna la riga a una delle 3 tabelle richieste"""
    group = row['Group']
    model = row['Model']
    
    if "Combined" in group or "Fusion" in group:
        return "3_MIXED"
    if "Cross" in group or "Ensemble" in model or "IT->EN" in model or "EN->IT" in model:
        return "2_CROSS_ENGLISH"
    return "1_ITALIAN_BASELINE"

def print_table(df, title, filename):
    if df.empty: return
    
    # Arrotondamento
    cols_num = ["Accuracy", "F1-Score", "Precision", "Sensitivity", "Specificity"]
    df[cols_num] = df[cols_num].round(2)
    
    # Ordinamento
    df = df.sort_values(by=['Accuracy', 'Sensitivity'], ascending=[False, False])
    
    print("\n" + "="*160)
    print(f"📊 {title}")
    print("="*160)
    # Selezione colonne pulite
    print(df[['Model', 'Task', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'Precision']].to_string(index=False))
    
    df.to_csv(filename, index=False)
    print(f"\n💾 Salvato in: {filename}")

def main():
    print(f"🔍 GENERAZIONE REPORT SUDDIVISO...\n")
    all_rows = []
    found_files = []
    for base_dir in SEARCH_PATHS:
        if base_dir.exists(): found_files.extend(list(base_dir.rglob("*.csv")))
    valid_files = [f for f in found_files if "preds" in f.name or "results" in f.name]
    
    for file_path in valid_files:
        try: df = pd.read_csv(file_path)
        except: continue
        cols_map = {'Target': 'Label', 'target': 'Label', 'id': 'ID', 'Subject_ID': 'ID', 'subject_id': 'ID', 'Ensemble_Prob': 'Prob', 'Soft_Prob': 'Prob', 'Hard_Pred': 'Pred', 'Ensemble_Pred': 'Pred'}
        df.rename(columns=cols_map, inplace=True)
        if 'Label' not in df.columns: continue
        prob_col = 'Prob' if 'Prob' in df.columns else None
        pred_col = 'Pred' if 'Pred' in df.columns else None
        if not prob_col and not pred_col: continue
        
        cat, raw_name, task = extract_info_from_path(file_path)
        asr_tag = get_asr_info(file_path)
        pretty_name = prettify_model_name(raw_name, cat)
        if asr_tag.strip().lower() in pretty_name.lower(): asr_tag = ""

        # Logic for Combined OR Voting
        if cat == "SOTA (Combined)" and prob_col and 'ID' in df.columns:
            patient_row = get_patient_metrics(df, pretty_name, asr_tag)
            if patient_row: all_rows.append(patient_row)
            
        # Logic for Cross
        y_true = df['Label'].values
        if cat == "SOTA (Cross-Dataset)" and prob_col:
            thr = 0.35 if "en_to_it" in raw_name.lower() else 0.5
            y_pred = (df[prob_col].values >= thr).astype(int)
            if thr != 0.5: pretty_name += " (Tuned)"
        else:
            y_pred = (df[prob_col].values >= 0.5).astype(int) if prob_col else df[pred_col].values
            
        acc, f1, prec, sens, spec = calculate_metrics_raw(y_true, y_pred)
        all_rows.append({"Group": cat, "Model": f"{pretty_name}{asr_tag}", "Task": "Cross-Eval" if "Cross" in cat else task, 
                         "Accuracy": acc, "F1-Score": f1, "Precision": prec, "Sensitivity": sens, "Specificity": spec, "Samples": len(y_true)})

    # --- AGGREGAZIONE ---
    df_all = pd.DataFrame(all_rows)
    if df_all.empty: print("❌ Nessun dato trovato."); return

    final_report = []
    
    for (group, model), sub_df in df_all.groupby(['Group', 'Model']):
        if "SOTA" in group or "Combined" in model or "Cross" in group:
            final_report.append(sub_df.iloc[0].to_dict())
            continue
            
        best_row = sub_df.loc[sub_df['Accuracy'].idxmax()]
        final_report.append({"Group": group, "Model": model, "Task": f"BEST ({best_row['Task']})", 
                             "Accuracy": best_row['Accuracy'], "F1-Score": best_row['F1-Score'], "Precision": best_row['Precision'],
                             "Sensitivity": best_row['Sensitivity'], "Specificity": best_row['Specificity'], "Samples": best_row['Samples']})
        
        if len(sub_df) >= 7:
            final_report.append({"Group": group, "Model": model, "Task": f"AVG ({len(sub_df)} Tasks)", 
                                 "Accuracy": sub_df['Accuracy'].mean(), "F1-Score": sub_df['F1-Score'].mean(), "Precision": sub_df['Precision'].mean(),
                                 "Sensitivity": sub_df['Sensitivity'].mean(), "Specificity": sub_df['Specificity'].mean(), "Samples": sub_df['Samples'].iloc[0]})

    res_df = pd.DataFrame(final_report)
    
    # --- SUDDIVISIONE IN 3 TABELLE ---
    res_df['Macro_Cat'] = res_df.apply(assign_macro_category, axis=1)
    
    it_df = res_df[res_df['Macro_Cat'] == "1_ITALIAN_BASELINE"]
    cross_df = res_df[res_df['Macro_Cat'] == "2_CROSS_ENGLISH"]
    mixed_df = res_df[res_df['Macro_Cat'] == "3_MIXED"]
    
    # Stampe
    print_table(mixed_df, "3. MIXED & COMBINED (Data Fusion - Sicily)", "REPORT_MIXED.csv")
    print_table(cross_df, "2. CROSS-DATASET & ENGLISH (Generalization)", "REPORT_CROSS.csv")
    print_table(it_df, "1. ITALIAN DATASET (Sicily - Baselines)", "REPORT_ITALIAN.csv")

if __name__ == "__main__":
    main()