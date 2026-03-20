import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
import warnings

warnings.filterwarnings("ignore")

# --- PERCORSI DEI FILE SPECIFICI ---
FILES = {
    # Phase 4: Cross-Lingual
    "IT_to_EN": Path("outputs/cross_dataset_IT_to_EN/cross_dataset_results.csv"),
    "EN_to_IT": Path("outputs/cross_dataset_EN_to_IT/cross_dataset_results.csv"),
    
    # Phase 4b: Ensemble Cross (se esiste)
    "EN_to_IT_Ensemble": Path("outputs/ensemble_cross_EN_to_IT/ensemble_results.csv"),
    
    # Phase 6: Mixed/Multilingual Training
    "Mixed_Eval_IT": Path("outputs/balanced_mix_evaluation/predictions_ITALIAN_TEST_SET.csv"),
    "Mixed_Eval_EN": Path("outputs/balanced_mix_evaluation/predictions_ENGLISH_TEST_SET.csv")
}

def calculate_metrics(y_true, y_pred):
    if len(np.unique(y_true)) < 1: return None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {'Acc': acc, 'F1': f1, 'Sens': sens, 'Spec': spec, 'Prec': prec}

def analyze_file(path, label_col, prob_col, threshold=0.5):
    if not path.exists():
        return None
    
    df = pd.read_csv(path)
    
    # Normalizza colonne
    cols = {c.lower(): c for c in df.columns}
    
    # Trova colonna Label
    lbl = cols.get(label_col.lower())
    if not lbl: 
        # Fallback
        if 'target' in cols: lbl = cols['target']
        elif 'diagnosis' in cols: lbl = cols['diagnosis']
        else: return None
        
    # Trova colonna Probabilità
    prb = cols.get(prob_col.lower())
    if not prb:
        # Fallback
        if 'ensemble_prob' in cols: prb = cols['ensemble_prob']
        elif 'prob_ad' in cols: prb = cols['prob_ad']
        else: return None

    y_true = df[lbl].astype(int)
    y_pred = (df[prb] >= threshold).astype(int)
    
    return calculate_metrics(y_true, y_pred)

def main():
    print("🌍 GENERAZIONE TABELLE CROSS-LINGUAL & MIXED DATASET")
    print("="*80)

    # --- TABELLA 6.5: CROSS-DATASET GENERALIZATION ---
    cross_results = []
    
    # 1. IT -> EN (Zero-Shot)
    res = analyze_file(FILES["IT_to_EN"], label_col="Label", prob_col="Prob")
    if res:
        cross_results.append({
            "Direction": "IT -> EN", "Strategy": "Cross (Zero-Shot)", "Thr": 0.5, **res
        })
        
    # 2. EN -> IT (Standard)
    res = analyze_file(FILES["EN_to_IT"], label_col="Label", prob_col="Prob")
    if res:
        cross_results.append({
            "Direction": "EN -> IT", "Strategy": "Cross (Standard)", "Thr": 0.5, **res
        })
        # 3. EN -> IT (Tuned Threshold per Triage)
        res_tuned = analyze_file(FILES["EN_to_IT"], label_col="Label", prob_col="Prob", threshold=0.35)
        if res_tuned:
            cross_results.append({
                "Direction": "EN -> IT", "Strategy": "Cross (Tuned Thr 0.35)", "Thr": 0.35, **res_tuned
            })

    # 4. EN -> IT (Ensemble 10-Fold) - Se disponibile
    res = analyze_file(FILES["EN_to_IT_Ensemble"], label_col="Label", prob_col="Ensemble_Prob")
    if res:
        cross_results.append({
            "Direction": "EN -> IT", "Strategy": "Ensemble Multimodal ALL", "Thr": 0.5, **res
        })

    print("\n📊 TABELLA 6.5: CROSS-DATASET GENERALIZATION")
    df_cross = pd.DataFrame(cross_results)
    if not df_cross.empty:
        # Formatta colonne
        for c in ['Acc', 'F1', 'Sens', 'Spec']:
            df_cross[c] = df_cross[c].apply(lambda x: f"{x:.2f}")
        print(df_cross[['Direction', 'Strategy', 'Acc', 'Sens', 'Spec', 'F1']].to_string(index=False))
        df_cross.to_csv("TABELLA_6_5_CROSS_LINGUAL.csv", index=False)
    else:
        print("❌ Nessun risultato Cross-Dataset trovato.")


    # --- TABELLA 6.6: MIXED DATASET (MULTILINGUAL) ---
    mix_results = []
    
    # 1. Mixed Model su IT
    res = analyze_file(FILES["Mixed_Eval_IT"], label_col="Label", prob_col="Prob_AD")
    if res:
        mix_results.append({
            "Test Set": "Italian (Target)", "Strategy": "Mixed Training (Weighted)", **res
        })
        
    # 2. Mixed Model su EN
    res = analyze_file(FILES["Mixed_Eval_EN"], label_col="Label", prob_col="Prob_AD")
    if res:
        mix_results.append({
            "Test Set": "English (Source)", "Strategy": "Mixed Training (Weighted)", **res
        })

    print("\n\n📊 TABELLA 6.6: MULTILINGUAL TRAINING (MIXED DATASET)")
    df_mix = pd.DataFrame(mix_results)
    if not df_mix.empty:
        for c in ['Acc', 'F1', 'Sens', 'Spec']:
            df_mix[c] = df_mix[c].apply(lambda x: f"{x:.2f}")
        print(df_mix[['Test Set', 'Strategy', 'Acc', 'Sens', 'Spec', 'F1']].to_string(index=False))
        df_mix.to_csv("TABELLA_6_6_MIXED_TRAINING.csv", index=False)
    else:
        print("❌ Nessun risultato Mixed Dataset trovato (controlla outputs/balanced_mix_evaluation).")

    print("\n✅ File salvati: TABELLA_6_5_CROSS_LINGUAL.csv, TABELLA_6_6_MIXED_TRAINING.csv")

if __name__ == "__main__":
    main()