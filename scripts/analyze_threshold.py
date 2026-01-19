import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Carica i risultati dell'Ensemble
csv_path = "outputs/ensemble_cross_EN_to_IT/ensemble_results.csv"
df = pd.read_csv(csv_path)

y_true = df['Label'].values
y_probs = df['Ensemble_Prob'].values

print(f"📊 ANALISI SOGLIA OTTIMALE (Su {len(df)} pazienti)")
print(f"{'Threshold':<10} | {'Sens (Recall)':<15} | {'Spec':<10} | {'Acc':<10} | {'F1':<10}")
print("-" * 65)

best_f1 = 0
best_thr = 0

# Proviamo tutte le soglie da 0.20 a 0.60
for thr in np.arange(0.20, 0.61, 0.05):
    y_pred = (y_probs >= thr).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    
    sens = tp / (tp + fn) if (tp+fn)>0 else 0
    spec = tn / (tn + fp) if (tn+fp)>0 else 0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
        
    print(f"{thr:.2f}       | {sens:.4f}          | {spec:.4f}     | {acc:.4f}     | {f1:.4f}")

print("-" * 65)
print(f"💡 CONSIGLIO: Usa Threshold = {best_thr:.2f}")