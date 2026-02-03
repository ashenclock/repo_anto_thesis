import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

# Carica i risultati
csv_path = "outputs/mmse_regression_v2/italian_mmse_final.csv"
df = pd.read_csv(csv_path)

print(f"📊 Analisi Statistica Approfondita ({len(df)} soggetti)")
print("-" * 75)

pairs = [('CTR', 'MCI'), ('MCI', 'MILD-AD'), ('CTR', 'MILD-AD')]

def cohen_d(x, y):
    """Calcola l'Effect Size (quanto è grande la differenza)"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

for g1, g2 in pairs:
    s1 = df[df['Diagnosis'] == g1]['Predicted_MMSE'].values
    s2 = df[df['Diagnosis'] == g2]['Predicted_MMSE'].values
    
    if len(s1) == 0 or len(s2) == 0: continue
        
    t_stat, p_val_t = ttest_ind(s1, s2, equal_var=False) # Welch's t-test (più sicuro)
    u_stat, p_val_u = mannwhitneyu(s1, s2)
    d_val = cohen_d(s1, s2)
    
    print(f"\n🔹 CONFRONTO: {g1} vs {g2}")
    print(f"   Mean {g1}: {s1.mean():.4f} (±{s1.std():.4f})")
    print(f"   Mean {g2}: {s2.mean():.4f} (±{s2.std():.4f})")
    print(f"   Delta Assoluto: {abs(s1.mean() - s2.mean()):.4f}")
    print(f"   Cohen's d (Effect Size): {d_val:.4f}  ", end="")
    
    if abs(d_val) < 0.2: print("(Trascurabile)")
    elif abs(d_val) < 0.5: print("(Piccolo)")
    elif abs(d_val) < 0.8: print("(Medio)")
    else: print("(GRANDE 🔥)")

    # Stampa in notazione scientifica
    print(f"   P-Value (T-Test):      {p_val_t:.2e}  ({'Significativo' if p_val_t < 0.05 else 'Non Sig.'})")
    print(f"   P-Value (Mann-Whitney):{p_val_u:.2e}  ({'Significativo' if p_val_u < 0.05 else 'Non Sig.'})")

print("-" * 75)