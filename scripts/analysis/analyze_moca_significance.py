import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

# Percorso dei risultati MoCA
csv_path = "outputs/moca_regression_final/italian_moca_final.csv"

def cohen_d(x, y):
    """Calcola l'Effect Size (quanto è grande la separazione)"""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    std_pooled = np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / std_pooled

def main():
    if not pd.io.common.file_exists(csv_path):
        print(f"❌ Errore: File non trovato in {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"📊 Analisi Statistica MoCA (Regressione Zero-Shot)")
    print(f"Dataset: Sicily (IT) | Campioni: {len(df)}")
    print("-" * 80)

    pairs = [('CTR', 'MCI'), ('MCI', 'MILD-AD'), ('CTR', 'MILD-AD')]

    for g1, g2 in pairs:
        s1 = df[df['Diagnosis'] == g1]['Predicted_MoCA'].values
        s2 = df[df['Diagnosis'] == g2]['Predicted_MoCA'].values
        
        if len(s1) == 0 or len(s2) == 0: continue
            
        # Welch's t-test (non assume varianza uguale)
        t_stat, p_val_t = ttest_ind(s1, s2, equal_var=False)
        # Mann-Whitney U (robusto ai non-parametrici)
        u_stat, p_val_u = mannwhitneyu(s1, s2)
        # Effect size
        d_val = cohen_d(s1, s2)
        
        print(f"\n🔹 CONFRONTO: {g1} vs {g2}")
        print(f"   Media {g1}: {s1.mean():.4f} (±{s1.std():.4f})")
        print(f"   Media {g2}: {s2.mean():.4f} (±{s2.std():.4f})")
        print(f"   Delta: {abs(s1.mean() - s2.mean()):.4f}")
        print(f"   Cohen's d: {d_val:.4f} ", end="")
        
        if abs(d_val) < 0.2: print("(Trascurabile)")
        elif abs(d_val) < 0.5: print("(Piccolo)")
        elif abs(d_val) < 0.8: print("(Medio)")
        else: print("(GRANDE 🔥)")

        print(f"   P-Value (T-Test):      {p_val_t:.2e} ({'✅ SIG' if p_val_t < 0.05 else '❌ n.s.'})")
        print(f"   P-Value (Mann-Whitney):{p_val_u:.2e} ({'✅ SIG' if p_val_u < 0.05 else '❌ n.s.'})")

    print("-" * 80)

if __name__ == "__main__":
    main()