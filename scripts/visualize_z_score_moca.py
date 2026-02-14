import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/moca_regression_final/italian_moca_final.csv")

# Normalizziamo le predizioni per "zoomare" sulle differenze
df['Z_Score_Pred'] = (df['Predicted_MoCA'] - df['Predicted_MoCA'].mean()) / df['Predicted_MoCA'].std()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Diagnosis', y='Z_Score_Pred', order=['CTR', 'MCI', 'AD'], palette="magma")
plt.title("Z-Score delle Predizioni MoCA (Normalizzazione delle differenze)")
plt.ylabel("Deviazioni Standard dalla Media")
plt.savefig("outputs/moca_regression_final/zscore_plot.png")
print("✅ Grafico Z-Score salvato. Guarda se qui le scatole sono più staccate.")