import pandas as pd
import numpy as np
import torch
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

# CONFIGURAZIONE
METADATA_FILE = "data/metadata/mmse_experiment_metadata.csv" 
CLINICAL_FILE = "data/metadata/dataset_metadata.csv"
TRANSCRIPTS_ROOT = Path("data/transcripts/Parakeet_Unified/Task_01")
AUDIO_FEATURES_ROOT = Path("data/features") 

def clean_education(level):
    level = str(level).lower()
    if 'laurea' in level: return 18
    if 'superiore' in level or 'diploma' in level: return 13
    if 'media' in level or 'secondaria' in level: return 8
    return 5

def load_data_complete():
    print("🔄 Caricamento e Fusione Dati...")
    df = pd.read_csv(METADATA_FILE)
    df = df[df['Language'] == 'IT']
    df = df[df['Subject_ID'].str.contains("Task_01")].reset_index(drop=True)
    
    # 1. Clinica
    df_clinic = pd.read_csv(CLINICAL_FILE)
    df_clinic['Clean_ID'] = df_clinic['Subject_ID'].apply(lambda x: x.split('_Task_')[0])
    df['Clean_ID'] = df['Subject_ID'].apply(lambda x: x.split('_Task_')[0])
    df = pd.merge(df, df_clinic[['Clean_ID', 'Age', 'Gender', 'Education']], on='Clean_ID', how='left')
    
    X_age = df['Age'].fillna(70).values
    X_edu = df['Education'].apply(clean_education).values
    X_sex = df['Gender'].map({'Female': 0, 'Male': 1}).fillna(0).values
    X_clinic = np.vstack([X_age, X_edu, X_sex]).T

    # 2. Testo
    texts = []
    for _, row in df.iterrows():
        fname = f"{row['Clean_ID']}.txt"
        path = TRANSCRIPTS_ROOT / fname
        texts.append(path.read_text(encoding='utf-8').strip() if path.exists() else "")
    
    tfidf = TfidfVectorizer(max_features=32, ngram_range=(1,2))
    X_text = tfidf.fit_transform(texts).toarray()

    # 3. Audio (Wav2Vec2 vectors)
    # Cerchiamo la cartella in modo dinamico
    import glob
    search_path = list(AUDIO_FEATURES_ROOT.glob("*wav2vec2-large-xlsr-53*_sequences/Task_01"))
    real_path = search_path[0] if search_path else None
    
    audio_feats = []
    valid_idxs = []
    
    if real_path:
        for idx, row in df.iterrows():
            fpath = real_path / f"{row['Clean_ID']}.pt"
            if fpath.exists():
                t = torch.load(fpath, weights_only=True).float()
                # Mean + Std pooling
                if t.dim() > 1:
                    mu = torch.mean(t, dim=0).numpy()
                    sigma = torch.std(t, dim=0).numpy()
                    audio_feats.append(np.concatenate([mu, sigma]))
                else:
                    audio_feats.append(np.concatenate([t.numpy(), np.zeros_like(t.numpy())]))
                valid_idxs.append(idx)
    
    # Allinea tutto
    df = df.iloc[valid_idxs].reset_index(drop=True)
    X_clinic = X_clinic[valid_idxs]
    X_text = X_text[valid_idxs]
    X_audio_raw = np.array(audio_feats)
    
    pca = PCA(n_components=32, random_state=42)
    X_audio = pca.fit_transform(StandardScaler().fit_transform(X_audio_raw))
    
    # Fusione Finale
    X = np.hstack([X_audio, X_text, X_clinic])
    
    # TARGET BINARIO: CTR=0, MCI+AD=1
    y = df['Diagnosis'].map({'CTR': 0, 'MCI': 1, 'MILD-AD': 1}).values
    
    return X, y, df['Diagnosis'].values

def main():
    X, y, y_orig = load_data_complete()
    print(f"✅ Dataset Binario Pronto: {X.shape}")
    print(f"   Sani (0): {sum(y==0)} | Patologici (1): {sum(y==1)}")

    # XGBoost Binario
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(xgb, X, y, cv=skf)
    y_proba = cross_val_predict(xgb, X, y, cv=skf, method='predict_proba')[:, 1]

    print("\n" + "="*60)
    print("📊 SCREENING REPORT (Sani vs MCI/AD)")
    print("="*60)
    print(classification_report(y, y_pred, target_names=['Sani', 'Patologici']))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sani', 'Patologici'], yticklabels=['Sani', 'Patologici'])
    plt.title("Confusion Matrix: Binary Screening")
    plt.savefig("outputs/binary_screening_cm.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    print(f"🏆 AUC Score: {roc_auc:.4f}")
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Screening Alzheimer')
    plt.legend(loc="lower right")
    plt.savefig("outputs/binary_screening_roc.png")
    
    print("\n✅ Grafici salvati in outputs/")

if __name__ == "__main__":
    main()