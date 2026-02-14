import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# --- CONFIGURAZIONE ---
INPUT_CSV = "data/metadata/italy_clinical_complete.csv"
OUT_DIR = Path("outputs/clinical_xgboost_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colonne da ESCLUDERE (Target o Leakage diretto)
DROP_COLS = [
    'Diagnosis', 'NOME', 'COGNOME', 'CELLULARE', 'DATA ARRUOLAMENTO', 'ARRUOLATO DA',
    'Unnamed: 0', 'Subject_ID', 
    'MMSE', 'MOCA', 'CDR', 'GDS', 'HACHINSKI', 'ADL', 'IADL', 'TC', 'RMN' # Leakage: questi definiscono la diagnosi
]

def clean_data(df):
    """Pulisce i dati e converte in numerico"""
    cols = [c for c in df.columns if c not in DROP_COLS and df[c].dtype == 'object']
    for c in cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.').str.replace('N.A.', '').str.replace('N.C.', ''), errors='coerce')
    return df

def main():
    print("🚀 AVVIO XGBOOST CLINICO (Grid Search)...")
    
    # 1. Caricamento e Pulizia
    df = pd.read_csv(INPUT_CSV)
    df = clean_data(df)
    
    # 2. Selezione Features
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    # Rimuoviamo colonne vuote
    df_clean = df.dropna(subset=['Diagnosis'])
    
    # Teniamo solo colonne con almeno il 30% di dati reali
    valid_features = []
    for col in feature_cols:
        if df_clean[col].count() > len(df_clean) * 0.3:
            valid_features.append(col)
            
    print(f"✅ Features utilizzate ({len(valid_features)}): {valid_features}")
    
    X = df_clean[valid_features]
    y_raw = df_clean['Diagnosis']
    
    # Encoding Label (CTR=0, MCI=1, AD=2)
    le = LabelEncoder()
    # Forziamo l'ordine per avere CTR=0, MCI=1, MILD-AD=2 se possibile, altrimenti lo fa in ordine alfabetico
    # Facciamo map manuale per sicurezza
    label_map = {'CTR': 0, 'MCI': 1, 'MILD-AD': 2}
    # Filtriamo solo le righe che hanno una di queste label
    mask = y_raw.isin(label_map.keys())
    X = X[mask]
    y = y_raw[mask].map(label_map).astype(int)
    
    print(f"   Campioni totali: {len(X)}")

    # 3. XGBoost Grid Search
    xgb = XGBClassifier(
        objective='multi:softmax', 
        num_class=3, 
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Griglia Iperparametri
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.9],         # Evita overfitting
        'colsample_bytree': [0.7, 0.9]   # Evita overfitting sulle feature
    }
    
    print("\n🔍 Inizio Grid Search (5-Fold)...")
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid.fit(X, y)
    
    best_model = grid.best_estimator_
    print(f"\n🏆 Migliori Parametri: {grid.best_params_}")
    print(f"🏆 Best CV F1-Score: {grid.best_score_:.4f}")

    # 4. Validazione Finale (Stratified 5-Fold con modello ottimizzato)
    # Rifacciamo la CV per avere la std dev precisa e la confusion matrix accumulata
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    acc_scores, f1_scores = [], []
    all_y_true, all_y_pred = [], []
    
    feature_importances = np.zeros(len(valid_features))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Clone del best model
        clf = XGBClassifier(**grid.best_params_, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_val)
        
        acc_scores.append(accuracy_score(y_val, preds))
        f1_scores.append(f1_score(y_val, preds, average='weighted'))
        
        all_y_true.extend(y_val)
        all_y_pred.extend(preds)
        
        feature_importances += clf.feature_importances_

    # 5. Risultati Finali
    print("\n" + "="*50)
    print("📊 RISULTATI FINALI (XGBoost Optimized)")
    print(f"   Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"   F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print("="*50)

    # 6. Matrice di Confusione
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['CTR', 'MCI', 'AD'], yticklabels=['CTR', 'MCI', 'AD'])
    plt.title(f"Confusion Matrix (XGBoost)\nAcc: {np.mean(acc_scores):.2f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(OUT_DIR / "confusion_matrix_xgboost.png")
    
    # 7. Feature Importance
    feature_importances /= 5 # Media
    feat_df = pd.DataFrame({'Feature': valid_features, 'Importance': feature_importances})
    feat_df = feat_df.sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette="magma")
    plt.title("XGBoost Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "xgboost_feature_importance.png")
    
    feat_df.to_csv(OUT_DIR / "top_features.csv", index=False)
    print(f"✅ Grafici salvati in: {OUT_DIR}")

if __name__ == "__main__":
    main()