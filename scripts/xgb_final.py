import sys
import torch
import pandas as pd
import numpy as np
import xgboost as xgb
import multiprocessing
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from transformers import AutoTokenizer

# --- CONFIGURAZIONE ---
CHECKPOINT_PATH = "/home/speechlab/Desktop/repo_anto_thesis/outputs/combined_experiments/COMBINED_Task_01-Task_02_prova2/best_model.pt"
FOLDS_PATH = "/home/speechlab/Desktop/repo_anto_thesis/data/metadata/dataset_folds.csv"
RAW_METADATA_PATH = "data/metadata/speech_metadata.csv"
CONFIG_PATH = "config.yaml"

# Parametri ottimizzati (da tua GridSearch)
BEST_PARAMS = {
    'colsample_bytree': 1.0, 
    'gamma': 0.1, 
    'learning_rate': 0.01, 
    'max_depth': 4, 
    'n_estimators': 50, 
    'subsample': 0.9,
    'objective': 'multi:softmax',
    'num_class': 3,
    'random_state': 42,
    'n_jobs': multiprocessing.cpu_count() - 1
}

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal

def load_sota_model(device):
    config = load_config(CONFIG_PATH)
    config.model.text.name = "Musixmatch/umberto-commoncrawl-cased-v1"
    config.modality = "multimodal_cross_attention"
    if not hasattr(config, 'model'): config.model = type('obj', (object,), {})()
    config.model.output_dim = 2
    model = build_model(config).to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config

def get_probs_for_patient(model, pid, df_meta, config, tokenizer, device):
    rows = df_meta[df_meta['Subject_ID'].str.startswith(pid + "_")]
    res = {"T1": 0.5, "T2": 0.5}
    for _, row in rows.iterrows():
        task = "T1" if "Task_01" in row['Subject_ID'] else "T2"
        try:
            ds = MultimodalDataset(pd.DataFrame([row]), config, tokenizer)
            batch = collate_multimodal([ds[0]])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                out = model(batch)
                res[task] = torch.softmax(out, dim=1)[0, 1].item()
        except: continue
    return res

def calculate_spec(cm, class_idx):
    """Calcola la Specificity per una specifica classe in una matrice multi-classe"""
    temp_cm = np.delete(cm, class_idx, axis=0) # Rimuovi riga della classe
    temp_cm = np.delete(temp_cm, class_idx, axis=1) # Rimuovi colonna della classe
    tn = temp_cm.sum()
    fp = cm[:, class_idx].sum() - cm[class_idx, class_idx]
    fn = cm[class_idx, :].sum() - cm[class_idx, class_idx]
    tp = cm[class_idx, class_idx]
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Caricamento Dati
    df_folds = pd.read_csv(FOLDS_PATH)
    df_raw = pd.read_csv(RAW_METADATA_PATH)
    diag_map = {'CTR': 0, 'MCI': 1, 'AD': 2, 'MILD-AD': 2}
    df_folds['target'] = df_folds['Diagnosis'].map(diag_map)

    # 2. Estrazione Feature SOTA
    model, config = load_sota_model(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    
    print(f"🚀 Estrazione probabilità per {len(df_folds)} pazienti...")
    features = []
    for _, row in df_folds.iterrows():
        p = get_probs_for_patient(model, row['Subject_ID'], df_raw, config, tokenizer, device)
        features.append({'T1': p['T1'], 'T2': p['T2'], 'fold': row['kfold'], 'y': row['target']})
    
    df = pd.DataFrame(features)
    X = df[['T1', 'T2']].values
    y = df['y'].values
    folds = df['fold'].values

    # 3. Ciclo di Valutazione Fold-by-Fold
    metrics = {f: {'acc': 0, 'sens': [], 'spec': []} for f in np.unique(folds)}
    
    print("\n🧪 Inizio Valutazione Cross-Validata...")
    
    for f in sorted(np.unique(folds)):
        train_idx, test_idx = folds != f, folds == f
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        clf = xgb.XGBClassifier(**BEST_PARAMS)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
        
        # Accuracy Fold
        metrics[f]['acc'] = accuracy_score(y_test, preds)
        
        # Sensitivity (Recall) e Specificity per ogni classe
        for i in range(3):
            sens = recall_score(y_test, preds, labels=[i], average='macro', zero_division=0)
            spec = calculate_spec(cm, i)
            metrics[f]['sens'].append(sens)
            metrics[f]['spec'].append(spec)

    # 4. AGGREGAZIONE E STAMPA TABELLA FINALE
    accs = [m['acc'] for m in metrics.values()]
    
    print("\n" + "="*70)
    print(f"🏆 RISULTATI FINALI 3-CLASSI (Media ± Dev.Std)")
    print("="*70)
    print(f"{'METRICA':<20} | {'MEDIA':<10} | {'STD DEV':<10}")
    print("-" * 70)
    print(f"{'Accuracy Totale':<20} | {np.mean(accs)*100:.2f}%     | ± {np.std(accs)*100:.2f}%")
    print("-" * 70)

    class_names = ['Sani (CTR)', 'Prodromici (MCI)', 'Alzheimer (AD)']
    for i, name in enumerate(class_names):
        senses = [m['sens'][i] for m in metrics.values()]
        specs = [m['spec'][i] for m in metrics.values()]
        print(f"{name + ' Sens.':<20} | {np.mean(senses)*100:.2f}%     | ± {np.std(senses)*100:.2f}%")
        print(f"{name + ' Spec.':<20} | {np.mean(specs)*100:.2f}%     | ± {np.std(specs)*100:.2f}%")
        print("-" * 70)

if __name__ == "__main__":
    main()