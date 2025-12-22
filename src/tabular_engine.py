import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
class TabularTrainer:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_set = config.feature_extraction.feature_set

    def load_data(self):
        """
        Carica le feature specifiche del TASK e le unisce ai fold.
        """
        # 1. Determina il nome del file
        # Se c'è un target_task nel config (es. "Task_01"), usalo nel nome file
        if hasattr(self.config.data, 'target_task') and self.config.data.target_task:
            task_suffix = f"_{self.config.data.target_task}"
        else:
            task_suffix = "" # Fallback per file generici
            
        fname = f"train_{self.feature_set}{task_suffix}.csv"
        feat_path = Path(self.config.data.features_root) / fname
        
        if not feat_path.exists():
            raise FileNotFoundError(f"File feature mancante: {feat_path}")
        
        print(f"   -> Caricamento feature da: {fname}")
        df_feats = pd.read_csv(feat_path)
        df_feats['ID'] = df_feats['ID'].astype(str)

        # 2. Carica Folds
        folds_path = Path(self.config.data.folds_file)
        df_folds = pd.read_csv(folds_path)
        if 'Subject_ID' in df_folds.columns:
            df_folds = df_folds.rename(columns={'Subject_ID': 'ID'})
        df_folds['ID'] = df_folds['ID'].astype(str)
        
        # 3. Merge
        df_merged = pd.merge(df_feats, df_folds[['ID', 'Diagnosis', 'kfold']], on='ID', how='inner')
        
        # 4. Mappa le etichette
        if hasattr(self.config.labels.mapping, 'to_dict'):
            mapping = self.config.labels.mapping.to_dict()
        else:
            mapping = self.config.labels.mapping
            
        df_merged['target'] = df_merged['Diagnosis'].map(mapping)
        df_merged = df_merged.dropna(subset=['target'])
        df_merged['target'] = df_merged['target'].astype(int)

        return df_merged

    def get_model_pipeline(self):
        model_name = self.config.tabular_model.name
        pca = PCA(n_components=0.95, random_state=self.config.seed)
        if model_name == 'svm':
            clf = SVC(probability=True, class_weight='balanced', random_state=self.config.seed)
        elif model_name == 'xgboost':
            clf = XGBClassifier(
                objective='binary:logistic', 
                use_label_encoder=True, 
                eval_metric='logloss',
                n_jobs=1, # Importante per evitare conflitti con joblib
                random_state=self.config.seed
            )
        elif model_name == 'lr':
            clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.config.seed)
        else:
            raise ValueError(f"Modello {model_name} non supportato")

        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])

    def train(self):
        # Ritorna lo score medio per il log
        
        df = self.load_data()
        
        feature_cols = [c for c in df.columns if c not in ['ID', 'Diagnosis', 'kfold', 'target']]
        X = df[feature_cols].values
        y = df['target'].values
        ids = df['ID'].values
        folds = df['kfold'].values
        
        oof_preds = []
        oof_ids = []
        oof_targets = []
        
        unique_folds = sorted(np.unique(folds))
        
        # GridSearch params
        model_name = self.config.tabular_model.name
        grid_dict = getattr(self.config.tabular_model.grids, model_name).to_dict()
        param_grid = {f'clf__{k}': v for k, v in grid_dict.items()}

        print(f"   -> Inizio 5-Fold CV con GridSearch interna...")
        
        for fold in unique_folds:
            train_mask = (folds != fold)
            val_mask = (folds == fold)
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            ids_val = ids[val_mask]
            
            # Grid Search interna
            pipeline = self.get_model_pipeline()
            cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.seed)
            grid = GridSearchCV(pipeline, param_grid, cv=cv_inner, scoring='f1', n_jobs=-1, verbose=0)
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            
            # Predict
            if hasattr(best_model, "predict_proba"):
                probs = best_model.predict_proba(X_val)[:, 1]
            else:
                probs = best_model.predict(X_val)
                
            oof_preds.extend(probs)
            oof_ids.extend(ids_val)
            oof_targets.extend(y_val)
            
            # Salva modello
            joblib.dump(best_model, self.output_dir / f"model_fold_{fold}.pkl")

        # Metriche Finali
        oof_preds = np.array(oof_preds)
        oof_targets = np.array(oof_targets)
        oof_labels = (oof_preds >= 0.5).astype(int)
        
        acc = accuracy_score(oof_targets, oof_labels)
        f1 = f1_score(oof_targets, oof_labels, average='weighted')
        
        # Salva predizioni
        df_results = pd.DataFrame({'ID': oof_ids, 'Label': oof_targets, 'Prob': oof_preds})
        
        # Nome file coerente per la fusione: preds_Task_XX.csv
        task_name = self.config.data.target_task if hasattr(self.config.data, 'target_task') else "Global"
        out_path = self.output_dir.parent / f"preds_{task_name}.csv" # Salva nella cartella superiore (results_csv_tabular)
        df_results.to_csv(out_path, index=False)
        
        return acc, f1