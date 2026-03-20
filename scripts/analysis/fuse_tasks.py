import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import yaml
import glob
import warnings

# Ignora warning pandas per slicing
warnings.filterwarnings("ignore")

def calculate_full_metrics(y_true, y_pred):
    """Calcola metriche cliniche su un singolo fold."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    # Confusion Matrix
    # Labels fissi [0, 1] per gestire casi limite (es. batch con solo classe 0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
    }

def print_summary(title, means, stds):
    print("\n" + "-"*65)
    print(f"🔹 {title}")
    print("-"*65)
    print(f"{'Metric':<15} | {'Mean':<10} | {'Std Dev':<10}")
    print("-"*65)
    for metric in means.index:
        print(f"{metric.capitalize():<15} | {means[metric]:.4f}     | ± {stds[metric]:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Ensemble dei Task (Soft & Hard Voting) con Metriche CV")
    parser.add_argument("--results_dir", type=str, default="results_csv", help="Cartella contenente i csv delle predizioni")
    parser.add_argument("--pattern", type=str, required=True, help="Pattern del nome file (es: 'preds_multimodal_cross_attention')")
    parser.add_argument("--tasks", type=str, nargs='+', default=[], help="Opzionale: Lista task specifici. Se vuoto, FODE TUTTO quello che trova.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Per recuperare i fold originali")
    args = parser.parse_args()

    # 1. Carica Config e Mappa Fold
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    folds_df = pd.read_csv(config['data']['folds_file'])
    id_col = 'Subject_ID' if 'Subject_ID' in folds_df.columns else 'ID'
    folds_map = folds_df.set_index(id_col)['kfold'].to_dict()

    # 2. Trova i file
    base_path = Path(args.results_dir)
    found_files = []
    
    if args.tasks:
        # Modalità Manuale: Cerca solo i task specificati
        for t in args.tasks:
            matches = list(base_path.glob(f"{args.pattern}*_{t}.csv"))
            if matches: found_files.append(matches[0])
            else: print(f"⚠️ File non trovato per {t}")
    else:
        # Modalità Automatica: Cerca TUTTI i file che matchano il pattern
        # Esclude file che iniziano con 'ensemble_' per evitare ricorsioni
        found_files = [f for f in base_path.glob(f"{args.pattern}*.csv") if "ensemble" not in f.name]

    if len(found_files) < 2:
        print(f"❌ Errore: Trovati solo {len(found_files)} file. Servono almeno 2 file per l'ensemble.")
        print(f"   Pattern cercato: {args.pattern}")
        return

    # Ordina i file per coerenza
    found_files.sort()

    print(f"\n🔗 FUSIONE AVVIATA SU {len(found_files)} TASK:")
    task_names = []
    for f in found_files:
        t_name = f.stem.split('Task_')[-1] if 'Task_' in f.stem else f.stem[-7:]
        task_names.append(t_name)
        print(f"   - {t_name:<10} ({f.name})")

    # 3. Merge dei Dataframe
    merged_df = None
    
    for f in found_files:
        task_label = f.stem.split('Task_')[-1] if 'Task_' in f.stem else f.stem[-2:]
        
        df = pd.read_csv(f)
        df['ID'] = df['ID'].astype(str)
        df = df.set_index('ID')
        
        # Rinomina per il merge
        # Salviamo la Probabilità (per Soft Voting)
        df = df.rename(columns={'Prob': f'Prob_{task_label}', 'Label': 'Target'})
        
        # Calcoliamo subito la Predizione Binaria (per Hard Voting)
        # Assumiamo threshold 0.5
        df[f'Pred_{task_label}'] = (df[f'Prob_{task_label}'] >= 0.5).astype(int)
        
        if merged_df is None:
            merged_df = df[['Target', f'Prob_{task_label}', f'Pred_{task_label}']]
        else:
            merged_df = merged_df.join(df[[f'Prob_{task_label}', f'Pred_{task_label}']], how='inner')

    # 4. Aggiungi info Fold
    merged_df['kfold'] = merged_df.index.map(folds_map)
    merged_df = merged_df.dropna(subset=['kfold'])
    
    print(f"\n   -> Soggetti comuni (Intersection): {len(merged_df)}")

    # ==========================================
    # 5. CALCOLO STRATEGIE DI VOTING
    # ==========================================
    
    # A. SOFT VOTING (Media delle probabilità)
    prob_cols = [c for c in merged_df.columns if 'Prob_' in c]
    merged_df['Soft_Prob'] = merged_df[prob_cols].mean(axis=1)
    merged_df['Soft_Pred'] = (merged_df['Soft_Prob'] >= 0.5).astype(int)

    # B. HARD VOTING (Maggioranza delle classi)
    pred_cols = [c for c in merged_df.columns if 'Pred_' in c]
    # mode(axis=1) ritorna la moda. [0] prende la prima colonna in caso di pareggio.
    # In caso di pareggio (es. 2 vs 2), pandas prende il valore più basso (0, cioè CTR).
    # È un approccio conservativo.
    merged_df['Hard_Pred'] = merged_df[pred_cols].mode(axis=1)[0].astype(int)

    # ==========================================
    # 6. VALUTAZIONE CROSS-VALIDATION
    # ==========================================
    
    soft_fold_results = []
    hard_fold_results = []
    unique_folds = sorted(merged_df['kfold'].unique())

    print("\n📊 CALCOLO METRICHE SUI FOLD...")
    
    for fold in unique_folds:
        fold_data = merged_df[merged_df['kfold'] == fold]
        y_true = fold_data['Target']
        
        # Metriche Soft
        soft_fold_results.append(calculate_full_metrics(y_true, fold_data['Soft_Pred']))
        
        # Metriche Hard
        hard_fold_results.append(calculate_full_metrics(y_true, fold_data['Hard_Pred']))

    # 7. Aggregazione
    soft_df = pd.DataFrame(soft_fold_results)
    hard_df = pd.DataFrame(hard_fold_results)

    print("\n" + "="*65)
    print(f"🏆 RISULTATI ENSEMBLE FINALI ({len(unique_folds)} Folds)")
    print("="*65)
    
    print_summary("SOFT VOTING (Media Probabilità)", soft_df.mean(), soft_df.std())
    print_summary("HARD VOTING (Maggioranza Classi)", hard_df.mean(), hard_df.std())
    print("="*65)

    # 8. Salvataggio CSV
    tasks_suffix = "ALL"
    if args.tasks:
        tasks_suffix = "-".join(args.tasks)
        
    out_path = base_path / f"ensemble_{args.pattern}_{tasks_suffix}.csv"
    
    # Salviamo un CSV pulito con le colonne utili
    final_cols = ['Target', 'kfold', 'Soft_Prob', 'Soft_Pred', 'Hard_Pred'] + prob_cols
    merged_df[final_cols].to_csv(out_path)
    
    print(f"\n💾 File predizioni ensemble salvato in:\n   -> {out_path}")

if __name__ == "__main__":
    main()