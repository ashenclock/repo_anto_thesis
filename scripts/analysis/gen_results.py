import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
import yaml
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score

warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE ---
SEARCH_DIRS = [Path("results_csv"), Path("results_csv_combined"), Path("outputs")]
CONFIG_PATH = "config.yaml"
BEST_MODEL_PATTERN = "COMBINED_Task_01-Task_02_prova2"

def calculate_metrics(y_true, y_pred):
    if len(np.unique(y_true)) < 1: return None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Calcolo metriche
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {'Acc': acc, 'F1': f1, 'Sens': sens, 'Spec': spec, 'Prec': prec}

def load_fold_map():
    try:
        with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)
        df = pd.read_csv(config['data']['folds_file'])
        id_col = 'Subject_ID' if 'Subject_ID' in df.columns else 'ID'
        return df.set_index(id_col)['kfold'].to_dict()
    except: return {}

def format_stat(mu, std):
    return f"{mu:.2f} ± {std:.2f}"

def get_model_info(path_obj):
    path_str = str(path_obj).lower()
    task_match = re.search(r'task_0(\d)', path_str)
    task = f"T0{task_match.group(1)}" if task_match else "Multi-Task"
    
    is_multimodal = 'multimodal' in path_str or 'combined' in path_str
    
    backbone = "Unknown"
    if 'xphonebert' in path_str: backbone = "XPhoneBERT"
    elif 'umberto' in path_str: backbone = "UmBERTo"
    elif 'xxl-cased' in path_str or 'bert-base-italian' in path_str: backbone = "BERT-IT"
    
    clf = ""
    if '_lr_' in path_str: clf = "LogReg"
    elif '_svm_' in path_str: clf = "SVM"
    elif '_xgboost_' in path_str: clf = "XGBoost"
    
    feat = ""
    if 'egemaps' in path_str: feat = "eGeMAPS"
    elif 'whisper' in path_str and 'text' not in path_str: feat = "Whisper-v3"
    elif 'compare' in path_str: feat = "ComParE"
    
    if is_multimodal:
        cat = "FUSION"
        name = f"Multimodal ({backbone if backbone != 'Unknown' else 'SOTA'})"
    elif backbone != "Unknown":
        cat = "TEXTUAL"
        name = backbone
    else:
        cat = "ACOUSTIC"
        name = f"{clf} + {feat}" if clf else feat

    return cat, name, task

def run_fold_analysis(df, fold_map, threshold, strategy):
    f_res = {m: [] for m in ['Acc', 'F1', 'Sens', 'Spec', 'Prec']}
    unique_folds = sorted([x for x in df['Fold'].unique() if not pd.isna(x)])
    for f_id in unique_folds:
        f_df = df[df['Fold'] == f_id]
        if strategy == 'max':
            p_df = f_df.groupby('PID').agg({'L': 'first', 'P': 'max'})
            y_pred = (p_df['P'] >= threshold).astype(int)
        elif strategy == 'mean':
            p_df = f_df.groupby('PID').agg({'L': 'first', 'P': 'mean'})
            y_pred = (p_df['P'] >= threshold).astype(int)
        elif strategy == 'hard':
            f_df['hp'] = (f_df['P'] >= 0.5).astype(int)
            p_df = f_df.groupby('PID').agg({'L': 'first', 'hp': lambda x: x.mode()[0] if not x.mode().empty else 0})
            y_pred = p_df['hp']
        elif strategy == 'confidence':
            p_df = f_df.groupby('PID').agg({'L': 'first', 'P': lambda x: x.iloc[np.argmax(np.abs(x.values - 0.5))]})
            y_pred = (p_df['P'] >= threshold).astype(int)
        
        m = calculate_metrics(p_df['L'], y_pred)
        if m:
            for k in f_res: f_res[k].append(m[k])
    return f_res

def main():
    fold_map = load_fold_map()
    all_files = []
    for d in SEARCH_DIRS:
        if d.exists(): all_files.extend(list(d.rglob("preds_*.csv")))
    
    full_results = []
    voting_strat = []

    for f in all_files:
        try:
            df = pd.read_csv(f)
            df.rename(columns={c: 'L' for c in df.columns if c.lower() in ['label', 'target']}, inplace=True)
            df.rename(columns={c: 'P' for c in df.columns if c.lower() in ['prob', 'ensemble_prob', 'probability', 'prob_ad']}, inplace=True)
            df.rename(columns={c: 'I' for c in df.columns if c.lower() in ['id', 'subject_id']}, inplace=True)
            if not all(k in df.columns for k in ['I', 'L', 'P']): continue
            df['PID'] = df['I'].apply(lambda x: str(x).split('_Task_')[0])
            df['Fold'] = df['PID'].map(fold_map)
            df = df.dropna(subset=['Fold'])

            cat, m_name, task = get_model_info(f)
            f_res = run_fold_analysis(df, fold_map, 0.5, 'max')
            
            if f_res['Acc']:
                row = {'Category': cat, 'Model': m_name, 'Task': task}
                for m in ['Acc', 'F1', 'Sens', 'Spec', 'Prec']:
                    row[f'{m}_mu'] = np.mean(f_res[m])
                    row[f'{m}_std'] = np.std(f_res[m])
                full_results.append(row)

                if BEST_MODEL_PATTERN in f.stem:
                    for s_name, s_type, thr in [('Soft Voting', 'mean', 0.5), ('Hard Voting', 'hard', 0.5), 
                                                ('Max Confidence', 'confidence', 0.5), ('Patient OR Triage', 'max', 0.35)]:
                        r = run_fold_analysis(df, fold_map, thr, s_type)
                        vs_row = {'Strategy': s_name, 'Acc_mu': np.mean(r['Acc'])}
                        for m in ['Acc', 'F1', 'Sens', 'Spec', 'Prec']:
                            vs_row[m] = format_stat(np.mean(r[m]), np.std(r[m]))
                        voting_strat.append(vs_row)
        except: continue

    df_full = pd.DataFrame(full_results)
    
    # Funzione helper per applicare il formato mu +- std a un dataframe
    def apply_stats(df_target):
        for m in ['Acc', 'F1', 'Sens', 'Spec', 'Prec']:
            df_target[m] = df_target.apply(lambda x: format_stat(x[f'{m}_mu'], x[f'{m}_std']), axis=1)
        return df_target

    print("\n" + "="*140)
    print(f"{'🎙️ TABELLA 1: MODELLI ACUSTICI (Dettaglio Completo)':^140}")
    print("="*140)
    ac = df_full[df_full['Category']=='ACOUSTIC'].sort_values('Acc_mu', ascending=False).copy()
    print(apply_stats(ac)[['Task', 'Model', 'Acc', 'F1', 'Sens', 'Spec', 'Prec']].to_string(index=False))

    print("\n" + "="*140)
    print(f"{'📝 TABELLA 2: MODELLI TESTUALI E FONETICI (Dettaglio Completo)':^140}")
    print("="*140)
    tx = df_full[df_full['Category']=='TEXTUAL'].sort_values(['Model', 'Task']).copy()
    print(apply_stats(tx)[['Task', 'Model', 'Acc', 'F1', 'Sens', 'Spec', 'Prec']].to_string(index=False))

    print("\n" + "="*140)
    print(f"{'🏆 TABELLA 3: PERFORMANCE MEDIA ARCHITETTURE (Overall)':^140}")
    print("="*140)
    arch = df_full.groupby(['Category', 'Model']).agg({f'{m}_{s}': 'mean' for m in ['Acc', 'F1', 'Sens', 'Spec', 'Prec'] for s in ['mu', 'std']}).reset_index()
    print(apply_stats(arch).sort_values('Acc_mu', ascending=False)[['Category', 'Model', 'Acc', 'F1', 'Sens', 'Spec', 'Prec']].to_string(index=False))

    print("\n" + "="*140)
    print(f"{'🏅 TABELLA 4: BEST MODEL PER TASK (Tutti i parametri)':^140}")
    print("="*140)
    task_winners = []
    for t in sorted(df_full['Task'].unique()):
        if t == "Multi-Task": continue
        winner = df_full[df_full['Task']==t].sort_values('Acc_mu', ascending=False).iloc[0:1].copy()
        task_winners.append(apply_stats(winner))
    print(pd.concat(task_winners)[['Task', 'Model', 'Acc', 'F1', 'Sens', 'Spec', 'Prec']].to_string(index=False))

    print("\n" + "="*140)
    print(f"{'⚖️ TABELLA 5: STRATEGIE DI VOTING (Modello Top: T01+T02)':^140}")
    print("="*140)
    print(pd.DataFrame(voting_strat).sort_values('Acc_mu', ascending=False)[['Strategy', 'Acc', 'F1', 'Sens', 'Spec', 'Prec']].to_string(index=False))
    print("="*140)
    print("\n💾 Salvataggio tabelle in corso...")
    
    # 1. Acustici
    ac_final = apply_stats(ac)[['Task', 'Model', 'Acc', 'F1', 'Sens', 'Spec', 'Prec']]
    ac_final.to_csv("TABELLA_1_ACUSTICI_DETTAGLIO.csv", index=False)
    print("- Salvata Tabella 1")

    # 2. Testuali
    tx_final = apply_stats(tx)[['Task', 'Model', 'Acc', 'F1', 'Sens', 'Spec', 'Prec']]
    tx_final.to_csv("TABELLA_2_TESTUALI_DETTAGLIO.csv", index=False)
    print("- Salvata Tabella 2")

    # 3. Architetture (Overall)
    arch_final = apply_stats(arch).sort_values('Acc_mu', ascending=False)[['Category', 'Model', 'Acc', 'F1', 'Sens', 'Spec', 'Prec']]
    arch_final.to_csv("TABELLA_3_ARCHI_OVERALL.csv", index=False)
    print("- Salvata Tabella 3")

    # 4. Best per Task
    winners_final = pd.concat(task_winners)[['Task', 'Model', 'Acc', 'F1', 'Sens', 'Spec', 'Prec']]
    winners_final.to_csv("TABELLA_4_BEST_PER_TASK.csv", index=False)
    print("- Salvata Tabella 4")

    # 5. Strategie Voting
    voting_final = pd.DataFrame(voting_strat).sort_values('Acc_mu', ascending=False)[['Strategy', 'Acc', 'F1', 'Sens', 'Spec', 'Prec']]
    voting_final.to_csv("TABELLA_5_STRATEGIE_VOTING.csv", index=False)
    print("- Salvata Tabella 5")

    print("\n✅ Tutte le tabelle sono state salvate nella cartella corrente.")

if __name__ == "__main__":
    main()