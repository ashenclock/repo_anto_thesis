import sys
import os
import re
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal
from src.utils import set_seed

import warnings
warnings.filterwarnings("ignore")

# --- PATHS ---
NAPOLI_ROOT = Path("/home/speechlab/Desktop/Dati Tesi/Napoli")
WORKSPACE = Path("outputs/zero_shot_napoli_sota")
WORKSPACE.mkdir(parents=True, exist_ok=True)

TRANSCRIPT_DIR = WORKSPACE / "transcripts_parakeet"
AUDIO_FEAT_DIR = WORKSPACE / "features_xlsr"

# Modelli: Classificazione (UmBERTo) e Regressione (XLM-R)
CLASS_DIR = Path("outputs/combined_experiments/COMBINED_Task_01-Task_02_prova2")
REGR_DIR = Path("outputs/moca_regression_XLMR_139") # IL TUO MODELLO XLM-R MIGLIORE!

def prepare_napoli_metadata():
    records = []
    diag_map = {"Healthy": "CTR", "MCI": "MCI", "AD": "MILD-AD", "MCI_follow_up": "MCI"}
    dirs = [d for d in NAPOLI_ROOT.rglob("*") if d.is_dir() and "Audio" in d.name]
    for audio_dir in dirs:
        subj_id = audio_dir.parent.name
        diagnosis = "UNK"
        for k, v in diag_map.items():
            if k in str(audio_dir): diagnosis = v; break
        for wav in audio_dir.glob("*.wav"):
            if "shure" in wav.name.lower():
                match = re.search(r'(Task_[12])_', wav.name, re.IGNORECASE)
                if match:
                    t_name = f"Task_0{match.group(1)[-1]}"
                    records.append({
                        "Subject_ID": f"{subj_id}_{t_name}",
                        "Patient_ID": subj_id, "Diagnosis": diagnosis,
                        "Task": t_name, "audio_path": str(wav)
                    })
    return pd.DataFrame(records)

def get_predictions(df_sub, model_dir, text_backbone, device, is_regr=False):
    if df_sub.empty or not model_dir.exists(): return {}
    
    conf = load_config("config.yaml")
    conf.model.text.name = text_backbone
    conf.task = "regression" if is_regr else "classification"
    conf.model.output_dim = 1 if is_regr else 2
    conf.data.metadata_file = str(WORKSPACE / "napoli_meta.csv")
    conf.data.transcripts_root = str(TRANSCRIPT_DIR)
    conf.data.features_root = str(AUDIO_FEAT_DIR.parent)

    tokenizer = AutoTokenizer.from_pretrained(text_backbone)
    loader = torch.utils.data.DataLoader(MultimodalDataset(df_sub, conf, tokenizer), batch_size=1, collate_fn=collate_multimodal)
    models = list(model_dir.glob("*.pt"))
    accumulated = {sid: [] for sid in df_sub['Subject_ID']}
    
    for m_path in tqdm(models, desc=f"Inferenza {'Regr' if is_regr else 'Class'}", leave=False):
        model = build_model(conf).to(device)
        model.load_state_dict(torch.load(m_path, map_location=device, weights_only=False))
        model.eval()
        with torch.no_grad():
            for b in loader:
                sid = b.pop('id')[0]
                batch_in = {k: v.to(device) for k, v in b.items() if k != 'labels'}
                if is_regr: val = model(batch_in).view(-1).item()
                else: val = torch.softmax(model(batch_in), dim=1)[0, 1].item()
                accumulated[sid].append(val)
                
    return {sid: np.mean(v) for sid, v in accumulated.items() if v}

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = prepare_napoli_metadata()
    if df.empty: return
    df.to_csv(WORKSPACE / "napoli_meta.csv", index=False)

    print("🧠 Estrazione Probabilità e Score Cognitivi...")
    patients = {pid: {'GT': '', 'T1_P': 0.5, 'T2_P': 0.5, 'MoCA': []} for pid in df['Patient_ID'].unique()}

    # 1. Classificazione (UmBERTo)
    p1 = get_predictions(df[df['Task']=='Task_01'], CLASS_DIR, "Musixmatch/umberto-commoncrawl-cased-v1", device, is_regr=False)
    for sid, v in p1.items(): patients[sid.split('_Task_')[0]]['T1_P'] = v
    
    p2 = get_predictions(df[df['Task']=='Task_02'], CLASS_DIR, "Musixmatch/umberto-commoncrawl-cased-v1", device, is_regr=False)
    for sid, v in p2.items(): patients[sid.split('_Task_')[0]]['T2_P'] = v

    # 2. Regressione (XLM-RoBERTa)
    reg = get_predictions(df, REGR_DIR, "xlm-roberta-base", device, is_regr=True)
    for sid, v in reg.items(): patients[sid.split('_Task_')[0]]['MoCA'].append(v)

    # Salvataggio
    rows = []
    for pid, d in patients.items():
        gt = df[df['Patient_ID']==pid]['Diagnosis'].iloc[0]
        moca = np.mean(d['MoCA']) if d['MoCA'] else 0.0
        rows.append({'ID': pid, 'GT': gt, 'P1': d['T1_P'], 'P2': d['T2_P'], 'Predicted_MoCA': moca})
        
    out_csv = WORKSPACE / "REFERTO_NAPOLI_SOTA.csv"
    pd.DataFrame(rows).sort_values('GT').to_csv(out_csv, index=False)
    print(f"\n✅ Estrazione completata! Dati grezzi salvati in: {out_csv}")
    print("   -> Esegui 'python scripts/analysis.py' per vedere le metriche!")

if __name__ == "__main__":
    main()