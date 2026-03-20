import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- CONFIGURAZIONE ---
METADATA_FILE = Path("data/metadata/mmse_experiment_metadata.csv")
AUDIO_FEATS_ROOT = Path("data/features/wav2vec2-large-xlsr-53_sequences/Task_01")

def load_audio_embeddings_robust(df_input):
    df = df_input.reset_index(drop=True)
    embeddings, found_indices = [], []
    
    # Crea una lista di tutti i file .pt disponibili per velocizzare la ricerca
    available_files = {f.stem: f for f in AUDIO_FEATS_ROOT.glob("*.pt")}
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Caricamento Audio"):
        sid = str(row['Subject_ID'])
        clean_id = sid.split('_Task_')[0]
        
        # Genera possibili varianti del nome file
        variants = [
            clean_id,
            clean_id.replace("EN_", "").replace("TEST_", ""),
            clean_id.replace("EN_TEST_", ""),
            "adrso" + clean_id.split("adrso")[-1] if "adrso" in clean_id else clean_id,
            "adrsdt" + clean_id.split("adrsdt")[-1] if "adrsdt" in clean_id else clean_id
        ]
        
        pt_path = None
        for v in variants:
            if v in available_files:
                pt_path = available_files[v]
                break
        
        if pt_path:
            try:
                tensor = torch.load(pt_path, map_location='cpu', weights_only=True)
                # Mean pooling temporale
                vec = torch.mean(tensor, dim=0).numpy() if tensor.dim() > 1 else tensor.numpy()
                embeddings.append(vec)
                found_indices.append(i)
            except: continue

    return np.array(embeddings), df.iloc[found_indices].reset_index(drop=True)

def main():
    print(f"\n{'='*70}")
    print(f"🕵️  RICERCA ALLINEAMENTO NASCOSTO NELLO SPAZIO AUDIO (V2)")
    print(f"{'='*70}")

    if not METADATA_FILE.exists(): return
    
    df_full = pd.read_csv(METADATA_FILE)
    df_full['Diagnosis'] = df_full['Diagnosis'].replace({'MILD-AD': 'AD', 'cn': 'CTR'})
    df_target = df_full[df_full['Diagnosis'].isin(['CTR', 'AD'])].copy()
    
    X, df = load_audio_embeddings_robust(df_target)
    
    it_mask = (df['Language'] == 'IT')
    en_mask = (df['Language'] == 'EN')

    print(f"✅ Dati caricati: IT={it_mask.sum()} samples, EN={en_mask.sum()} samples")

    if it_mask.sum() == 0 or en_mask.sum() == 0:
        print("❌ Gruppi linguistici mancanti. Controlla i path dei file .pt!")
        return

    # --- FASE 1: DE-NOISING (Domain Removal) ---
    # Rimuoviamo le componenti che spiegano la differenza tra le lingue
    # Calcoliamo il vettore medio dello "shift linguistico"
    mu_it = X[it_mask].mean(axis=0)
    mu_en = X[en_mask].mean(axis=0)
    
    # Centriamo i dati (rimuove lo shift globale)
    X_centered = X.copy()
    X_centered[it_mask] -= mu_it
    X_centered[en_mask] -= mu_en

    # --- FASE 2: PROCRUSTE (Rotazione Ottimale) ---
    it_ctr = X_centered[it_mask & (df['Diagnosis'] == 'CTR')].mean(axis=0)
    it_ad  = X_centered[it_mask & (df['Diagnosis'] == 'AD')].mean(axis=0)
    en_ctr = X_centered[en_mask & (df['Diagnosis'] == 'CTR')].mean(axis=0)
    en_ad  = X_centered[en_mask & (df['Diagnosis'] == 'AD')].mean(axis=0)

    # Verifichiamo che i centroidi siano validi (niente NaN)
    if np.any(np.isnan([it_ctr, it_ad, en_ctr, en_ad])):
        print("❌ Errore: Alcuni sottogruppi (CTR/AD) sono vuoti.")
        return

    A = np.vstack([it_ctr, it_ad])
    B = np.vstack([en_ctr, en_ad])

    # Calcoliamo la rotazione ottimale
    R, _ = orthogonal_procrustes(A, B)
    B_transformed = B @ R

    # Calcoliamo la distanza tra le direzioni della malattia
    vec_it = it_ad - it_ctr
    vec_en_rot = B_transformed[1] - B_transformed[0]
    
    dist_raw = cosine(it_ad - it_ctr, en_ad - en_ctr)
    dist_hidden = cosine(vec_it, vec_en_rot)

    print("\n" + "-"*70)
    print(f"📊 ANALISI DELLA TOPOLOGIA ACUSTICA:")
    print(f"  - Distanza Coseno Raw:      {dist_raw:.4f}")
    print(f"  - Distanza Coseno Allineata: {dist_hidden:.4f}")
    print("-" * 70)

    if dist_hidden < 0.2:
        print("\n✨ SCOPERTA: L'isomorfismo audio esiste ed è quasi perfetto!")
    elif dist_hidden < dist_raw:
        improvement = ((dist_raw - dist_hidden) / dist_raw) * 100
        print(f"\n✅ ALLINEAMENTO TROVATO: La rotazione spiega il {improvement:.1f}% della divergenza.")
        print("L'Alzheimer ha una firma acustica simile, ma 'ruotata' dal dominio hardware/lingua.")
    else:
        print("\n❌ Divergenza strutturale: l'audio non è isomorfo tra queste due lingue.")

if __name__ == "__main__":
    main()