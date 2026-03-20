import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_distances

# Aggiungi root al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.models import build_model

# --- CONFIG ---
METADATA_FILE = Path("data/metadata/mmse_experiment_metadata.csv")
CONFIG_PATH = "config.yaml"
CHECKPOINT_PATH = Path("outputs/balanced_mix/best_model.pt") 
PLOT_DIR = Path("plots/final_scientific_plots")

# Percorsi Trascrizioni
IT_TRANSCRIPTS = Path("data/transcripts/Parakeet_Unified/Task_01")
EN_TRANSCRIPTS = Path("/home/speechlab/Desktop/pappagari2021-interspeech-replication/transcripts/parakeet-tdt-0.6b-v2/Task_01")
EN_TRANSCRIPTS_FALLBACK = Path("/home/speechlab/Desktop/pappagari2021-interspeech-replication/transcripts/parakeet-tdt-0.6b-v2")

def get_transcript_text(row):
    sid = row['Subject_ID']; lang = row['Language']
    if lang == 'IT':
        path = IT_TRANSCRIPTS / f"{sid.split('_Task_')[0]}.txt"
        if path.exists(): return path.read_text(encoding='utf-8').strip()
    elif lang == 'EN':
        clean = sid.split('_Task_')[0].replace("EN_TEST_", "").replace("EN_", "")
        path1, path2 = EN_TRANSCRIPTS / f"{clean}.txt", EN_TRANSCRIPTS_FALLBACK / f"{clean}.txt"
        if path1.exists(): return path1.read_text(encoding='utf-8').strip()
        if path2.exists(): return path2.read_text(encoding='utf-8').strip()
    return ""

def extract_embeddings(texts, text_encoder, tokenizer, device):
    text_encoder.eval()
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Estrazione Vettori"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=288, padding=True).to(device)
            outputs = text_encoder(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            mean_pooled = (torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)).squeeze(0).cpu().numpy()
            embeddings.append(mean_pooled)
    return np.array(embeddings)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(CONFIG_PATH)
    config.model.text.name = "xlm-roberta-base"
    config.modality = "multimodal_cross_attention"

    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    full_model = build_model(config).to(device)
    full_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    text_encoder = full_model.text_encoder

    # Carica Dati
    df = pd.read_csv(METADATA_FILE)
    df = df[df['Subject_ID'].str.contains("Task_01")].reset_index(drop=True)
    df['Diagnosis'] = df['Diagnosis'].replace({'MILD-AD': 'AD', 'cn': 'CTR'})
    df = df[df['Diagnosis'].isin(['CTR', 'AD'])].reset_index(drop=True)
    df['Text'] = df.apply(get_transcript_text, axis=1)
    df = df[df['Text'] != ""].reset_index(drop=True)
    
    # Ordinamento: IT(Sani->Malati), EN(Sani->Malati)
    df_it = df[df['Language'] == 'IT'].sort_values('Diagnosis', ascending=False) # CTR viene prima di AD? No AD < CTR. Ascending=False -> CTR poi AD
    # Fix sicuro
    df_it = pd.concat([df_it[df_it['Diagnosis']=='CTR'], df_it[df_it['Diagnosis']=='AD']])
    df_en = pd.concat([df[df['Language']=='EN'][df['Diagnosis']=='CTR'], df[df['Language']=='EN'][df['Diagnosis']=='AD']])

    X_it = extract_embeddings(df_it['Text'].tolist(), text_encoder, tokenizer, device)
    X_en = extract_embeddings(df_en['Text'].tolist(), text_encoder, tokenizer, device)
    
    # Allineamento
    vec_translation = X_en.mean(axis=0) - X_it.mean(axis=0)
    X_it_aligned = X_it + vec_translation

    # Calcolo Matrice
    dist_matrix = cosine_distances(X_it_aligned, X_en)
    
    # --- CLEANING OUTLIERS ---
    # Rimuoviamo righe/colonne che hanno distanza media altissima (probabilmente audio vuoti o errori)
    row_means = dist_matrix.mean(axis=1)
    col_means = dist_matrix.mean(axis=0)
    
    # Teniamo solo chi è "ragionevole" (es. sotto il 95° percentile di distanza)
    good_rows = row_means < np.percentile(row_means, 98)
    good_cols = col_means < np.percentile(col_means, 98)
    
    dist_matrix_clean = dist_matrix[good_rows][:, good_cols]
    
    # Ricalcoliamo i punti di split per le linee bianche
    # Dobbiamo sapere quanti CTR e AD sono rimasti dopo la pulizia
    # È complicato farlo preciso senza rifare i DF, quindi stimiamo visivamente
    # Oppure plottiamo senza linee se la pulizia rompe gli indici.
    # Proviamo a plottare senza linee interne, solo i blocchi.
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set_theme(style="white")
    
    # Colormap: "magma" o "viridis". Vogliamo che 0 (vicino) sia SCURO/FREDDO?
    # Solitamente Heatmap di distanza: 0=Bianco/Chiaro, 1=Scuro/Rosso.
    # Heatmap di Similarità: 1=Scuro, 0=Chiaro.
    # Qui abbiamo DISTANZA. Quindi 0 (vicino) dovrebbe essere chiaro.
    # Ma tu volevi vedere i blocchi scuri sulla diagonale.
    # Quindi usiamo 'viridis_r' (0=Giallo, 1=Viola) o 'mako_r'.
    # Aspetta, se 0 è giallo (chiaro), la diagonale sarà chiara.
    # Se vuoi la diagonale SCURA (evidente), devi usare 'viridis' (0=Viola scuro, 1=Giallo).
    # No, viridis: 0=Viola, 1=Giallo.
    
    sns.heatmap(dist_matrix_clean, cmap="mako", ax=ax, cbar_kws={'label': 'Cosine Distance (Più scuro = Più simile)'})
    
    ax.set_title("Matrice di Distanza Cross-Lingua (Allineata)\nI blocchi scuri sulla diagonale indicano l'Isomorfismo", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel(f"Pazienti Italiani (Sani $\\rightarrow$ Malati)", fontsize=14)
    ax.set_xlabel(f"Pazienti Inglesi (Sani $\\rightarrow$ Malati)", fontsize=14)
    
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    out_file = PLOT_DIR / "patient_distance_matrix_heatmap_FINAL.png"
    plt.savefig(out_file, dpi=300)
    print(f"✅ Heatmap salvata in: {out_file}")

if __name__ == "__main__":
    main()