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
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.models import build_model

# --- CONFIGURAZIONE ---
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
        for text in tqdm(texts, desc="Processing"):
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

    # 1. Caricamento Dati
    df = pd.read_csv(METADATA_FILE)
    df = df[df['Subject_ID'].str.contains("Task_01")].reset_index(drop=True)
    df['Diagnosis'] = df['Diagnosis'].replace({'MILD-AD': 'AD', 'cn': 'CTR'})
    df = df[df['Diagnosis'].isin(['CTR', 'AD'])].reset_index(drop=True)
    df['Text'] = df.apply(get_transcript_text, axis=1)
    df = df[df['Text'] != ""].reset_index(drop=True)
    
    X_768 = extract_embeddings(df['Text'].tolist(), text_encoder, tokenizer, device)
    
    # 2. Allineamento
    shift_vector = X_768[df['Language'] == 'EN'].mean(axis=0) - X_768[df['Language'] == 'IT'].mean(axis=0)
    X_it_aligned = X_768[df['Language'] == 'IT'] + shift_vector
    X_en_raw = X_768[df['Language'] == 'EN']
    
    it_mask_diag = df[df['Language'] == 'IT']['Diagnosis']
    en_mask_diag = df[df['Language'] == 'EN']['Diagnosis']

    # 3. Calcolo Matrice
    dist_matrix = np.zeros((2, 2))
    dist_matrix[0, 0] = cosine_distances(X_it_aligned[it_mask_diag=='CTR'], X_en_raw[en_mask_diag=='CTR']).mean()
    dist_matrix[0, 1] = cosine_distances(X_it_aligned[it_mask_diag=='CTR'], X_en_raw[en_mask_diag=='AD']).mean()
    dist_matrix[1, 0] = cosine_distances(X_it_aligned[it_mask_diag=='AD'],  X_en_raw[en_mask_diag=='CTR']).mean()
    dist_matrix[1, 1] = cosine_distances(X_it_aligned[it_mask_diag=='AD'],  X_en_raw[en_mask_diag=='AD']).mean()

    # 4. PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_768)
    df['pca_1'] = X_2d[:, 0]; df['pca_2'] = X_2d[:, 1]
    
    # Centroidi 2D
    it_ctr_mask = (df['Language'] == 'IT') & (df['Diagnosis'] == 'CTR')
    it_ad_mask  = (df['Language'] == 'IT') & (df['Diagnosis'] == 'AD')
    en_ctr_mask = (df['Language'] == 'EN') & (df['Diagnosis'] == 'CTR')
    en_ad_mask  = (df['Language'] == 'EN') & (df['Diagnosis'] == 'AD')
    
    c_it_ctr = df[it_ctr_mask][['pca_1', 'pca_2']].mean().values
    c_it_ad  = df[it_ad_mask][['pca_1', 'pca_2']].mean().values
    c_en_ctr = df[en_ctr_mask][['pca_1', 'pca_2']].mean().values
    c_en_ad  = df[en_ad_mask][['pca_1', 'pca_2']].mean().values

    # =========================================================================
    # PLOTTING IMMAGINE 1: LA MATRICE
    # =========================================================================
    sns.set_theme(style="whitegrid")
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(dist_matrix, annot=True, fmt=".4f", cmap="Reds_r", 
                xticklabels=["Sani (EN)", "AD (EN)"], yticklabels=["Sani (IT)", "AD (IT)"], ax=ax1,
                cbar_kws={'label': 'Distanza Coseno'}, annot_kws={"size": 14})
    
    plt.tight_layout()
    out_matrice = PLOT_DIR / "isomorfismo_matrice.png"
    plt.savefig(out_matrice, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"✅ Matrice salvata in: {out_matrice}")

    # =========================================================================
    # PLOTTING IMMAGINE 2: IL PARALLELOGRAMMA (PCA)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    # Sfondo sbiadito
    ax2.scatter(df[it_ctr_mask]['pca_1'], df[it_ctr_mask]['pca_2'], c='#66c2a5', alpha=0.15, s=50, edgecolors='none')
    ax2.scatter(df[it_ad_mask]['pca_1'],  df[it_ad_mask]['pca_2'],  c='#66c2a5', alpha=0.15, s=50, edgecolors='none')
    ax2.scatter(df[en_ctr_mask]['pca_1'], df[en_ctr_mask]['pca_2'], c='#fc8d62', alpha=0.15, s=50, edgecolors='none')
    ax2.scatter(df[en_ad_mask]['pca_1'],  df[en_ad_mask]['pca_2'],  c='#fc8d62', alpha=0.15, s=50, edgecolors='none')

    # Centroidi saturi
    ax2.scatter(c_it_ctr[0], c_it_ctr[1], color='#66c2a5', s=450, edgecolor='black', linewidth=2, zorder=10)
    ax2.scatter(c_it_ad[0],  c_it_ad[1],  color='#66c2a5', s=450, edgecolor='black', linewidth=2, zorder=10)
    ax2.scatter(c_en_ctr[0], c_en_ctr[1], color='#fc8d62', s=450, edgecolor='black', linewidth=2, zorder=10)
    ax2.scatter(c_en_ad[0],  c_en_ad[1],  color='#fc8d62', s=450, edgecolor='black', linewidth=2, zorder=10)
    
    # Frecce MALATTIA (Rosse e spesse)
    ax2.annotate('', xy=c_it_ad, xytext=c_it_ctr, arrowprops=dict(facecolor='red', edgecolor='red', width=6, shrink=0.08, zorder=20))
    ax2.annotate('', xy=c_en_ad, xytext=c_en_ctr, arrowprops=dict(facecolor='red', edgecolor='red', width=6, shrink=0.08, zorder=20))
    
    # Frecce LINGUA (Nere, tratteggiate, molto visibili)
    ax2.annotate('', xy=c_en_ctr, xytext=c_it_ctr, arrowprops=dict(facecolor='black', edgecolor='black', linestyle='--', width=2.5, headwidth=10, shrink=0.08, zorder=15))
    ax2.annotate('', xy=c_en_ad,  xytext=c_it_ad,  arrowprops=dict(facecolor='black', edgecolor='black', linestyle='--', width=2.5, headwidth=10, shrink=0.08, zorder=15))

    # TAG TESTUALI (Label sui punti)
    offset_y = 0.5
    ax2.text(c_it_ctr[0], c_it_ctr[1]-offset_y, "Sani (IT)", fontsize=14, ha='center', weight='bold')
    ax2.text(c_it_ad[0],  c_it_ad[1]+offset_y,  "AD (IT)", fontsize=14, ha='center', weight='bold')
    ax2.text(c_en_ctr[0], c_en_ctr[1]-offset_y, "Sani (EN)", fontsize=14, ha='center', weight='bold')
    ax2.text(c_en_ad[0],  c_en_ad[1]+offset_y,  "AD (EN)", fontsize=14, ha='center', weight='bold')

    # Legenda
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#66c2a5', markersize=15, label='Dominio Italiano', markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#fc8d62', markersize=15, label='Dominio Inglese', markeredgecolor='k'),
        Line2D([0], [0], color='red', lw=4, label='Direzione Malattia'),
        Line2D([0], [0], color='black', lw=2.5, linestyle='--', label='Direzione Lingua')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=13, frameon=True)
    
    plt.tight_layout()
    
    out_pca = PLOT_DIR / "isomorfismo_pca.png"
    plt.savefig(out_pca, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ PCA salvata in: {out_pca}")

if __name__ == "__main__":
    main()