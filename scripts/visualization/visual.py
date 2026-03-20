import sys
import pandas as pd
import numpy as np
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_distances

# Setup path per i tuoi moduli
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.config import load_config
from src.models import build_model

# --- CONFIG ---
METADATA_FILE = Path("data/metadata/mmse_experiment_metadata.csv")
CONFIG_PATH = "config.yaml"
CHECKPOINT_PATH = Path("outputs/cross_dataset_IT_to_EN/best_model.pt")

# Percorsi Trascrizioni reali
IT_DIR = Path("data/transcripts/Parakeet_Unified/Task_01")
EN_DIR = Path("/home/speechlab/Desktop/pappagari2021-interspeech-replication/transcripts/parakeet-tdt-0.6b-v2/Task_01")

def clean_and_split(text):
    """Pulisce il testo e lo divide in frasi sensate"""
    # Rimuove tag tipo <pause>, [laughter], ecc.
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    # Divide per punteggiatura
    sentences = re.split(r'[.!?]+', text)
    # Tiene solo frasi con almeno 5 parole per evitare "Sì", "No", "Allora"
    return [s.strip() for s in sentences if len(s.strip().split()) >= 5]

def get_embedding(texts, model, tokenizer, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
            out = model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            emb = (torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9))
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(CONFIG_PATH)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    full_model = build_model(config).to(device)
    full_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    text_encoder = full_model.text_encoder

    # 1. Caricamento Trascrizioni Reali
    print("📂 Estrazione frasi dai dati reali...")
    df = pd.read_csv(METADATA_FILE)
    it_sentences = []
    en_sentences = []

    for _, row in df.iterrows():
        sid = row['Subject_ID']
        clean_id = sid.split('_Task_')[0]
        if row['Language'] == 'IT':
            p = IT_DIR / f"{clean_id}.txt"
            if p.exists(): it_sentences.extend(clean_and_split(p.read_text()))
        else:
            clean_id = clean_id.replace("EN_TEST_", "").replace("EN_", "")
            p = EN_DIR / f"{clean_id}.txt"
            if p.exists(): en_sentences.extend(clean_and_split(p.read_text()))

    # Rimuovi duplicati per velocizzare
    it_sentences = list(set(it_sentences))[:300] # Limite per velocità
    en_sentences = list(set(en_sentences))[:300]

    print(f"✅ Estratte {len(it_sentences)} frasi IT e {len(en_sentences)} frasi EN.")

    # 2. Estrazione Embeddings
    emb_it = get_embedding(it_sentences, text_encoder, tokenizer, device)
    emb_en = get_embedding(en_sentences, text_encoder, tokenizer, device)

    # 3. Calcolo Distanze Cross-Lingua
    print("📏 Calcolo distanze semantiche...")
    dists = cosine_distances(emb_it, emb_en)

    # 4. Trova i match migliori
    # Prendiamo le 5 coppie con la distanza minore
    best_matches = []
    flat_indices = np.argsort(dists, axis=None)[:10]
    
    print(f"\n{'='*100}")
    print(f"{'TOP MATCHES SEMANTICI SCOPERTI NEI DATI REALI':^100}")
    print(f"{'='*100}\n")

    seen_it = set()
    count = 0
    for idx in flat_indices:
        it_idx, en_idx = np.unravel_index(idx, dists.shape)
        if it_idx in seen_it: continue # Evita di ripetere la stessa frase IT
        
        dist_val = dists[it_idx, en_idx]
        print(f"🏆 Match #{count+1} (Distanza Coseno: {dist_val:.4f})")
        print(f"   🇮🇹 IT: \"{it_sentences[it_idx]}\"")
        print(f"   🇬🇧 EN: \"{en_sentences[en_idx]}\"")
        print("-" * 100)
        
        seen_it.add(it_idx)
        count += 1
        if count >= 5: break

if __name__ == "__main__":
    main()