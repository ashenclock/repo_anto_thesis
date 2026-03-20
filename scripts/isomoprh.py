import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Aggiungi root al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.models import build_model

# --- CONFIGURAZIONE ---
METADATA_FILE = Path("data/metadata/mmse_experiment_metadata.csv")
CONFIG_PATH = "config.yaml"
CHECKPOINT_PATH = Path("outputs/cross_dataset_IT_to_EN/best_model.pt") 

# Percorsi
IT_TRANSCRIPTS = Path("data/transcripts/Parakeet_Unified/Task_01")
EN_TRANSCRIPTS = Path("/home/speechlab/Desktop/pappagari2021-interspeech-replication/transcripts/parakeet-tdt-0.6b-v2/Task_01")
EN_TRANSCRIPTS_FALLBACK = Path("/home/speechlab/Desktop/pappagari2021-interspeech-replication/transcripts/parakeet-tdt-0.6b-v2")

def get_transcript_text(row):
    sid = row['Subject_ID']
    lang = row['Language']
    if lang == 'IT':
        clean_id = sid.split('_Task_')[0]
        path = IT_TRANSCRIPTS / f"{clean_id}.txt"
        if path.exists(): return path.read_text(encoding='utf-8').strip()
    elif lang == 'EN':
        clean_id = sid.split('_Task_')[0].replace("EN_TEST_", "").replace("EN_", "")
        path1 = EN_TRANSCRIPTS / f"{clean_id}.txt"
        path2 = EN_TRANSCRIPTS_FALLBACK / f"{clean_id}.txt"
        if path1.exists(): return path1.read_text(encoding='utf-8').strip()
        if path2.exists(): return path2.read_text(encoding='utf-8').strip()
    return ""

def extract_embeddings_finetuned(texts, text_encoder, tokenizer, device):
    text_encoder.eval()
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Estrazione Vettori 768D"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=288, padding=True).to(device)
            outputs = text_encoder(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).squeeze(0).cpu().numpy()
            embeddings.append(mean_pooled)
    return np.array(embeddings)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = load_config(CONFIG_PATH)
    config.model.text.name = "xlm-roberta-base"
    config.modality = "multimodal_cross_attention"
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    
    full_model = build_model(config).to(device)
    try: full_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    except: full_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    text_encoder = full_model.text_encoder

    # Carica Dati
    df = pd.read_csv(METADATA_FILE)
    df = df[df['Subject_ID'].str.contains("Task_01")].reset_index(drop=True)
    df['Diagnosis'] = df['Diagnosis'].replace({'MILD-AD': 'AD', 'cn': 'CTR'})
    df = df[df['Diagnosis'].isin(['CTR', 'AD'])].reset_index(drop=True)
    df['Text'] = df.apply(get_transcript_text, axis=1)
    df = df[df['Text'] != ""].reset_index(drop=True)
    
    # 1. Estrazione RAW (768D)
    embeddings = extract_embeddings_finetuned(df['Text'].tolist(), text_encoder, tokenizer, device)
    
    # 2. Centroidi per Lingua e Diagnosi
    it_ctr = embeddings[(df['Language'] == 'IT') & (df['Diagnosis'] == 'CTR')].mean(axis=0)
    it_ad  = embeddings[(df['Language'] == 'IT') & (df['Diagnosis'] == 'AD')].mean(axis=0)
    en_ctr = embeddings[(df['Language'] == 'EN') & (df['Diagnosis'] == 'CTR')].mean(axis=0)
    en_ad  = embeddings[(df['Language'] == 'EN') & (df['Diagnosis'] == 'AD')].mean(axis=0)
    
    # Centroide generale per lingua
    it_mean = embeddings[df['Language'] == 'IT'].mean(axis=0)
    en_mean = embeddings[df['Language'] == 'EN'].mean(axis=0)

    print("\n" + "="*80)
    print("🧠 DIMOSTRAZIONE DELL'ISOMORFISMO (Spazio 768D Originale)")
    print("="*80)

    # --- PROVA 1: IL VETTORE DI TRADUZIONE ---
    # Calcoliamo il vettore che "sposta" l'italiano sull'inglese
    vec_translation = en_mean - it_mean

    # Applico la traduzione ai Malati Italiani: dovrei ottenere i Malati Inglesi!
    mapped_it_ad = it_ad + vec_translation
    sim_ad = cosine_similarity(mapped_it_ad.reshape(1, -1), en_ad.reshape(1, -1))[0][0]

    # Applico la traduzione ai Sani Italiani: dovrei ottenere i Sani Inglesi!
    mapped_it_ctr = it_ctr + vec_translation
    sim_ctr = cosine_similarity(mapped_it_ctr.reshape(1, -1), en_ctr.reshape(1, -1))[0][0]

    print("PROVA 1: Translation Vector (IT + T -> EN)")
    print(f"  - IT_Malato + Vettore Traduzione vs EN_Malato = Cosine Similarità: {sim_ad:.4f}")
    print(f"  - IT_Sano   + Vettore Traduzione vs EN_Sano   = Cosine Similarità: {sim_ctr:.4f}")
    print("  (Se > 0.85, la struttura è un isomorfismo perfetto per traslazione)")

    # --- PROVA 2: RETRIEVAL ZERO-SHOT ---
    print("\nPROVA 2: Cross-Lingual Diagnosis Retrieval (K-Nearest Neighbors)")
    # Se prendo un paziente italiano, lo "traduco" aggiungendo vec_translation, 
    # e cerco il paziente inglese più vicino nello spazio 768D... hanno la stessa diagnosi?
    
    it_indices = df[df['Language'] == 'IT'].index
    en_indices = df[df['Language'] == 'EN'].index
    
    correct_matches = 0
    
    for i in it_indices:
        # Prendo il paziente IT, lo traduco spostandolo nello spazio EN
        mapped_patient = embeddings[i] + vec_translation
        
        # Calcolo le distanze con tutti i pazienti EN
        sims = cosine_similarity(mapped_patient.reshape(1, -1), embeddings[en_indices])[0]
        
        # Prendo l'inglese più simile a lui nello spazio 768D
        best_en_idx = en_indices[np.argmax(sims)]
        
        # Controllo se la diagnosi dell'italiano e dell'inglese trovato coincidono
        if df.iloc[i]['Diagnosis'] == df.iloc[best_en_idx]['Diagnosis']:
            correct_matches += 1
            
    accuracy = correct_matches / len(it_indices)
    print(f"  - Precisione nel mappare Sani con Sani e Malati con Malati cross-lingua: {accuracy*100:.2f}%")
    print("  (Se è molto superiore al 50%, significa che i cluster sono distribuiti nello stesso modo)")
    print("="*80)

if __name__ == "__main__":
    main()