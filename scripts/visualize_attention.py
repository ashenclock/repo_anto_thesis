import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import numpy as np
import math
from pathlib import Path
from transformers import AutoTokenizer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config
from src.data import MultimodalDataset, collate_multimodal
from src.models import build_model

# --- CONFIG ---
CONFIG_FILE = "config.yaml"
# Percorso corretto del modello
MODEL_PATH = "outputs/cross_dataset_IT_to_EN/best_model.pt"
 # Assicurati che il path sia relativo o assoluto corretto
TARGET_ID = "EN_TEST_adrsdt44"

# Durata di ogni "pagina" del grafico in secondi
CHUNK_SECONDS = 30 

def visualize_chunked(text_tokens, attn_matrix, title, base_filename):
    # Wav2Vec2 frame rate = 0.02s
    frame_duration = 0.02
    total_frames = attn_matrix.shape[1]
    total_seconds = total_frames * frame_duration
    
    frames_per_chunk = int(CHUNK_SECONDS / frame_duration)
    num_chunks = math.ceil(total_frames / frames_per_chunk)
    
    print(f"🔹 Generazione grafici per {base_filename} ({total_seconds:.2f}s totali -> {num_chunks} parti)")

    # Calcolo range globale per scala colori coerente
    valid_values = attn_matrix[attn_matrix > 1e-9]
    vmin = np.percentile(valid_values, 1) if len(valid_values) > 0 else None
    vmax = np.percentile(valid_values, 99) if len(valid_values) > 0 else None

    out_dir = Path("plots/attention_maps_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_chunks):
        start_frame = i * frames_per_chunk
        end_frame = min((i + 1) * frames_per_chunk, total_frames)
        
        # Se il chunk è vuoto o troppo piccolo, salta
        if start_frame >= end_frame:
            continue
            
        chunk_matrix = attn_matrix[:, start_frame:end_frame]
        
        # Per leggibilità, filtriamo i token (asse Y) che non hanno attenzione in questo lasso di tempo
        # (Opzionale, ma aiuta se il testo è lunghissimo)
        # Qui manteniamo tutto per semplicità, ma potresti filtrare righe tutte a zero.

        # Setup Plot
        plt.figure(figsize=(20, 10)) # Dimensione fissa leggibile
        
        ax = sns.heatmap(
            chunk_matrix, 
            cmap="viridis", 
            yticklabels=text_tokens,
            vmin=vmin, 
            vmax=vmax,
            cbar_kws={'label': 'Attention Probability'}
        )
        
        # Gestione Asse X (Tempo)
        chunk_duration = (end_frame - start_frame) * frame_duration
        start_time_abs = start_frame * frame_duration
        
        # Ticks ogni 2 secondi
        step_sec = 2.0
        ticks_locs = np.arange(0, end_frame - start_frame, int(step_sec / frame_duration))
        ticks_labels = [f"{start_time_abs + (t * frame_duration):.0f}s" for t in ticks_locs]
        
        ax.set_xticks(ticks_locs)
        ax.set_xticklabels(ticks_labels, rotation=0)
        
        plt.title(f"{title} - Part {i+1}/{num_chunks} ({start_time_abs:.0f}s - {start_time_abs+chunk_duration:.0f}s)", fontsize=16)
        plt.xlabel("Time (seconds)", fontsize=14)
        plt.ylabel("Text Tokens", fontsize=14)
        
        # Salva
        out_path = out_dir / f"{base_filename}_part{i+1:02d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight') # DPI 150 è più leggero
        plt.close()
        print(f"   -> Salvato: {out_path.name}")

def main():
    if not Path(MODEL_PATH).exists(): 
        print(f"❌ Modello mancante: {MODEL_PATH}")
        return

    with open(CONFIG_FILE) as f: config = Config(yaml.safe_load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("⏳ Caricamento modello...")
    model = build_model(config).to(device)
    # Gestione caricamento pesi (sicuro vs non sicuro)
    try: 
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
    model.load_state_dict(checkpoint)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)

    # Cerca paziente
    df = pd.read_csv(config.data.metadata_file)
    candidates = df[df['Subject_ID'].str.contains(TARGET_ID)]
    if len(candidates) == 0: 
        print(f"❌ Paziente {TARGET_ID} non trovato nel metadata.")
        return
    sample_row = candidates.iloc[0]
    
    print(f"🔹 Analisi Paziente: {sample_row['Subject_ID']} (Duration: {sample_row['duration']}s)")

    ds = MultimodalDataset(pd.DataFrame([sample_row]), config, tokenizer)
    batch = collate_multimodal([ds[0]])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        _, t_attn, a_attn = model(batch, return_attention=True)
    
    input_ids = batch['input_ids'][0].cpu().numpy()
    tokens = [t.replace(' ', '') for t in tokenizer.convert_ids_to_tokens(input_ids)]
    
    # Visualizza a pezzi
    # Text-to-Audio
    visualize_chunked(tokens, t_attn[0].cpu().numpy(), 
                      f"Text-to-Audio ({TARGET_ID})", f"attn_TA_{TARGET_ID}")

    # Audio-to-Text (Trasposta per avere Audio su X e Text su Y)
    visualize_chunked(tokens, a_attn[0].cpu().numpy().T, 
                      f"Audio-to-Text ({TARGET_ID})", f"attn_AT_{TARGET_ID}")

if __name__ == "__main__":
    main()