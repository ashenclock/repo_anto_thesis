import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import numpy as np
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
MODEL_PATH = "/home/speechlab/Desktop/repo_anto_thesis/outputs/balanced_mix/best_model.pt"
TARGET_ID = "EN_adrso234" 

def visualize_full(text_tokens, attn_matrix, title, filename):
    # Wav2Vec2 frame rate = 0.02s
    frame_duration = 0.02
    total_frames = attn_matrix.shape[1]
    total_seconds = total_frames * frame_duration
    
    # Calcoliamo una larghezza dinamica: 20 pollici ogni 10 secondi di audio
    # Così il grafico si allarga se l'audio è lungo
    dynamic_width = max(20, int(total_seconds * 2))
    
    plt.figure(figsize=(dynamic_width, 10))
    
    # Smart Scaling (Ignora padding per i colori)
    valid_values = attn_matrix[attn_matrix > 1e-9]
    vmin = np.percentile(valid_values, 1) if len(valid_values) > 0 else None
    vmax = np.percentile(valid_values, 99) if len(valid_values) > 0 else None

    # Plot
    ax = sns.heatmap(
        attn_matrix, 
        cmap="viridis", 
        yticklabels=text_tokens,
        vmin=vmin, 
        vmax=vmax,
        cbar_kws={'label': 'Attention Probability'}
    )
    
    # Asse X in Secondi (un tick ogni 5 secondi per non affollare)
    step_sec = 5.0 
    ticks_secs = np.arange(0, total_seconds, step_sec)
    ticks_frames = ticks_secs / frame_duration
    
    # Filtra ticks fuori range
    valid_indices = [i for i, t in enumerate(ticks_frames) if t < total_frames]
    
    ax.set_xticks(ticks_frames[valid_indices])
    ax.set_xticklabels([f"{ticks_secs[i]:.0f}s" for i in valid_indices], rotation=0)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel("Text Tokens", fontsize=14)
    
    # Salva
    out_dir = Path("plots/attention_maps_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot salvato: {out_path}")

def main():
    if not Path(MODEL_PATH).exists(): print("Modello mancante"); return

    with open(CONFIG_FILE) as f: config = Config(yaml.safe_load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(config).to(device)
    try: model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except: model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)

    # Cerca paziente
    df = pd.read_csv(config.data.metadata_file)
    candidates = df[df['Subject_ID'].str.contains(TARGET_ID)]
    if len(candidates) == 0: print(f"Paziente {TARGET_ID} non trovato."); return
    sample_row = candidates.iloc[0]
    
    print(f"🔹 Analisi FULL ({sample_row['duration']}s): {sample_row['Subject_ID']}")

    ds = MultimodalDataset(pd.DataFrame([sample_row]), config, tokenizer)
    batch = collate_multimodal([ds[0]])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        _, t_attn, a_attn = model(batch, return_attention=True)
    
    input_ids = batch['input_ids'][0].cpu().numpy()
    tokens = [t.replace(' ', '') for t in tokenizer.convert_ids_to_tokens(input_ids)]
    
    # Visualizza TUTTO
    # Nota: Usiamo .cpu().numpy() senza slicing [:max]
    visualize_full(tokens, t_attn[0].cpu().numpy(), 
                   f"Text-to-Audio FULL ({TARGET_ID})", "attn_TA_FULL")

    visualize_full(tokens, a_attn[0].cpu().numpy().T, 
                   f"Audio-to-Text FULL ({TARGET_ID})", "attn_AT_FULL")

if __name__ == "__main__":
    main()