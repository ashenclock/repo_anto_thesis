import sys
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.data import get_data_splits, get_dataloaders

def main():
    config = load_config("config.yaml")
    
    # Prendi il primo fold disponibile
    for fold, train_df, val_df in get_data_splits(config):
        print(f"\n--- DEBUG FOLD {fold} ---")
        print(f"Train samples: {len(train_df)}")
        print(f"Val samples: {len(val_df)}")
        
        # Crea i loader
        train_loader, val_loader = get_dataloaders(config, train_df, val_df)
        
        # Prendi UN solo batch
        print("\nEstraendo un batch dal train_loader...")
        batch = next(iter(train_loader))
        
        # --- ANALISI DEL BATCH ---
        print("\n[BATCH INFO]")
        print(f"ID Soggetti: {batch['id']}")
        print(f"Labels: {batch['labels']}")
        
        # Check Audio
        if 'waveform' in batch:
            wav = batch['waveform']
            print(f"Audio Shape: {wav.shape}")
            print(f"Audio Mean: {wav.mean().item():.6f}")
            print(f"Audio Max: {wav.max().item():.6f}")
            if wav.sum() == 0:
                print("❌ ERRORE GRAVE: L'audio è tutto zeri!")
            else:
                print("✅ Audio caricato correttamente (non è zero).")
        
        # Check Testo
        if 'input_ids' in batch:
            txt = batch['input_ids']
            print(f"Text Input IDs Shape: {txt.shape}")
            # Controlla se sono tutti token di padding (di solito 0 o 1)
            # Un input_id valido per UmBERTo/BERT è > 5 solitamente
            valid_tokens = (txt > 5).sum().item()
            print(f"Token validi (non speciali/padding): {valid_tokens}")
            
            if valid_tokens < 5:
                print("❌ ERRORE GRAVE: Il testo sembra vuoto o solo padding!")
            else:
                print("✅ Testo caricato correttamente.")

        break # Fermati al primo fold

if __name__ == "__main__":
    main()