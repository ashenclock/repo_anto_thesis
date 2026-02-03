import sys
import torch
import torch.nn as nn
from pathlib import Path
from prettytable import PrettyTable # Se non ce l'hai: pip install prettytable

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models import build_model

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters", "Trainable %"])
    total_params = 0
    trainable_params = 0
    
    # Dettaglio per sottoclassi principali
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        ratio = (trainable / params * 100) if params > 0 else 0
        table.add_row([name, f"{params:,}", f"{ratio:.1f}%"])
        
        total_params += params
        trainable_params += trainable
        
    print(table)
    print(f"\n🔹 Totale Parametri:     {total_params:,}")
    print(f"🔥 Parametri Addestrabili: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    return total_params, trainable_params

def main():
    print("🔍 ISPEZIONE MODELLO SOTA...")
    
    # Carichiamo il config della regressione
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print("⚠️ Config regressione non trovato, uso default.")
        config_path = "config.yaml"
        
    config = load_config(config_path)
    
    # Costruiamo il modello
    model = build_model(config)
    model.eval()
    
    # CHECK AUDIO
    print("\n--- ANALISI ENCODER AUDIO ---")
    has_audio_encoder = hasattr(model, 'audio_encoder')
    if has_audio_encoder:
        print("✅ L'Audio Encoder (Wav2Vec2) è presente nel modello.")
        # Controllo se è congelato
        audio_params = list(model.audio_encoder.parameters())
        if audio_params:
            is_trainable = any(p.requires_grad for p in audio_params)
            print(f"   Stato: {'🔓 SCONGELATO (Trainable)' if is_trainable else '❄️ CONGELATO (Frozen)'}")
        else:
            print("   ⚠️ L'encoder è vuoto o non ha parametri.")
    else:
        print("❌ L'Audio Encoder NON è nel modello.")
        print("   (Il modello sta usando feature pre-estratte dai file .pt)")
        print("   Se vuoi fare fine-tuning audio, devi cambiare architettura.")

    # CHECK TEXT
    print("\n--- ANALISI ENCODER TESTO ---")
    if hasattr(model, 'text_encoder'):
        txt_params = list(model.text_encoder.parameters())
        is_trainable = any(p.requires_grad for p in txt_params)
        print(f"✅ Text Encoder presente.")
        print(f"   Stato: {'🔓 SCONGELATO (Trainable)' if is_trainable else '❄️ CONGELATO (Frozen)'}")

    # SUMMARY
    print("\n--- REPORT PARAMETRI ---")
    count_parameters(model)

if __name__ == "__main__":
    main()