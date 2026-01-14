import argparse
import sys
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor, Wav2Vec2Model, WhisperModel
from torchaudio.functional import resample

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    device = torch.device(config.device)
    model_name = config.model.audio.pretrained
    
    # Cartella di output per i tensori
    # Es: data/features/wav2vec2-xls-r-300m_sequences
    safe_name = model_name.split('/')[-1]
    output_root = Path(config.data.features_root) / f"{safe_name}_sequences"
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Pre-computing Audio Features: {model_name} ---")
    print(f"Output: {output_root}")

    # Caricamento Modello Audio
    print("Caricamento modello su GPU...")
    if "whisper" in model_name.lower():
        model = WhisperModel.from_pretrained(model_name).encoder.to(device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        is_whisper = True
    else:
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        is_whisper = False
    model.eval()

    # Leggi metadati
    df = pd.read_csv(config.data.metadata_file)
    df = df[df['has_audio'] == True]
    
    # Processa tutti i file
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            audio_path = Path(row['audio_path'])
            
            # Crea sottocartella per il Task (es. Task_01)
            # Estraiamo il task dal Subject_ID (che abbiamo creato come SUBJ_XXX_Task_XX)
            task_name = row['Subject_ID'].split('_Task_')[-1]
            task_name = f"Task_{task_name}"
            
            task_dir = output_root / task_name
            task_dir.mkdir(exist_ok=True)
            
            # Nome file output (SUBJ_XXXX.pt)
            subj_only_id = row['Subject_ID'].split('_Task_')[0]
            out_file = task_dir / f"{subj_only_id}.pt"
            
            if out_file.exists(): continue

            # Carica e processa audio
            try:
                wav, sr = torchaudio.load(audio_path)
                if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
                if sr != 16000: wav = resample(wav, sr, 16000)
                
                # Taglio a 10s (o 15s) per sicurezza memoria e dimensione disco
                max_len = 16000 * 120 
                if wav.shape[1] > max_len: wav = wav[:, :max_len]

                if is_whisper:
                    inputs = feature_extractor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
                    feats = inputs.input_features.to(device)
                    out = model(feats).last_hidden_state # [1, Time, Dim]
                else:
                    wav = wav.to(device)
                    out = model(wav).last_hidden_state # [1, Time, Dim]
                
                # Salva il tensore su disco (CPU)
                # Salviamo [Time, Dim] spremendo il batch per risparmiare spazio
                torch.save(out.squeeze(0).cpu(), out_file)
                
            except Exception as e:
                print(f"Errore {audio_path}: {e}")

    print("\n✅ Pre-computazione completata!")

if __name__ == "__main__":
    main()