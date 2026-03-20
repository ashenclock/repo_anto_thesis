import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

# --- CONFIG ---
TARGET_ID = "EN_adrso234" 
SEARCH_ROOT = Path("data/dataset")
OUTPUT_DIR = Path("plots/attention_maps_full") 
CHUNK_SECONDS = 30  # Durata per ogni immagine

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = list(SEARCH_ROOT.rglob(f"*{TARGET_ID}*.wav"))
    if not candidates: 
        print(f"❌ Audio non trovato per {TARGET_ID}")
        return
    
    audio_path = candidates[0]
    print(f"🔹 Analisi Spettrogramma Chunked: {audio_path.name}")

    y, sr = librosa.load(audio_path, sr=16000)
    total_duration = len(y) / sr
    print(f"   Durata totale: {total_duration:.2f} secondi")
    
    num_chunks = math.ceil(total_duration / CHUNK_SECONDS)

    for i in range(num_chunks):
        start_sec = i * CHUNK_SECONDS
        end_sec = min((i + 1) * CHUNK_SECONDS, total_duration)
        
        # Taglia l'audio
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        y_chunk = y[start_sample:end_sample]
        
        if len(y_chunk) == 0: continue

        # Calcoli feature sul chunk
        S = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        mfccs = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=20)

        # Plot dimensioni fisse
        plt.figure(figsize=(15, 12))

        # Subplot 1: Waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y_chunk, sr=sr, alpha=0.6, color="blue")
        plt.title(f"Waveform - Part {i+1}/{num_chunks} ({start_sec:.0f}s - {end_sec:.0f}s)", fontsize=14)
        plt.xlim(0, CHUNK_SECONDS) # Mantiene asse X fisso per confronto visivo
        plt.xlabel("")

        # Subplot 2: Mel Spectrogram
        plt.subplot(3, 1, 2)
        # Nota: x_axis='time' in librosa parte da 0 relativo al chunk
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='inferno')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel-Spectrogram", fontsize=14)
        plt.xlim(0, CHUNK_SECONDS)
        plt.xlabel("")

        # Subplot 3: MFCC
        plt.subplot(3, 1, 3)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='viridis')
        plt.colorbar()
        plt.title("MFCC", fontsize=14)
        plt.xlim(0, CHUNK_SECONDS)
        plt.xlabel("Time (seconds relative to chunk)")

        plt.tight_layout()
        
        out_file = OUTPUT_DIR / f"SPECTROGRAM_{TARGET_ID}_part{i+1:02d}.png"
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"   -> Salvato: {out_file.name}")

    print("✅ Generazione completata.")

if __name__ == "__main__":
    main()