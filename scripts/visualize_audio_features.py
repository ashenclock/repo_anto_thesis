import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

TARGET_ID = "EN_adrso234" 
SEARCH_ROOT = Path("data/dataset")
OUTPUT_DIR = Path("plots/attention_maps_full") # Cartella separata per i FULL

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = list(SEARCH_ROOT.rglob(f"*{TARGET_ID}*.wav"))
    if not candidates: print("Audio non trovato"); return
    
    audio_path = candidates[0]
    print(f"🔹 Analisi Spettrogramma FULL: {audio_path.name}")

    y, sr = librosa.load(audio_path, sr=16000)
    total_duration = len(y) / sr
    print(f"   Durata: {total_duration:.2f} secondi")
    
    # Calcoli completi
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Larghezza dinamica per matchare l'attention
    dynamic_width = max(20, int(total_duration * 2))
    plt.figure(figsize=(dynamic_width, 12))

    # Subplot 1: Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6, color="blue")
    plt.title(f"Waveform FULL - {TARGET_ID}", fontsize=14)
    plt.xlim(0, total_duration)
    plt.xlabel("")

    # Subplot 2: Mel Spectrogram
    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Spectrogram", fontsize=14)
    plt.xlim(0, total_duration)
    plt.xlabel("")

    # Subplot 3: MFCC
    plt.subplot(3, 1, 3)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='viridis')
    plt.colorbar()
    plt.title("MFCC", fontsize=14)
    plt.xlim(0, total_duration)
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"SPECTROGRAM_FULL_{TARGET_ID}.png", dpi=300)
    print(f"✅ Grafico salvato: {OUTPUT_DIR}/SPECTROGRAM_FULL_{TARGET_ID}.png")

if __name__ == "__main__":
    main()