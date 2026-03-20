import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path

def load_pro_segment(path, offset=5, duration=10):
    """Carica l'audio saltando il setup iniziale e pulendo il segnale"""
    # Carichiamo con un offset per saltare il setup
    y, sr = librosa.load(path, offset=offset, duration=duration)
    
    # Rimuoviamo eventuale silenzio residuo all'inizio del segmento caricato
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # Se il trim lo accorcia troppo, torniamo all'originale per mantenere i 10s
    if len(y_trimmed) < sr * 2: 
        y_trimmed = y
        
    # Pre-emphasis per far risaltare le alte frequenze (formanti)
    y_pre = librosa.effects.preemphasis(y_trimmed)
    
    S = librosa.feature.melspectrogram(y=y_pre, sr=sr, n_mels=128, fmax=8000)
    return librosa.power_to_db(S, ref=np.max), sr

def main():
    # Selezioniamo i file (Assicurati che i path siano corretti per il tuo sistema)
    try:
        ctr_file = glob.glob("data/dataset/CTR/SUBJ_0006/Audio/*Task_01*.wav")[0]
        ad_file = glob.glob("data/dataset/MILD-AD/SUBJ_0033/Audio/*Task_01*.wav")[0]
    except IndexError:
        print("❌ File non trovati. Controlla i Subject_ID.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    
    # --- PLOT CTR (Sano) ---
    # Saltiamo 5 secondi per evitare il "setup"
    S_ctr, sr = load_pro_segment(ctr_file, offset=5, duration=10)
    img1 = librosa.display.specshow(S_ctr, sr=sr, x_axis='time', y_axis='mel', ax=ax1, cmap='magma')
    ax1.set_title("CONTROLLO SANO (CTR): Flusso Mel-Spettrale Regolare", fontsize=15, fontweight='bold', pad=10)
    ax1.set_ylabel("Frequenza (Hz)")
    
    # --- PLOT AD (Alzheimer) ---
    # Saltiamo 5 secondi anche qui
    S_ad, sr = load_pro_segment(ad_file, offset=5, duration=10)
    img2 = librosa.display.specshow(S_ad, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
    ax2.set_title("PAZIENTE ALZHEIMER (AD): Discontinuità e Frammentazione", fontsize=15, fontweight='bold', pad=10)
    ax2.set_ylabel("Frequenza (Hz)")
    ax2.set_xlabel("Tempo (secondi di parlato effettivo)")
    
    # Annotazioni Professionali
    ax1.annotate('Pattern Armonico Stabile', xy=(2, 2000), xytext=(3, 5000),
                 arrowprops=dict(facecolor='white', shrink=0.05, width=1), color='white', weight='bold')
    
    ax2.annotate('Pausa Patologica (Silenzio)', xy=(4, 1000), xytext=(5, 4000),
                 arrowprops=dict(facecolor='cyan', shrink=0.05, width=1), color='cyan', weight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig("images/final_clinical_comparison.png", dpi=300)
    print("🚀 Spettrogrammi 'puliti' salvati in: images/final_clinical_comparison.png")

if __name__ == "__main__":
    main()