import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "data/metadata/adresso_FULL_mmse.csv"
OUTPUT_FILE = "data/metadata/adresso_FULL_MOCA_converted.csv"

def get_trzepacz_mapping():
    """
    Mappatura inversa basata su Trzepacz et al. (2015), BMC Geriatrics, Fig 2.
    La tabella originale mappa MoCA -> MMSE. Qui invertiamo MMSE -> MoCA.
    Quando un MMSE mappa più MoCA, prendiamo la media arrotondata.
    """
    return {
        30: 28, # MoCA 26-30 -> MMSE 30 (Media 28) - Qui sta il trucco per l'MCI!
        29: 25, # MoCA 24-25 -> MMSE 29 (Media 24.5 -> 25)
        28: 23, # MoCA 23 -> MMSE 28
        27: 22, # MoCA 21-22 -> MMSE 27
        26: 20, # MoCA 20 -> MMSE 26
        25: 19, # MoCA 19 -> MMSE 25
        24: 18, # MoCA 18 -> MMSE 24 (Cutoff classico)
        23: 17, # Interpolato (gap nella tabella)
        22: 17, # MoCA 17 -> MMSE 22
        21: 16, # MoCA 16 -> MMSE 21
        20: 15, # MoCA 15 -> MMSE 20
        19: 14, # MoCA 14 -> MMSE 19
        18: 13, # MoCA 13 -> MMSE 18
        17: 12, # MoCA 12 -> MMSE 17
        16: 11, # MoCA 11 -> MMSE 16
        15: 10, # MoCA 9-10 -> MMSE 15
        14: 8,  # MoCA 7-8 -> MMSE 14
        13: 6,  # MoCA 6 -> MMSE 13
        12: 5,  # MoCA 4-5 -> MMSE 12
        11: 3,  # MoCA 3 -> MMSE 11
        10: 2,  # MoCA 2 -> MMSE 10
        9: 1,   # MoCA 1 -> MMSE 9
        # Sotto il 9 scendiamo a 0 rapidamente
    }

def convert_score(mmse):
    mapping = get_trzepacz_mapping()
    mmse_int = int(round(float(mmse)))
    
    if mmse_int in mapping:
        return mapping[mmse_int]
    
    # Fallback per valori molto bassi (<6)
    if mmse_int < 6: return 0
    return mmse_int # Fallback generico

def main():
    print(f"🔄 Conversione MMSE -> MoCA (Trzepacz 2015 - PDF Fig.2)...")
    
    if not Path(INPUT_FILE).exists():
        print(f"❌ Errore: Manca il file {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Applica conversione
    df['Score_MMSE_Original'] = df['Score']
    df['Score'] = df['Score_MMSE_Original'].apply(convert_score)
    
    print("\n   Esempi di conversione (MMSE -> MoCA):")
    print(df[['Score_MMSE_Original', 'Score']].head(15))
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ File convertito salvato in: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()