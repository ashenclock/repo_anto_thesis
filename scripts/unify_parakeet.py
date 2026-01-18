import shutil
from pathlib import Path
import os

# Configurazione
DEST_DIR = Path("data/transcripts/Parakeet_Unified/Task_01")
IT_SRC = Path("data/transcripts/parakeet-tdt-0.6b-v3/Task_01") # IT v3
EN_SRC = Path("data/transcripts/parakeet-tdt-0.6b-v2/Task_01") # EN v2 (o dove sono finiti dopo import)

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Copia IT (v3)
    if IT_SRC.exists():
        print(f"Copia IT da {IT_SRC}...")
        for f in IT_SRC.glob("*.txt"):
            shutil.copy2(f, DEST_DIR / f.name)
    
    # 2. Copia EN (v2)
    # Nota: Se dopo l'import sono finiti in un'altra cartella, aggiusta EN_SRC
    if EN_SRC.exists():
        print(f"Copia EN da {EN_SRC}...")
        for f in EN_SRC.glob("*.txt"):
            shutil.copy2(f, DEST_DIR / f.name)
            
    print(f"Fatto. Ora usa: {DEST_DIR.parent} nel config.")

if __name__ == "__main__":
    main()