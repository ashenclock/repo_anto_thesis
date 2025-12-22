import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from phonemizer import phonemize
from phonemizer.separator import Separator

# Aggiunge la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

def get_asr_folder_name(config_dict):
    """Helper per ottenere il nome della cartella del modello ASR dal config."""
    try:
        engine = config_dict['transcription']['engine'].lower()
        if engine == "whisperx":
            # Es: WhisperX_large-v3
            return f"WhisperX_{config_dict['transcription']['whisperx']['model_name']}"
        elif engine == "nemo":
            return config_dict['transcription']['nemo']['model_name'].split('/')[-1]
        elif engine == "crisperwhisper":
            return config_dict['transcription']['crisperwhisper']['model_id'].split('/')[-1]
    except KeyError:
        return "UnknownASR"
    return "UnknownASR"

def main():
    parser = argparse.ArgumentParser(description="Converte le trascrizioni testuali esistenti in fonemi.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    # 1. Determina i percorsi basandosi sul config
    # Es: "WhisperX_large-v3"
    asr_model_folder = get_asr_folder_name(config.to_dict())
    
    # Es: data/transcripts/WhisperX_large-v3
    base_transcripts_root = Path(config.data.transcripts_root) / asr_model_folder
    
    # Es: data/transcripts/WhisperX_large-v3_phonemes
    phonemes_root = base_transcripts_root.parent / f"{base_transcripts_root.name}_phonemes"
    
    print(f"--- Conversione da Testo a Fonemi (G2P) ---")
    print(f"Input:  {base_transcripts_root}")
    print(f"Output: {phonemes_root}")

    # 2. Scansiona le cartelle dei Task esistenti nella cartella delle trascrizioni
    if not base_transcripts_root.exists():
        sys.exit(f"ERRORE: La cartella delle trascrizioni non esiste: {base_transcripts_root}")
        
    task_dirs = [d for d in base_transcripts_root.iterdir() if d.is_dir() and "Task" in d.name]
    
    if not task_dirs:
        sys.exit(f"ERRORE: Nessuna cartella 'Task_XX' trovata in {base_transcripts_root}")

    # 3. Processa i file di testo trovati per ogni task
    for task_dir in tqdm(task_dirs, desc="Processing Tasks"):
        output_task_dir = phonemes_root / task_dir.name
        output_task_dir.mkdir(parents=True, exist_ok=True)
        
        files_to_process = list(task_dir.glob("*.txt"))
        if not files_to_process:
            continue
            
        texts = [f.read_text(encoding='utf-8').strip() for f in files_to_process]
        
        # Filtra testi vuoti per evitare errori con phonemizer
        valid_texts = [t for t in texts if t]
        
        if not valid_texts:
            continue
            
        phonemized_list = phonemize(
            valid_texts, 
            language='it', 
            backend='espeak',
            strip=True, 
            preserve_punctuation=True, 
            with_stress=False,
            separator=Separator(phone=" ", word=" | ", syllable="")
        )
        
        # Associa i testi fonetizzati ai file originali
        phonemized_map = dict(zip(valid_texts, phonemized_list))
        
        # Salva i file con i fonemi
        for f_path in files_to_process:
            original_text = f_path.read_text(encoding='utf-8').strip()
            # Se il testo originale è stato processato, salvalo
            if original_text in phonemized_map:
                out_file = output_task_dir / f_path.name
                out_file.write_text(phonemized_map[original_text], encoding='utf-8')

    print(f"\n✅ Conversione in fonemi completata. Controlla la cartella: {phonemes_root}")

if __name__ == "__main__":
    main()