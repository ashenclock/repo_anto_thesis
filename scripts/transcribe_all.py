import argparse
import sys
import os
import re
from pathlib import Path
from tqdm import tqdm
import torch

# Aggiunge la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config

# --- Funzioni Engine (Invariate nella logica, cambiano solo nel main) ---

def get_crisperwhisper_transcriber(config):
    try: from dotenv import load_dotenv; load_dotenv()
    except ImportError: pass
    try: from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError: sys.exit(1)

    cfg = config.transcription.crisperwhisper
    print(f"Caricamento CrisperWhisper: '{cfg.model_id}'...")
    device = config.device
    model = AutoModelForSpeechSeq2Seq.from_pretrained(cfg.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, chunk_length_s=30, batch_size=cfg.batch_size, torch_dtype=torch.float16, device=device)

    def process_batch(paths, output_dir):
        for audio_path in tqdm(paths, leave=False):
            subj_id = audio_path.parent.parent.name
            out_file = output_dir / f"{subj_id}.txt"
            if out_file.exists() and not config.transcription.overwrite: continue
            try:
                res = pipe(str(audio_path))
                out_file.write_text(res["text"].strip(), encoding='utf-8')
            except Exception as e: print(f"Errore {audio_path.name}: {e}")
    return process_batch

def get_nemo_transcriber(config):
    try: import nemo.collections.asr as nemo_asr
    except ImportError: sys.exit(1)
    cfg = config.transcription.nemo
    print(f"Caricamento NeMo: '{cfg.model_name}'...")
    backend = nemo_asr.models.ASRModel.from_pretrained(model_name=cfg.model_name)
    backend.to(torch.device(config.device))
    
    def process_batch(paths, output_dir):
        to_transcribe, out_files = [], []
        for p in paths:
            subj_id = p.parent.parent.name
            out_f = output_dir / f"{subj_id}.txt"
            if not out_f.exists() or config.transcription.overwrite:
                to_transcribe.append(str(p))
                out_files.append(out_f)
        if not to_transcribe: return
        try:
            results = backend.transcribe(audio=to_transcribe, batch_size=cfg.batch_size, verbose=False)
            transcriptions = results[0] if isinstance(results, tuple) else results
            for text_obj, out_f in zip(transcriptions, out_files):
                text = text_obj if isinstance(text_obj, str) else text_obj.text
                out_f.write_text(text.strip(), encoding='utf-8')
        except Exception as e: print(f"Errore batch NeMo: {e}")
    return process_batch

def get_whisperx_transcriber(config):
    try: import whisperx
    except ImportError: sys.exit(1)
    cfg = config.transcription.whisperx
    print(f"Caricamento WhisperX: '{cfg.model_name}'...")
    model = whisperx.load_model(cfg.model_name, config.device, compute_type=cfg.compute_type, language=cfg.language)
    
    def process_batch(paths, output_dir):
        for audio_path in tqdm(paths, leave=False):
            subj_id = audio_path.parent.parent.name
            out_file = output_dir / f"{subj_id}.txt"
            if out_file.exists() and not config.transcription.overwrite: continue
            try:
                audio = whisperx.load_audio(str(audio_path))
                result = model.transcribe(audio, batch_size=cfg.batch_size)
                text = " ".join([seg['text'].strip() for seg in result["segments"]])
                out_file.write_text(text, encoding='utf-8')
            except Exception as e: print(f"Errore {audio_path.name}: {e}")
    return process_batch

# ===================================================================
#                       MAIN LOGIC
# ===================================================================

def extract_task_name(filename):
    match = re.search(r"(Task_?\d+)", filename, re.IGNORECASE)
    if match:
        raw = match.group(1).replace("task", "Task").replace("_", "")
        if "_" not in raw: raw = raw.replace("Task", "Task_")
        return raw
    return "Unknown_Task"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    engine_name = config.transcription.engine.lower()
    
    # 1. Determina il nome della cartella del modello (es. "CrisperWhisper" o "parakeet-tdt...")
    model_folder_name = "Unknown_Model"
    if engine_name == "crisperwhisper":
        model_folder_name = config.transcription.crisperwhisper.model_id.split('/')[-1]
        transcriber = get_crisperwhisper_transcriber(config)
    elif engine_name == "nemo":
        model_folder_name = config.transcription.nemo.model_name.split('/')[-1]
        transcriber = get_nemo_transcriber(config)
    elif engine_name == "whisperx":
        model_folder_name = f"WhisperX_{config.transcription.whisperx.model_name}"
        transcriber = get_whisperx_transcriber(config)
    else:
        print("Engine non supportato."); return

    dataset_root = Path(config.data.dataset_root)
    # Cartella Base: data/transcripts/[NOME_MODELLO]
    base_transcripts_root = Path(config.data.transcripts_root) / model_folder_name
    base_transcripts_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Output Dir: {base_transcripts_root} ---")
    
    # 2. Scansiona e raggruppa per Task
    all_wavs = list(dataset_root.rglob("*.wav"))
    tasks_map = {}
    for wav in all_wavs:
        task_name = extract_task_name(wav.name)
        if task_name == "Unknown_Task": continue
        if task_name not in tasks_map: tasks_map[task_name] = []
        tasks_map[task_name].append(wav)
    
    # 3. Esegui
    for task, files in tasks_map.items():
        print(f"\nProcessing {task} ({len(files)} files)...")
        # Crea cartella specifica: data/transcripts/[NOME_MODELLO]/[NOME_TASK]
        task_out_dir = base_transcripts_root / task
        task_out_dir.mkdir(parents=True, exist_ok=True)
        transcriber(files, task_out_dir)
        
    print(f"\n✅ Trascrizione completata per modello: {model_folder_name}")

if __name__ == "__main__":
    main()