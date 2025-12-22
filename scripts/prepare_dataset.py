import argparse # <-- L'IMPORT MANCANTE
import sys
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from collections import defaultdict

# Aggiunge la root del progetto al path per importare config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

# ================= CONFIGURAZIONE =================
MIC_PRIORITY = ["Shure", "ATR", "WO_Mic", "Realtek"]

# ================= VAD & SNR (OPZIONALE) =================
def calculate_snr(waveform, speech_timestamps):
    if not speech_timestamps: return -10.0
    if waveform.dim() > 1: waveform = waveform.squeeze()
    mask = torch.zeros_like(waveform, dtype=torch.bool)
    for segment in speech_timestamps:
        mask[int(segment['start']):min(int(segment['end']), len(waveform))] = True
    speech_frames = waveform[mask]
    if len(speech_frames) == 0: return -10.0
    signal_power = torch.mean(speech_frames ** 2)
    noise_frames = waveform[~mask]
    noise_power = torch.mean(noise_frames ** 2) if len(noise_frames) > 0 else torch.tensor(1e-9)
    if noise_power == 0: noise_power = torch.tensor(1e-9)
    return 10 * torch.log10(signal_power / noise_power).item()

def analyze_file(file_path, vad_model, utils):
    (get_speech_timestamps, _, read_audio, _, _) = utils
    try:
        wav = read_audio(str(file_path)).to(vad_model.device)
        speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000, threshold=0.4)
        if not speech_timestamps: return 0.0, -10.0
        duration_sec = sum([i['end'] - i['start'] for i in speech_timestamps]) / 16000
        snr = calculate_snr(wav, speech_timestamps)
        return duration_sec, snr
    except Exception:
        return 0.0, -10.0

# ================= FUNZIONI PRINCIPALI =================

def extract_task_name(filename):
    match = re.search(r"(Task_?\d+)", filename, re.IGNORECASE)
    if match:
        raw = match.group(1).replace("task", "Task").replace("_", "")
        if "_" not in raw: raw = raw.replace("Task", "Task_")
        return raw
    return "Unknown"

def select_best_mic_for_task(files_for_task):
    if not files_for_task: return None
    for mic in MIC_PRIORITY:
        for f in files_for_task:
            if mic.lower() in f.name.lower():
                return f
    return files_for_task[0]

def main():
    parser = argparse.ArgumentParser(description="Prepara il file metadata per tutti i task audio.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--no-vad", action='store_true', help="Salta l'analisi VAD/SNR per velocità.")
    args = parser.parse_args()
    
    config = load_config(args.config)

    vad_model, utils = None, None
    if not args.no_vad:
        try:
            vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
            vad_model.to(config.device)
            print("VAD model caricato.")
        except Exception as e:
            print(f"Warning: Impossibile caricare VAD ({e}). Continuo senza analisi SNR/durata.")
            args.no_vad = True

    folds_df = pd.read_csv(Path(config.data.folds_file))
    if 'Subject_ID' in folds_df.columns:
        folds_df = folds_df.rename(columns={'Subject_ID': 'ID'})
    
    subject_info = folds_df.set_index('ID').to_dict('index')
    
    all_audio_files = list(Path(config.data.dataset_root).rglob("*.wav"))
    
    grouped_files = defaultdict(list)
    for f in all_audio_files:
        subj_id_match = re.search(r'(SUBJ_\d+)', f.name)
        task_name = extract_task_name(f.name)
        if subj_id_match and task_name != "Unknown":
            subj_id = subj_id_match.group(1)
            grouped_files[(subj_id, task_name)].append(f)
            
    print(f"Trovati {len(all_audio_files)} file, raggruppati in {len(grouped_files)} coppie (Soggetto, Task).")
    
    metadata = []
    
    for (subj_id, task_name), files in tqdm(grouped_files.items(), desc="Analizzando file..."):
        best_path = select_best_mic_for_task(files)
        
        info = subject_info.get(subj_id)
        if not info: continue
        
        duration, snr = 0.0, 0.0
        if not args.no_vad:
            duration, snr = analyze_file(best_path, vad_model, utils)

        metadata.append({
            'Subject_ID': f"{subj_id}_{task_name}",
            'Diagnosis': info['Diagnosis'],
            'kfold': info['kfold'],
            'audio_path': str(best_path),
            'duration': duration,
            'snr': snr,
            'has_audio': True
        })

    df_final = pd.DataFrame(metadata)
    df_final = df_final.sort_values(by=['Subject_ID']).reset_index(drop=True)
    
    output_path = Path(config.data.metadata_file)
    df_final.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print(f"✅ Creato nuovo file: {output_path}")
    print(f"   -> Righe totali (soggetti x task): {len(df_final)}")
    print("\nDistribuzione Task:")
    print(df_final['Subject_ID'].str.extract(r'(Task_\d+)')[0].value_counts())
    print("="*50)

if __name__ == "__main__":
    main()