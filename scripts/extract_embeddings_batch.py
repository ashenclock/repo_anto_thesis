import argparse
import sys
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import torch
from transformers import AutoProcessor, AutoModel
import torchaudio
from torchaudio.functional import resample
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.utils import clear_memory

warnings.filterwarnings("ignore")

class AudioFileDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path_str = self.file_paths[idx]
        
        if pd.isna(audio_path_str):
             return {"waveform": torch.tensor([]), "id": "error", "task": "error"}
             
        audio_path = Path(audio_path_str)
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0)
            else: waveform = waveform.squeeze(0)
            if sr != 16000: waveform = resample(waveform, sr, 16000)
            
            task_match = re.search(r'(Task_\d+)', audio_path.name, re.IGNORECASE)
            task_name = task_match.group(1) if task_match else "Unknown"
            
            subj_id_match = re.search(r'(SUBJ_\d+)', audio_path.name)
            subj_id = subj_id_match.group(1) if subj_id_match else "Unknown"
            
            return {"waveform": waveform, "id": subj_id, "task": task_name}
        except Exception as e:
            print(f"Errore caricamento {audio_path.name}: {e}", file=sys.stderr)
            return {"waveform": torch.tensor([]), "id": "error", "task": "error"}

def collate_fn(batch):
    batch = [item for item in batch if item['id'] != "error"]
    if not batch: return None
    waveforms = [item['waveform'].numpy() for item in batch]
    ids = [item['id'] for item in batch]
    tasks = [item['task'] for item in batch]
    return {"waveforms": waveforms, "ids": ids, "tasks": tasks}

class EmbeddingExtractor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.model_id = config.embedding_extraction.model_id
        self.pooling = config.embedding_extraction.pooling_strategy
        self.batch_size = config.embedding_extraction.batch_size
        print(f"Caricamento modello '{self.model_id}' su {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def process(self, audio_paths_by_task, features_root, feature_set_name):
        for task, files in audio_paths_by_task.items():
            out_name = f"train_{feature_set_name}_{task}.csv"
            out_path = features_root / out_name
            if out_path.exists() and not self.config.feature_extraction.overwrite:
                print(f"File {out_name} esistente, salto.")
                continue

            print(f"\nProcessing {task} ({len(files)} files)...")
            dataset = AudioFileDataset(files)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=min(4, cpu_count()), collate_fn=collate_fn)
            
            all_embeddings = []
            all_ids = []
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Extracting {task}"):
                    if batch is None: continue
                    inputs = self.processor(batch['waveforms'], sampling_rate=16000, return_tensors="pt", padding="max_length")
                    input_features = inputs.input_features.to(self.device)
                    last_hidden_state = self.model.encoder(input_features).last_hidden_state
                    embeddings = last_hidden_state.mean(dim=1) if self.pooling == "mean" else last_hidden_state[:, 0, :]
                    
                    all_embeddings.append(embeddings.cpu().numpy())
                    all_ids.extend(batch['ids'])

            if not all_ids: continue
            
            embeddings_matrix = np.vstack(all_embeddings)
            feature_names = [f"emb_{i}" for i in range(embeddings_matrix.shape[1])]
            df = pd.DataFrame(embeddings_matrix, columns=feature_names)
            df.insert(0, 'ID', all_ids)
            
            df.to_csv(out_path, index=False)
            print(f"✅ Salvato: {out_path} ({len(df)} samples)")

def main():
    parser = argparse.ArgumentParser(description="Estrae embedding audio per tutti i task.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    if not hasattr(config, 'embedding_extraction'):
        sys.exit("ERRORE: Manca la sezione 'embedding_extraction' nel config.yaml")
        
    df_meta = pd.read_csv(Path(config.data.metadata_file))
    df_meta = df_meta[df_meta['has_audio'] == True].copy()
    
    # Raggruppa i percorsi per task
    audio_paths_by_task = {}
    df_meta['task'] = df_meta['audio_path'].str.extract(r'(Task_\d+)', expand=False)
    df_meta.dropna(subset=['task'], inplace=True)
    
    for task_name, group in df_meta.groupby('task'):
        audio_paths_by_task[task_name] = group['audio_path'].tolist()

    clear_memory()
    extractor = EmbeddingExtractor(config)
    
    features_root = Path(config.data.features_root)
    features_root.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Inizio Estrazione Embeddings ---")
    extractor.process(audio_paths_by_task, features_root, config.feature_extraction.feature_set)
    print("\nEstrazione completata.")

if __name__ == "__main__":
    main()