import os
import glob
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torchaudio.functional import resample

class BaseDataset(Dataset):
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.dataset_root = Path(config.data.dataset_root)
        self.label_mapping = config.labels.mapping

    def __len__(self):
        return len(self.df)

    def _get_label(self, diagnosis):
        return torch.tensor(self.label_mapping.get(diagnosis, 0), dtype=torch.long)

    def _get_subject_path(self, row):
        # Costruisce il percorso: data/dataset/[DIAGNOSI]/[SUBJECT_ID]
        return self.dataset_root / row['Diagnosis'] / row['Subject_ID']

class TextDataset(BaseDataset):
    def __init__(self, df, config, tokenizer):
        super().__init__(df, config)
        self.tokenizer = tokenizer
        # Punta alla cartella base delle trascrizioni definita nel config
        self.transcripts_root = Path(config.data.transcripts_root)
        # Recupera il nome del task corrente (es. Task_01)
        self.task_name = config.data.audio_file_pattern 

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subj_id = row['Subject_ID']
        
        # Costruisce il percorso: data/transcripts/WhisperX.../Task_01/SUBJ_0001.txt
        transcript_path = self.transcripts_root / self.task_name / f"{subj_id}.txt"
        
        text = ""
        if transcript_path.exists():
            text = transcript_path.read_text(encoding="utf-8").strip()
        else:
            # Fallback: prova a cercarlo senza la sottocartella Task se non lo trova
            fallback_path = self.transcripts_root / f"{subj_id}.txt"
            if fallback_path.exists():
                text = fallback_path.read_text(encoding="utf-8").strip()
            else:
                print(f"Warning: Trascrizione non trovata per {subj_id} in {transcript_path}")

        # Tokenizzazione per BERT
        inputs = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.config.model.text.max_length,
            padding="max_length", 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors="pt",
        )
        
        item = {
            "input_ids": inputs["input_ids"].flatten(), 
            "attention_mask": inputs["attention_mask"].flatten(), 
            "id": subj_id,
            "labels": self._get_label(row['Diagnosis'])
        }
        return item

class AudioDataset(BaseDataset):
    def __init__(self, df, config):
        super().__init__(df, config)
        self.target_sr = config.model.audio.sample_rate
        self.file_pattern = config.data.audio_file_pattern

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subj_path = self._get_subject_path(row)
        audio_folder = subj_path / "Audio"
        
        # Trova il file audio specifico (es. Task_01)
        # Cerca file .wav che contengono il pattern nel nome
        found_files = list(audio_folder.glob(f"*{self.file_pattern}*.wav"))
        
        if not found_files:
            # Fallback: prova a prendere il primo wav disponibile se il pattern fallisce
            found_files = list(audio_folder.glob("*.wav"))
            
        if found_files:
            audio_path = found_files[0] # Prendi il primo match
            waveform, sr = torchaudio.load(audio_path)
        else:
            print(f"Warning: Nessun file audio trovato per {row['Subject_ID']} in {audio_folder}")
            # Ritorna tensore vuoto/zero per evitare crash immediato
            waveform = torch.zeros(1, 16000) 
            sr = 16000

        # Forza Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sr != self.target_sr:
            waveform = resample(waveform, sr, self.target_sr)
        
        item = {
            "waveform": waveform.squeeze(0), 
            "id": row['Subject_ID'],
            "labels": self._get_label(row['Diagnosis'])
        }
        return item

def collate_audio(batch):
    waveforms = [item['waveform'] for item in batch]
    ids = [item['id'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    
    # Padding delle waveform
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
    return {
        "waveform": padded_waveforms, 
        "id": ids, 
        "labels": labels
    }

def get_data_splits(config):
    """
    Legge il file dataset_folds.csv e yielda i DataFrame di train e val 
    basati sulla colonna 'kfold'.
    """
    folds_path = config.data.folds_file
    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"File folds non trovato in: {folds_path}")
        
    df = pd.read_csv(folds_path)
    
    # Filtra solo le diagnosi presenti nel mapping del config
    valid_diagnoses = list(config.labels.mapping.keys())
    df = df[df['Diagnosis'].isin(valid_diagnoses)].reset_index(drop=True)
    
    # Itera sui fold definiti nel file CSV (0, 1, 2, 3, 4...)
    unique_folds = sorted(df['kfold'].unique())
    
    for val_fold_idx in unique_folds:
        # Split basato sulla colonna 'kfold' pre-calcolata
        train_df = df[df['kfold'] != val_fold_idx].copy()
        val_df = df[df['kfold'] == val_fold_idx].copy()
        
        # Il test set in questo setup è implicito nella Cross Validation (il fold di validazione è il test),
        # oppure, se hai un file test separato, puoi caricarlo qui. 
        # Per ora usiamo Val come Test per mantenere la struttura CV.
        
        print(f"Generazione Fold {val_fold_idx}: Train {len(train_df)} samples, Val {len(val_df)} samples")
        yield val_fold_idx, train_df, val_df

def get_dataloaders(config, train_df, val_df):
    if config.modality == 'text':
        tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
        train_dataset = TextDataset(train_df, config, tokenizer)
        val_dataset = TextDataset(val_df, config, tokenizer)
        collate_fn = None
    elif config.modality == 'audio':
        train_dataset = AudioDataset(train_df, config)
        val_dataset = AudioDataset(val_df, config)
        collate_fn = collate_audio
    else:
        raise ValueError(f"Modalità '{config.modality}' non supportata.")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader