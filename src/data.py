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

# ===================================================================
# 1. CLASSE BASE
# ===================================================================
class BaseDataset(Dataset):
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.dataset_root = Path(config.data.dataset_root)
        
        if hasattr(config.labels.mapping, 'to_dict'):
            self.label_mapping = config.labels.mapping.to_dict()
        else:
            self.label_mapping = config.labels.mapping

    def __len__(self):
        return len(self.df)

    def _get_label(self, diagnosis):
        return torch.tensor(self.label_mapping.get(diagnosis, 0), dtype=torch.long)

    def _get_subject_path(self, row):
        return self.dataset_root / row['Diagnosis'] / row['Subject_ID']

# ===================================================================
# 2. DATASET TESTUALE (PULITO)
# ===================================================================
class TextDataset(BaseDataset):
    def __init__(self, df, config, tokenizer):
        super().__init__(df, config)
        self.tokenizer = tokenizer
        self.transcripts_root = Path(config.data.transcripts_root)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subj_id = row['Subject_ID']
        
        transcript_path = self.transcripts_root / f"{subj_id}.txt"
        
        text = ""
        if transcript_path.exists():
            text = transcript_path.read_text(encoding="utf-8").strip()

        # Tokenizzazione
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

# ===================================================================
# 3. DATASET AUDIO
# ===================================================================
class AudioDataset(BaseDataset):
    def __init__(self, df, config):
        super().__init__(df, config)
        self.target_sr = config.model.audio.sample_rate
        self.file_pattern = config.data.audio_file_pattern

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subj_path = self._get_subject_path(row)
        audio_folder = subj_path / "Audio"
        
        found_files = list(audio_folder.glob(f"*{self.file_pattern}*.wav"))
        
        if not found_files:
            found_files = list(audio_folder.glob("*.wav"))
            
        if found_files:
            try:
                waveform, sr = torchaudio.load(found_files[0])
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != self.target_sr:
                    waveform = resample(waveform, sr, self.target_sr)
            except Exception:
                waveform = torch.zeros(1, self.target_sr)
        else:
            waveform = torch.zeros(1, self.target_sr)

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
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    return {"waveform": padded_waveforms, "id": ids, "labels": labels}

# ===================================================================
# 4. DATASET MULTIMODALE (PULITO)
# ===================================================================
class MultimodalDataset(BaseDataset):
    def __init__(self, df, config, tokenizer):
        super().__init__(df, config)
        self.tokenizer = tokenizer
        self.transcripts_root = Path(config.data.transcripts_root)
        self.target_sr = config.model.audio.sample_rate
        self.file_pattern = config.data.audio_file_pattern

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subj_id = row['Subject_ID']
        label = self._get_label(row['Diagnosis'])
        
        # Testo
        transcript_path = self.transcripts_root / f"{subj_id}.txt"
        text = ""
        if transcript_path.exists():
            text = transcript_path.read_text(encoding="utf-8").strip()
        
        text_inputs = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.config.model.text.max_length,
            padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt"
        )

        # Audio
        subj_path = self._get_subject_path(row)
        audio_folder = subj_path / "Audio"
        found_files = list(audio_folder.glob(f"*{self.file_pattern}*.wav"))
        
        if found_files:
            try:
                waveform, sr = torchaudio.load(found_files[0])
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != self.target_sr: waveform = resample(waveform, sr, self.target_sr)
            except:
                waveform = torch.zeros(1, self.target_sr)
        else:
            waveform = torch.zeros(1, self.target_sr)

        return {
            "id": subj_id, "labels": label,
            "input_ids": text_inputs["input_ids"].flatten(),
            "attention_mask": text_inputs["attention_mask"].flatten(),
            "waveform": waveform.squeeze(0)
        }

def collate_multimodal(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    waveforms = [item['waveform'] for item in batch]
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    labels = torch.stack([item['labels'] for item in batch])
    ids = [item['id'] for item in batch]
    
    return {
        "input_ids": input_ids, "attention_mask": attention_mask, 
        "waveform": padded_waveforms, "labels": labels, "id": ids
    }

# ===================================================================
# 5. SPLIT E LOADER
# ===================================================================

def get_data_splits(config):
    folds_path = config.data.folds_file
    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"File folds non trovato in: {folds_path}")
    
    df = pd.read_csv(folds_path)
    
    if hasattr(config.labels.mapping, 'to_dict'):
        mapping_dict = config.labels.mapping.to_dict()
    else:
        mapping_dict = config.labels.mapping
        
    valid_diagnoses = list(mapping_dict.keys())
    df = df[df['Diagnosis'].isin(valid_diagnoses)].reset_index(drop=True)
    
    unique_folds = sorted(df['kfold'].unique())
    
    for val_fold_idx in unique_folds:
        train_df = df[df['kfold'] != val_fold_idx].copy()
        val_df = df[df['kfold'] == val_fold_idx].copy()
        yield val_fold_idx, train_df, val_df

def get_dataloaders(config, train_df, val_df):
    if config.modality == 'text':
        tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
        train_ds = TextDataset(train_df, config, tokenizer)
        val_ds = TextDataset(val_df, config, tokenizer)
        collate_fn = None
    elif config.modality == 'audio':
        train_ds = AudioDataset(train_df, config)
        val_ds = AudioDataset(val_df, config)
        collate_fn = collate_audio
    elif config.modality == 'multimodal':
        tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
        train_ds = MultimodalDataset(train_df, config, tokenizer)
        val_ds = MultimodalDataset(val_df, config, tokenizer)
        collate_fn = collate_multimodal
    else:
        raise ValueError(f"Modalità '{config.modality}' non supportata.")

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader