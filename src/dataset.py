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
        if hasattr(config.labels.mapping, 'to_dict'):
            self.label_mapping = config.labels.mapping.to_dict()
        else:
            self.label_mapping = config.labels.mapping

    def __len__(self): return len(self.df)
    def _get_label(self, diagnosis):
        return torch.tensor(self.label_mapping.get(diagnosis, 0), dtype=torch.long)
    def _get_subject_path(self, row):
        return self.dataset_root / row['Diagnosis'] / row['Subject_ID']

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

        inputs = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.config.model.text.max_length,
            padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt",
        )
        return {"input_ids": inputs["input_ids"].flatten(), "attention_mask": inputs["attention_mask"].flatten(), "id": subj_id, "labels": self._get_label(row['Diagnosis'])}

class AudioDataset(BaseDataset):
    def __init__(self, df, config):
        super().__init__(df, config)
        self.target_sr = config.model.audio.sample_rate
        self.file_pattern = config.data.audio_file_pattern

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subj_path = self._get_subject_path(row)
        audio_folder = subj_path / "Audio"
        found_files = list(audio_folder.glob(f"*{self.file_pattern}*.wav")) or list(audio_folder.glob("*.wav"))
        
        if found_files:
            try:
                waveform, sr = torchaudio.load(found_files[0])
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != self.target_sr: waveform = resample(waveform, sr, self.target_sr)
                
                # --- FIX: TAGLIO AUDIO ---
                if hasattr(self.config.model.audio, 'max_duration'):
                    max_len = int(self.config.model.audio.max_duration * self.target_sr)
                    if waveform.shape[1] > max_len: waveform = waveform[:, :max_len]
                # -------------------------
            except: waveform = torch.zeros(1, self.target_sr)
        else: waveform = torch.zeros(1, self.target_sr)

        return {"waveform": waveform.squeeze(0), "id": row['Subject_ID'], "labels": self._get_label(row['Diagnosis'])}

def collate_audio(batch):
    waveforms = pad_sequence([item['waveform'] for item in batch], batch_first=True, padding_value=0.0)
    labels = torch.stack([item['labels'] for item in batch])
    ids = [item['id'] for item in batch]
    return {"waveform": waveforms, "id": ids, "labels": labels}

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
        text = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.exists() else ""
        text_inputs = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.config.model.text.max_length,
            padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt"
        )

        # Audio
        subj_path = self._get_subject_path(row)
        audio_folder = subj_path / "Audio"
        found_files = list(audio_folder.glob(f"*{self.file_pattern}*.wav")) or list(audio_folder.glob("*.wav"))
        
        if found_files:
            try:
                waveform, sr = torchaudio.load(found_files[0])
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != self.target_sr: waveform = resample(waveform, sr, self.target_sr)
                
                # --- FIX: TAGLIO AUDIO ---
                if hasattr(self.config.model.audio, 'max_duration'):
                    max_len = int(self.config.model.audio.max_duration * self.target_sr)
                    if waveform.shape[1] > max_len: waveform = waveform[:, :max_len]
                # -------------------------
            except: waveform = torch.zeros(1, self.target_sr)
        else: waveform = torch.zeros(1, self.target_sr)

        return {
            "id": subj_id, "labels": label,
            "input_ids": text_inputs["input_ids"].flatten(),
            "attention_mask": text_inputs["attention_mask"].flatten(),
            "waveform": waveform.squeeze(0)
        }

def collate_multimodal(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    waveforms = pad_sequence([item['waveform'] for item in batch], batch_first=True, padding_value=0.0)
    labels = torch.stack([item['labels'] for item in batch])
    ids = [item['id'] for item in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "waveform": waveforms, "labels": labels, "id": ids}

def get_data_splits(config):
    df = pd.read_csv(config.data.folds_file)
    mapping = config.labels.mapping.to_dict() if hasattr(config.labels.mapping, 'to_dict') else config.labels.mapping
    df = df[df['Diagnosis'].isin(mapping.keys())].reset_index(drop=True)
    for fold in sorted(df['kfold'].unique()):
        yield fold, df[df['kfold'] != fold].copy(), df[df['kfold'] == fold].copy()

def get_dataloaders(config, train_df, val_df):
    if config.modality == 'text':
        tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
        train_ds, val_ds = TextDataset(train_df, config, tokenizer), TextDataset(val_df, config, tokenizer)
        collate_fn = None
    elif config.modality == 'audio':
        train_ds, val_ds = AudioDataset(train_df, config), AudioDataset(val_df, config)
        collate_fn = collate_audio
    elif config.modality in ['multimodal', 'multimodal_cross_attention']:
        tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
        train_ds, val_ds = MultimodalDataset(train_df, config, tokenizer), MultimodalDataset(val_df, config, tokenizer)
        collate_fn = collate_multimodal
    else: raise ValueError(f"Modalità '{config.modality}' non supportata.")

    return DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2), \
           DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)