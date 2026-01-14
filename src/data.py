import os
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

class MultimodalDataset(Dataset):
    def __init__(self, df, config, tokenizer):
        self.df = df
        self.config = config
        self.tokenizer = tokenizer
        
        # Percorsi dinamici
        self.transcripts_root = Path(config.data.transcripts_root)
        
        # Costruzione path audio precomputato: data/features/MODELS_sequences/TASK_XX
        model_name_safe = config.model.audio.pretrained.split('/')[-1]
        self.precomputed_root = Path(config.data.features_root) / f"{model_name_safe}_sequences" / config.data.audio_file_pattern
        
        if hasattr(config.labels.mapping, 'to_dict'):
            self.label_mapping = config.labels.mapping.to_dict()
        else:
            self.label_mapping = config.labels.mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Gestione flessibile della colonna ID (Subject_ID o ID)
        subj_id = row['Subject_ID'] if 'Subject_ID' in row else row['ID']
        
        # Rimuove eventuali suffissi come _Task_01 per trovare il file .txt o .pt corretto
        clean_id = subj_id.split('_Task_')[0]

        # 1. TESTO
        transcript_path = self.transcripts_root / f"{clean_id}.txt"
        text = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.exists() else ""
        
        text_inputs = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.config.model.text.max_length,
            padding="max_length", 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors="pt"
        )

        # 2. AUDIO (FP32)
        tensor_path = self.precomputed_root / f"{clean_id}.pt"
        if tensor_path.exists():
            # weights_only=True risolve il FutureWarning di PyTorch
            audio_features = torch.load(tensor_path, weights_only=True).float()
        else:
            # Fallback dimensionale basato sul modello
            dim = 1024 if "large" in self.config.model.audio.pretrained or "xls-r" in self.config.model.audio.pretrained else 768
            audio_features = torch.zeros(10, dim)

        return {
            "id": clean_id,
            "input_ids": text_inputs["input_ids"].flatten(),
            "attention_mask": text_inputs["attention_mask"].flatten(),
            "audio_features": audio_features,
            "labels": torch.tensor(self.label_mapping.get(row['Diagnosis'], 0), dtype=torch.long)
        }

def collate_multimodal(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    ids = [item['id'] for item in batch]
    
    # Padding dinamico per l'audio (le sequenze hanno lunghezze temporali diverse)
    audio_list = [item['audio_features'] for item in batch]
    audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_features": audio_padded,
        "labels": labels,
        "id": ids
    }

def get_data_splits(config):
    """
    Legge il file dei fold e genera i set di train e validation.
    ASSUNZIONE: Il file CSV deve avere le colonne 'Diagnosis', 'kfold' e 'Subject_ID'.
    """
    folds_path = Path(config.data.folds_file)
    if not folds_path.exists():
        raise FileNotFoundError(f"File folds non trovato in: {folds_path.absolute()}")
    
    df = pd.read_csv(folds_path)
    
    # Mapping etichette dal config
    mapping = config.labels.mapping.to_dict() if hasattr(config.labels.mapping, 'to_dict') else config.labels.mapping
    valid_diagnoses = list(mapping.keys())
    
    # Filtriamo solo le diagnosi che ci interessano (es. CTR e MILD-AD)
    df = df[df['Diagnosis'].isin(valid_diagnoses)].reset_index(drop=True)
    
    unique_folds = sorted(df['kfold'].unique())
    print(f"📊 Dataset filtrato: {len(df)} campioni, Diagnosi: {valid_diagnoses}, Fold trovati: {unique_folds}")

    for fold_idx in unique_folds:
        train_df = df[df['kfold'] != fold_idx].copy()
        val_df = df[df['kfold'] == fold_idx].copy()
        yield fold_idx, train_df, val_df
def get_dataloaders(config, train_df, val_df):
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    train_ds = MultimodalDataset(train_df, config, tokenizer)
    val_ds = MultimodalDataset(val_df, config, tokenizer)
    
    # drop_last=True risolve il crash della BatchNorm su batch size 1
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        collate_fn=collate_multimodal,
        num_workers=4,
        pin_memory=True,
        drop_last=True 
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        collate_fn=collate_multimodal,
        num_workers=4,
        pin_memory=True,
        drop_last=False # In validazione non serve drop_last se il modello è in .eval()
    )
    
    return train_loader, val_loader