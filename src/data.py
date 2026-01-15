import os
import pandas as pd
import torch
import re
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

class MultimodalDataset(Dataset):
    def __init__(self, df, config, tokenizer):
        self.df = df
        self.config = config
        self.tokenizer = tokenizer
        
        # Root base delle trascrizioni (es. data/transcripts/WhisperX...)
        self.transcripts_root = Path(config.data.transcripts_root)
        # Se nel config c'è un path che finisce con Task_XX, torniamo su di un livello
        if "Task_" in self.transcripts_root.name:
            self.transcripts_root = self.transcripts_root.parent

        # Root base delle feature audio (es. data/features/wav2vec2...)
        model_name_safe = config.model.audio.pretrained.split('/')[-1]
        self.audio_features_root = Path(config.data.features_root) / f"{model_name_safe}_sequences"
        
        if hasattr(config.labels.mapping, 'to_dict'):
            self.label_mapping = config.labels.mapping.to_dict()
        else:
            self.label_mapping = config.labels.mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Subject_ID nel metadata è tipo: SUBJ_001_Task_01
        full_id = row['Subject_ID'] 
        
        # Estraiamo l'ID pulito (SUBJ_001) e il Task (Task_01)
        # Assumiamo formato standard: XXXX_Task_YY
        parts = full_id.split('_Task_')
        if len(parts) == 2:
            clean_id = parts[0]
            task_name = f"Task_{parts[1]}"
        else:
            # Fallback se il formato è strano
            clean_id = full_id
            task_name = "Task_01"

        # 1. TESTO
        # Path: transcripts_root / Task_XX / SUBJ_001.txt
        transcript_path = self.transcripts_root / task_name / f"{clean_id}.txt"
        
        text = ""
        if transcript_path.exists():
            text = transcript_path.read_text(encoding="utf-8").strip()
        
        text_inputs = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.config.model.text.max_length,
            padding="max_length", 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors="pt"
        )

        # 2. AUDIO (Pre-computed tensors)
        # Path: features_root / Task_XX / SUBJ_001.pt
        tensor_path = self.audio_features_root / task_name / f"{clean_id}.pt"
        
        if tensor_path.exists():
            audio_features = torch.load(tensor_path, weights_only=True).float()
        else:
            # Se manca il file, crea tensore vuoto (o di zeri)
            # Dimensione dipende dal modello (1024 o 1280 o 768)
            dim = 1024 
            if "whisper" in str(self.audio_features_root): dim = 1280
            elif "base" in str(self.audio_features_root): dim = 768
            audio_features = torch.zeros(50, dim) # 50 frames fittizi

        return {
            "id": full_id, # Ritorniamo l'ID completo per tracciabilità
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
    
    # Padding dinamico audio
    audio_list = [item['audio_features'] for item in batch]
    audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_features": audio_padded,
        "labels": labels,
        "id": ids
    }

def get_data_splits_combined(config, selected_tasks=None):
    """
    Versione speciale per caricare PIÙ task contemporaneamente.
    Mantiene la separazione dei fold basata sui PAZIENTI per evitare data leakage.
    """
    # 1. Carica Metadata (tutti i file audio esistenti)
    meta_path = Path(config.data.metadata_file)
    df_meta = pd.read_csv(meta_path)
    
    # 2. Filtra per Task selezionati (es. ['Task_01', 'Task_02'])
    if selected_tasks and "ALL" not in selected_tasks:
        # Crea una regex o controlla la stringa
        # Subject_ID contiene _Task_01
        pattern = "|".join(selected_tasks)
        df_meta = df_meta[df_meta['Subject_ID'].str.contains(pattern)].copy()
    
    # 3. Carica Folds (assegnati per Paziente, non per file)
    folds_path = Path(config.data.folds_file)
    df_folds = pd.read_csv(folds_path)
    
    # Mappa Paziente -> Fold
    # Subject_ID nel fold file è solitamente "SUBJ_001" (senza task)
    id_col = 'Subject_ID' if 'Subject_ID' in df_folds.columns else 'ID'
    patient_to_fold = df_folds.set_index(id_col)['kfold'].to_dict()
    
    # 4. Assegna il fold a ogni riga del metadata
    def get_patient_id(full_id):
        return full_id.split('_Task_')[0]
    
    df_meta['patient_id'] = df_meta['Subject_ID'].apply(get_patient_id)
    df_meta['kfold'] = df_meta['patient_id'].map(patient_to_fold)
    
    # Rimuovi chi non ha un fold (magari esclusi dal dataset folds originale)
    df_meta = df_meta.dropna(subset=['kfold'])
    
    # Filtra Diagnosi
    mapping = config.labels.mapping.to_dict() if hasattr(config.labels.mapping, 'to_dict') else config.labels.mapping
    df_meta = df_meta[df_meta['Diagnosis'].isin(mapping.keys())].reset_index(drop=True)
    
    print(f"📊 Dataset COMBINATO ({selected_tasks}): {len(df_meta)} campioni totali.")
    
    unique_folds = sorted(df_meta['kfold'].unique())
    
    for fold_idx in unique_folds:
        train_df = df_meta[df_meta['kfold'] != fold_idx].copy()
        val_df = df_meta[df_meta['kfold'] == fold_idx].copy()
        yield fold_idx, train_df, val_df

def get_dataloaders(config, train_df, val_df):
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
    train_ds = MultimodalDataset(train_df, config, tokenizer)
    val_ds = MultimodalDataset(val_df, config, tokenizer)
    
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
        drop_last=False
    )
    
    return train_loader, val_loader