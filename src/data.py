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

        # Base root for transcripts (e.g. data/transcripts/WhisperX...)
        self.transcripts_root = Path(config.data.transcripts_root)
        # If the config path ends with Task_XX, move up one level
        if "Task_" in self.transcripts_root.name:
            self.transcripts_root = self.transcripts_root.parent

        # Base root for audio features (e.g. data/features/wav2vec2...)
        model_name_safe = config.model.audio.pretrained.split('/')[-1]
        self.audio_features_root = (
            Path(config.data.features_root) / f"{model_name_safe}_sequences"
        )

        if hasattr(config.labels.mapping, 'to_dict'):
            self.label_mapping = config.labels.mapping.to_dict()
        else:
            self.label_mapping = config.labels.mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Subject_ID in the metadata has the form: SUBJ_001_Task_01
        full_id = row['Subject_ID']

        # Extract the clean ID (SUBJ_001) and the task (Task_01)
        # Assumes standard format: XXXX_Task_YY
        parts = full_id.split('_Task_')
        if len(parts) == 2:
            clean_id = parts[0]
            task_name = f"Task_{parts[1]}"
        else:
            # Fallback for unexpected format
            clean_id = full_id
            task_name = "Task_01"

        # 1. TEXT
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
            return_tensors="pt",
        )

        # 2. AUDIO (load .pt file)
        tensor_path = self.audio_features_root / task_name / f"{clean_id}.pt"

        if tensor_path.exists():
            audio_features = torch.load(tensor_path, weights_only=True).float()
        else:
            # Fallback when the tensor file is missing
            dim = 1024 if "large" in str(self.audio_features_root) else 768
            audio_features = torch.zeros(50, dim)

        # 3. TARGET (Regression vs Classification)
        task_type = self.config.get('task', 'classification')
        # Safe default initialization
        target = torch.tensor(0, dtype=torch.long)

        if task_type == "regression":
            # Regression on MoCA/MMSE scores
            score = row.get('mmse_score', row.get('Score', -1.0))
            target = torch.tensor(float(score), dtype=torch.float)
        else:
            # Classification
            diag = row.get('Diagnosis', 'CTR')
            target = torch.tensor(self.label_mapping.get(diag, 0), dtype=torch.long)

        return {
            "id": full_id,  # Return the full ID for traceability
            "input_ids": text_inputs["input_ids"].flatten(),
            "attention_mask": text_inputs["attention_mask"].flatten(),
            "audio_features": audio_features,
            "labels": target,
        }


def collate_multimodal(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    ids = [item['id'] for item in batch]

    # Dynamic audio padding
    audio_list = [item['audio_features'] for item in batch]
    audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_features": audio_padded,
        "labels": labels,
        "id": ids,
    }


def get_data_splits_combined(config, selected_tasks=None):
    """
    Special version for loading multiple tasks simultaneously.
    Maintains fold separation based on PATIENTS to avoid data leakage.
    """
    # 1. Load metadata (all existing audio files)
    meta_path = Path(config.data.metadata_file)
    df_meta = pd.read_csv(meta_path)

    # 2. Filter by selected tasks (e.g. ['Task_01', 'Task_02'])
    if selected_tasks and "ALL" not in selected_tasks:
        pattern = "|".join(selected_tasks)
        df_meta = df_meta[df_meta['Subject_ID'].str.contains(pattern)].copy()

    # 3. Load folds (assigned per patient, not per file)
    folds_path = Path(config.data.folds_file)
    df_folds = pd.read_csv(folds_path)

    # Map Patient -> Fold
    # Subject_ID in the fold file is typically "SUBJ_001" (without task)
    id_col = 'Subject_ID' if 'Subject_ID' in df_folds.columns else 'ID'
    patient_to_fold = df_folds.set_index(id_col)['kfold'].to_dict()

    # 4. Assign the fold to each row in the metadata
    def get_patient_id(full_id):
        return full_id.split('_Task_')[0]

    df_meta['patient_id'] = df_meta['Subject_ID'].apply(get_patient_id)
    df_meta['kfold'] = df_meta['patient_id'].map(patient_to_fold)

    # Remove entries without a fold (possibly excluded from the original folds dataset)
    df_meta = df_meta.dropna(subset=['kfold'])

    # Filter by diagnosis
    mapping = (
        config.labels.mapping.to_dict()
        if hasattr(config.labels.mapping, 'to_dict')
        else config.labels.mapping
    )
    df_meta = df_meta[df_meta['Diagnosis'].isin(mapping.keys())].reset_index(drop=True)

    print(f"COMBINED dataset ({selected_tasks}): {len(df_meta)} total samples.")

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
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_multimodal,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
