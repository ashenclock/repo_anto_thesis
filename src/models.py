# FILE: ./src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# ==============================================================================
# 1. BLOCCHI COMUNI (Pooling & Gating)
# ==============================================================================

class AttentivePooling(nn.Module):
    """Pooling pesato: impara quali parti della sequenza sono importanti."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        # x: [Batch, Seq, Dim]
        scores = self.attn(x).squeeze(-1) # [Batch, Seq]
        if mask is not None:
            # Mask: 1=Valido, 0=Padding.
            # Mettiamo -inf dove c'è padding per annullare il softmax
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1) # [Batch, Dim]

class GatedMultimodalUnit(nn.Module):
    """Fusione intelligente che impara a pesare Testo vs Audio."""
    def __init__(self, dim):
        super().__init__()
        self.linear_text = nn.Linear(dim, dim)
        self.linear_audio = nn.Linear(dim, dim)
        self.sigmoid_gate = nn.Linear(dim * 2, 1)

    def forward(self, h_text, h_audio):
        h_t = torch.tanh(self.linear_text(h_text))
        h_a = torch.tanh(self.linear_audio(h_audio))
        concat = torch.cat([h_text, h_audio], dim=1)
        z = torch.sigmoid(self.sigmoid_gate(concat))
        return z * h_t + (1 - z) * h_a

def get_audio_dim(config):
    """Helper per capire la dimensione dell'audio in base al modello scelto."""
    name = config.model.audio.pretrained.lower()
    if "whisper-large" in name: return 1280
    if "xls-r" in name or "large" in name: return 1024
    return 768

# ==============================================================================
# 2. MODELLO SOLO TESTO (Text Baseline)
# ==============================================================================

class TextClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model.text.name)
        dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.model.text.dropout)
        
        # Pooling attenzionale (meglio del semplice CLS token)
        self.pooler = AttentivePooling(dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 2)
        )

    def forward(self, batch):
        # batch['input_ids']: [B, Seq]
        out = self.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        seq = out.last_hidden_state
        
        pooled = self.pooler(seq, mask=batch['attention_mask']) # [B, Dim]
        pooled = self.dropout(pooled)
        
        return self.classifier(pooled)

# ==============================================================================
# 3. MODELLO SOLO AUDIO (Audio Baseline)
# ==============================================================================

class AudioClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Qui assumiamo di ricevere già le feature estratte (Tensor [B, Seq, Dim])
        # dal dataloader (pre-computate), quindi non carichiamo Wav2Vec2 intero per risparmiare RAM.
        # Se volessi fare fine-tuning dell'audio, dovresti caricare Wav2Vec2Model qui.
        
        dim = get_audio_dim(config)
        self.dropout = nn.Dropout(config.model.audio.dropout)
        
        # Proiezione per stabilizzare
        self.proj = nn.Linear(dim, 256)
        
        self.pooler = AttentivePooling(256)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 2)
        )

    def forward(self, batch):
        x = batch['audio_features'].float() # [B, Seq, Dim]
        x = F.gelu(self.proj(x))            # [B, Seq, 256]
        
        # Audio non ha mask esplicita nel tuo dataset (è padded a 0), 
        # AttentivePooling gestisce bene se impara che 0 non è importante.
        pooled = self.pooler(x)             
        pooled = self.dropout(pooled)
        
        return self.classifier(pooled)

# ==============================================================================
# 4. MODELLO MULTIMODALE SOTA (Co-Attention + Gated Fusion)
# ==============================================================================

class SotaMultimodalClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Encoders
        self.text_encoder = AutoModel.from_pretrained(config.model.text.name)
        text_dim = self.text_encoder.config.hidden_size
        audio_dim = get_audio_dim(config)
        
        self.common_dim = 256
        dropout = config.model.text.dropout

        # Proiezioni
        self.text_proj = nn.Linear(text_dim, self.common_dim)
        self.audio_proj = nn.Linear(audio_dim, self.common_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Co-Attention
        self.cross_attn_T_A = nn.MultiheadAttention(self.common_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.cross_attn_A_T = nn.MultiheadAttention(self.common_dim, num_heads=4, batch_first=True, dropout=dropout)
        
        self.norm_t = nn.LayerNorm(self.common_dim)
        self.norm_a = nn.LayerNorm(self.common_dim)

        # Pooling
        self.text_pooler = AttentivePooling(self.common_dim)
        self.audio_pooler = AttentivePooling(self.common_dim)

        # Gated Fusion
        self.fusion_gate = GatedMultimodalUnit(self.common_dim)

        # Classificatore
        self.classifier = nn.Sequential(
            nn.Linear(self.common_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, batch):
        # 1. Feature Raw
        t_raw = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state
        a_raw = batch['audio_features'].float()
        
        # 2. Proiezione
        t_emb = self.dropout_layer(F.gelu(self.text_proj(t_raw)))
        a_emb = self.dropout_layer(F.gelu(self.audio_proj(a_raw)))

        # 3. Co-Attention
        t_fused, _ = self.cross_attn_T_A(query=t_emb, key=a_emb, value=a_emb)
        t_final = self.norm_t(t_emb + t_fused)
        
        key_padding_mask = (batch['attention_mask'] == 0)
        a_fused, _ = self.cross_attn_A_T(query=a_emb, key=t_emb, value=t_emb, key_padding_mask=key_padding_mask)
        a_final = self.norm_a(a_emb + a_fused)

        # 4. Pooling
        t_vec = self.text_pooler(t_final, mask=batch['attention_mask'])
        a_vec = self.audio_pooler(a_final)

        # 5. Fusion
        fused_vec = self.fusion_gate(t_vec, a_vec)

        # 6. Logits
        return self.classifier(fused_vec)

# ==============================================================================
# 5. BUILDER (Lo switch nel Config)
# ==============================================================================

def build_model(config):
    mode = config.modality.lower()
    
    if mode == "text":
        print("🏗️  Building TEXT-ONLY Model")
        return TextClassifier(config)
    
    elif mode == "audio":
        print("🏗️  Building AUDIO-ONLY Model")
        return AudioClassifier(config)
    
    elif "multimodal" in mode:
        print("🏗️  Building MULTIMODAL SOTA Model (Co-Attention)")
        return SotaMultimodalClassifier(config)
    
    else:
        raise ValueError(f"Modalità '{mode}' non riconosciuta in build_model")