import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor, Wav2Vec2Model, WhisperModel
from speechbrain.pretrained import EncoderClassifier

# --- HELPER PER GESTIRE IL CONFIG ---
def get_num_classes(config):
    """Estrae il numero di classi dal config in modo sicuro."""
    if hasattr(config.labels.mapping, 'to_dict'):
        mapping = config.labels.mapping.to_dict()
    else:
        mapping = config.labels.mapping
    
    num = len(set(mapping.values()))
    # Se per qualche motivo (es. binary strict) c'è 1 sola label nel mapping, 
    # impostiamo comunque 2 output per la classificazione binaria
    if num < 2: 
        return 2
    return num

# --- 1. ENCODER AUDIO UNIVERSALE (SSL o ECAPA) ---
class AudioEncoderWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_name = config.model.audio.pretrained
        self.device = config.device
        self.is_ssl = False
        self.is_whisper = False
        self.output_dim = 0

        # CASO A: Modelli HuggingFace SSL (Wav2Vec2, HuBERT, Whisper)
        if "/" in self.model_name and "speechbrain" not in self.model_name:
            self.is_ssl = True
            print(f"🔊 Caricamento Audio SSL: {self.model_name}")
            
            if "whisper" in self.model_name.lower():
                self.model = WhisperModel.from_pretrained(self.model_name).encoder
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
                self.is_whisper = True
            else:
                # Wav2Vec2, XLS-R, etc.
                self.model = Wav2Vec2Model.from_pretrained(self.model_name)
                self.is_whisper = False
            
            self.output_dim = self.model.config.hidden_size
            
            # Freeze opzionale
            if not config.model.audio.trainable_encoder:
                for param in self.model.parameters(): param.requires_grad = False
                
        # CASO B: SpeechBrain (ECAPA-TDNN)
        else:
            print(f"🔊 Caricamento Audio SpeechBrain: {self.model_name}")
            self.model = EncoderClassifier.from_hparams(
                source=self.model_name,
                run_opts={"device": self.device}
            )
            self.output_dim = 192 # ECAPA standard
            if not config.model.audio.trainable_encoder:
                for param in self.model.parameters(): param.requires_grad = False

    def forward(self, waveforms):
        if self.is_ssl:
            if self.is_whisper:
                # Whisper richiede feature extraction Mel
                wavs_cpu = [w.cpu().numpy() for w in waveforms]
                inputs = self.feature_extractor(wavs_cpu, sampling_rate=16000, return_tensors="pt", padding=True)
                feats = inputs.input_features.to(self.device)
                out = self.model(feats).last_hidden_state
            else:
                # Wav2Vec2 accetta raw audio
                out = self.model(waveforms).last_hidden_state
            
            # Mean Pooling temporale [Batch, Time, Dim] -> [Batch, Dim]
            return torch.mean(out, dim=1)
        else:
            # ECAPA-TDNN
            # Gestione gradiente manuale per SpeechBrain
            is_trainable = any(p.requires_grad for p in self.model.parameters())
            with torch.set_grad_enabled(is_trainable):
                emb = self.model.encode_batch(waveforms)
            return emb.squeeze(1)

# --- 2. MODELLO MULTIMODALE COMPLETO ---
class MultimodalClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 1. Text Encoder (UmBERTo / BERT)
        print(f"📝 Caricamento Text Encoder: {config.model.text.name}")
        self.text_encoder = AutoModel.from_pretrained(config.model.text.name)
        text_dim = self.text_encoder.config.hidden_size
        
        # 2. Audio Encoder (Wrapper Intelligente)
        self.audio_encoder = AudioEncoderWrapper(config)
        audio_dim = self.audio_encoder.output_dim
        
        print(f"🔗 Fusion Dimensions: Text({text_dim}) + Audio({audio_dim})")
        
        # 3. Classificatore
        fusion_dim = text_dim + audio_dim
        
        # --- FIX: Uso la funzione helper ---
        num_classes = get_num_classes(config)

        self.dropout = nn.Dropout(config.model.text.dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, batch):
        # Text Forward
        t_out = self.text_encoder(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask']
        )
        # UmBERTo/RoBERTa non ha pooler_output pre-addestrato affidabile come BERT,
        # spesso si usa il token <s> (indice 0) o il mean pooling.
        # Qui usiamo il primo token (CLS/Start)
        text_emb = t_out.last_hidden_state[:, 0, :] 
        
        # Audio Forward
        audio_emb = self.audio_encoder(batch['waveform'])
        
        # Concatenazione
        combined = torch.cat((text_emb, audio_emb), dim=1)
        
        logits = self.classifier(self.dropout(combined))
        return logits

# --- CLASSI MODELLO SINGOLO (Per completezza) ---
class TextClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.model.text.name)
        self.dropout = nn.Dropout(config.model.text.dropout)
        
        # --- FIX: Uso la funzione helper ---
        num_classes = get_num_classes(config)
        
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, batch):
        out = self.bert(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            emb = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(emb))
class XPhoneBERTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_name = config.model.text.name
        print(f"📝 Caricamento XPhoneBERT: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.model.text.dropout)
        self.classifier_head = nn.Linear(
            self.encoder.config.hidden_size,
            get_num_classes(config)
        )

    def forward(self, batch):
        outputs = self.encoder(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask']
        )
        # Per modelli basati su RoBERTa (come XPhoneBERT), è meglio usare l'embedding del primo token
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier_head(self.dropout(pooled_output))
        return logits

# --- MODIFICA LA FUNZIONE build_model ---
def build_model(config):
    if config.modality == 'text':
        # Scelta intelligente: se il nome contiene "xphonebert", usa il modello giusto
        if "xphonebert" in config.model.text.name.lower():
            return XPhoneBERTClassifier(config)
        else:
            return TextClassifier(config)
            
    elif config.modality == 'audio': return AudioClassifier(config)
    elif config.modality == 'multimodal': return MultimodalClassifier(config)
    else: raise ValueError(f"Modality {config.modality} not supported")

class AudioClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AudioEncoderWrapper(config)
        
        # --- FIX: Uso la funzione helper ---
        num_classes = get_num_classes(config)
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.model.audio.dropout),
            nn.Linear(self.encoder.output_dim, num_classes)
        )
    def forward(self, batch):
        emb = self.encoder(batch['waveform'])
        return self.classifier(emb)

# Factory
def build_model(config):
    if config.modality == 'text': return TextClassifier(config)
    elif config.modality == 'audio': return AudioClassifier(config)
    elif config.modality == 'multimodal': return MultimodalClassifier(config)
    else: raise ValueError(f"Modality {config.modality} not supported")