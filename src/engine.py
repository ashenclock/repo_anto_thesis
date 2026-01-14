import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from pathlib import Path
from src.utils import clear_memory
from src.models import build_model

def calculate_metrics(y_true, y_pred):
    """Calcola metriche cliniche dettagliate."""
    # Assicuriamoci che siano numpy array interi
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calcolo Specificity e Sensitivity (Binario: 0=CTR, 1=AD)
    # Forziamo labels=[0, 1] per gestire casi in cui un batch ha una sola classe
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity
    }

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        # Sposta su GPU
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch.pop('labels')
        
        # Forward
        outputs = model(batch)
        loss = loss_fn(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Clip gradienti
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device, loss_fn):
    model.eval()
    all_preds, all_labels, all_probs, all_ids = [], [], [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            ids = batch.pop('id')
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop('labels')
            
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(ids)
            
    if len(all_labels) == 0:
        return 0, {}, {}

    # Calcola metriche complete
    metrics = calculate_metrics(all_labels, all_preds)
    
    details = {
        "ids": all_ids,
        "probs": all_probs,
        "labels": all_labels,
        "preds": all_preds
    }
    
    return total_loss / len(loader), metrics, details

class Trainer:
    def __init__(self, config, train_loader, val_loader, fold):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold = fold
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default metrica target: f1
        self.target_metric = config.training.get("eval_metric", "f1")

    def train(self):
        clear_memory()
        model = build_model(self.config).to(self.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=float(self.config.training.learning_rate), 
            weight_decay=self.config.training.weight_decay
        )
        
        total_steps = len(self.train_loader) * self.config.training.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * self.config.training.warmup_ratio), 
            num_training_steps=total_steps
        )
        
        # Pesi classi se necessario (opzionale)
        loss_fn = nn.CrossEntropyLoss()
        
        best_score = -1
        best_metrics = None
        best_details = None
        patience = 0
        
        print(f"--- Fold {self.fold} | Target: {self.target_metric.upper()} | LR: {self.config.training.learning_rate} ---")

        for epoch in range(self.config.training.epochs):
            t_loss = train_epoch(model, self.train_loader, optimizer, scheduler, loss_fn, self.device)
            v_loss, metrics, details = evaluate(model, self.val_loader, self.device, loss_fn)
            
            current_score = metrics.get(self.target_metric, metrics['f1'])
            
            print(f"Ep {epoch+1:02d} | Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | {self.target_metric.upper()}: {current_score:.4f} | Acc: {metrics['accuracy']:.4f}")
            
            if current_score > best_score:
                best_score = current_score
                best_metrics = metrics
                best_details = details
                torch.save(model.state_dict(), self.output_dir / "best_model.pt")
                patience = 0
                print(f"  🌟 New Best {self.target_metric.upper()}: {best_score:.4f}")
            else:
                patience += 1
            
            if patience >= self.config.training.early_stopping_patience:
                print("  🛑 Early Stopping.")
                break
                
        return best_metrics, best_details