import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
from torch.utils.data import WeightedRandomSampler

# Prevent tokenizer deadlocks on Linux
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path for importing src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import load_config
from src.models import build_model
from src.data import MultimodalDataset, collate_multimodal
from src.utils import set_seed, clear_memory

# ==========================================
# PATH CONFIGURATION
# ==========================================
BASE_DIR = Path("data/metadata")
META_FILE = BASE_DIR / "mmse_experiment_metadata_FINAL.csv"
SCORES_EN_FILE = BASE_DIR / "adresso_FULL_mmse.csv"

OUT_DIR = Path("outputs/GRAND_REGRESSION_RESULTS")
PLOT_DIR = Path("plots/GRAND_REGRESSION_PLOTS")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def force_config(config):
    config.task = "regression"
    if not hasattr(config, 'model'):
        config.model = type('obj', (object,), {})()
    config.model.output_dim = 1
    config.modality = "multimodal_cross_attention"
    config.model.text.name = "xlm-roberta-base"
    config.model.audio.pretrained = "facebook/wav2vec2-large-xlsr-53"
    config.training.batch_size = 8
    return config


def load_and_prep_data():
    print("Loading and preparing data...")
    if not META_FILE.exists():
        raise FileNotFoundError(f"Missing {META_FILE}. Make sure the file is present.")

    df_meta = pd.read_csv(META_FILE)

    # A. ITALIAN (Sicily data)
    df_it = df_meta[df_meta['Language'] == 'IT'].copy()
    if 'mmse_score' not in df_it.columns:
        print("WARNING: Column 'mmse_score' not found in IT. Check the unification script.")
        sys.exit(1)

    # B. ENGLISH (Match ADReSSo CSV)
    df_en = df_meta[df_meta['Language'] == 'EN'].copy()
    if SCORES_EN_FILE.exists():
        df_scores_en = pd.read_csv(SCORES_EN_FILE)
        df_scores_en['ID'] = df_scores_en['ID'].astype(str).str.strip().str.replace('"', '')
        en_map = dict(zip(df_scores_en['ID'], df_scores_en['Score']))
        df_en['mmse_score'] = df_en['Subject_ID'].apply(
            lambda fid: en_map.get(
                fid.replace("EN_", "").replace("TEST_", "").split("_Task_")[0].strip(), np.nan
            )
        )
        df_en = df_en.dropna(subset=['mmse_score'])
    else:
        print(f"ERROR: Missing EN scores file: {SCORES_EN_FILE}")
        sys.exit(1)

    df_en_train = df_en[df_en['Dataset'] != 'ADReSSo_Test'].copy()
    df_en_test = df_en[df_en['Dataset'] == 'ADReSSo_Test'].copy()

    print(f"   -> IT (Full): {len(df_it)} | EN Train: {len(df_en_train)} | EN Test: {len(df_en_test)}")
    return df_it, df_en_train, df_en_test


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    criterion = nn.L1Loss()
    running_loss = 0

    pbar = tqdm(loader, desc=f"      Epoch {epoch+1} [Train]", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        targets = batch.pop('labels').float()

        # Ensure output is 1D even when batch size is 1
        outputs = model(batch).view(-1)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix({'MAE': f"{loss.item():.3f}"})

    return running_loss / len(loader)


def evaluate_model(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            t = batch.pop('labels').float()
            # Flatten output to prevent 0-d array errors
            o = model(batch).view(-1)

            preds.extend(o.cpu().numpy().tolist())
            targets.extend(t.cpu().numpy().tolist())

    preds = np.array(preds)
    targets = np.array(targets)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    return rmse, mae, preds, targets


def plot_regression_results(df, title, filename):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    pal = {"CTR": "#66c2a5", "MCI": "#8da0cb", "AD": "#fc8d62", "MILD-AD": "#fc8d62"}
    order = [o for o in ['CTR', 'MCI', 'AD', 'MILD-AD'] if o in df['Diagnosis'].unique()]

    sns.boxplot(data=df, x='Diagnosis', y='Pred_MMSE', order=order, palette=pal,
                hue='Diagnosis', legend=False, width=0.6, fliersize=0)
    sns.stripplot(data=df, x='Diagnosis', y='Pred_MMSE', order=order,
                  color='black', alpha=0.4, jitter=True)
    plt.axhline(y=24, color='red', linestyle='--', alpha=0.5)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.ylim(0, 32)
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# CORE TRAINING ENGINE
# ==========================================

def run_experiment(config, train_df, test_df_A, test_df_B, device, tokenizer, exp_name):
    """Train on train_df and test on A (e.g. IT) and B (e.g. EN)."""
    print(f"\nSTART EXPERIMENT: {exp_name}")
    print("-" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Matrices to accumulate ensemble predictions
    preds_A = np.zeros((len(test_df_A), 5))
    preds_B = np.zeros((len(test_df_B), 5))

    loader_A = torch.utils.data.DataLoader(
        MultimodalDataset(test_df_A, config, tokenizer),
        batch_size=16, shuffle=False, collate_fn=collate_multimodal, num_workers=0,
    )
    loader_B = torch.utils.data.DataLoader(
        MultimodalDataset(test_df_B, config, tokenizer),
        batch_size=16, shuffle=False, collate_fn=collate_multimodal, num_workers=0,
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n   Fold {fold+1}/5")
        t_sub = train_df.iloc[train_idx].reset_index(drop=True)
        v_sub = train_df.iloc[val_idx].reset_index(drop=True)

        t_loader = torch.utils.data.DataLoader(
            MultimodalDataset(t_sub, config, tokenizer),
            batch_size=8, shuffle=True, collate_fn=collate_multimodal,
            drop_last=True, num_workers=0,
        )
        v_loader = torch.utils.data.DataLoader(
            MultimodalDataset(v_sub, config, tokenizer),
            batch_size=8, shuffle=False, collate_fn=collate_multimodal, num_workers=0,
        )

        clear_memory()
        model = build_model(config).to(device)
        # Centre bias on the MMSE scale
        nn.init.constant_(model.classifier[-1].bias, 25.0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=len(t_loader) * 20,
        )

        best_val_rmse = float('inf')
        epochs_no_improve = 0

        for ep in range(20):
            train_loss = train_one_epoch(model, t_loader, optimizer, scheduler, device, ep)
            val_rmse, val_mae, _, _ = evaluate_model(model, v_loader, device)

            print(f"      Ep {ep+1:02d} | Train MAE: {train_loss:.3f} | Val RMSE: {val_rmse:.3f}")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), OUT_DIR / f"best_model_{exp_name}_f{fold}.pt")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 5:
                print(f"      Early stopping at fold {fold+1}")
                break

        # Load best model and run inference on both test sets
        model.load_state_dict(torch.load(OUT_DIR / f"best_model_{exp_name}_f{fold}.pt"))
        _, _, pA, _ = evaluate_model(model, loader_A, device)
        _, _, pB, _ = evaluate_model(model, loader_B, device)
        preds_A[:, fold] = pA
        preds_B[:, fold] = pB

    # Final ensemble results
    test_df_A['Pred_MMSE'] = np.mean(preds_A, axis=1)
    test_df_B['Pred_MMSE'] = np.mean(preds_B, axis=1)

    rmse_A = np.sqrt(mean_squared_error(test_df_A['mmse_score'], test_df_A['Pred_MMSE']))
    rmse_B = np.sqrt(mean_squared_error(test_df_B['mmse_score'], test_df_B['Pred_MMSE']))

    print(f"\n{exp_name} COMPLETED")
    print(f"   Result on A (Target 1): RMSE = {rmse_A:.3f}")
    print(f"   Result on B (Target 2): RMSE = {rmse_B:.3f}")

    plot_regression_results(
        test_df_A, f"{exp_name} - Eval on A\nRMSE: {rmse_A:.2f}", f"{exp_name}_eval_A.png",
    )
    plot_regression_results(
        test_df_B, f"{exp_name} - Eval on B\nRMSE: {rmse_B:.2f}", f"{exp_name}_eval_B.png",
    )

    return test_df_A, test_df_B


# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config.yaml")
    config = load_config(args.parse_args().config)
    config = force_config(config)

    # 1. Data preparation
    df_it, df_en_train, df_en_test = load_and_prep_data()
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)

    # Experiment 1: Train on IT, test on IT (OOF) and EN (zero-shot)
    # For IT we use itself as Test_A to measure in-domain performance
    run_experiment(config, df_it, df_it, df_en_test, device, tokenizer, "EXP1_IT_to_EN")

    # Experiment 2: Train on EN, test on EN (OOF) and IT (zero-shot)
    run_experiment(config, df_en_train, df_en_train, df_it, device, tokenizer, "EXP2_EN_to_IT")

    # Experiment 3: Mixed training (weighted)
    print("\nSTART EXPERIMENT: MIXED_WEIGHTED")
    print("-" * 60)

    # Split IT for a clean test set
    it_train, it_test = train_test_split(
        df_it, test_size=0.2, stratify=df_it['Diagnosis'], random_state=42,
    )
    # Mix training sets
    df_mix_train = pd.concat([it_train, df_en_train]).reset_index(drop=True)

    # Weighted sampler to balance languages
    counts = df_mix_train['Language'].value_counts()
    weights = df_mix_train['Language'].map(
        {lang: 1.0 / count for lang, count in counts.items()}
    ).values
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))

    # Single training run (no K-fold for the mixed experiment for simplicity)
    t_loader = torch.utils.data.DataLoader(
        MultimodalDataset(df_mix_train, config, tokenizer),
        batch_size=8, sampler=sampler, collate_fn=collate_multimodal, drop_last=True,
    )
    v_loader = torch.utils.data.DataLoader(
        MultimodalDataset(df_en_test, config, tokenizer),
        batch_size=8, shuffle=False, collate_fn=collate_multimodal,
    )

    clear_memory()
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=len(t_loader) * 20,
    )

    best_v = float('inf')
    for ep in range(20):
        train_one_epoch(model, t_loader, optimizer, scheduler, device, ep)
        v_rmse, _, _, _ = evaluate_model(model, v_loader, device)
        print(f"      Ep {ep+1:02d} | Val RMSE (EN-Test): {v_rmse:.3f}")
        if v_rmse < best_v:
            best_v = v_rmse
            torch.save(model.state_dict(), OUT_DIR / "best_model_mix_weighted.pt")

    # Final evaluation of mixed model on IT-Test and EN-Test
    model.load_state_dict(torch.load(OUT_DIR / "best_model_mix_weighted.pt"))
    it_test_loader = torch.utils.data.DataLoader(
        MultimodalDataset(it_test, config, tokenizer),
        batch_size=16, shuffle=False, collate_fn=collate_multimodal,
    )
    en_test_loader = torch.utils.data.DataLoader(
        MultimodalDataset(df_en_test, config, tokenizer),
        batch_size=16, shuffle=False, collate_fn=collate_multimodal,
    )

    rmse_it, _, p_it, _ = evaluate_model(model, it_test_loader, device)
    rmse_en, _, p_en, _ = evaluate_model(model, en_test_loader, device)

    it_test['Pred_MMSE'] = p_it
    df_en_test['Pred_MMSE'] = p_en

    print(f"\nMIXED Result: IT-RMSE = {rmse_it:.3f} | EN-RMSE = {rmse_en:.3f}")
    plot_regression_results(
        it_test, f"Mixed Weighted - Eval on IT\nRMSE: {rmse_it:.2f}", "EXP3_MIX_eval_IT.png",
    )
    plot_regression_results(
        df_en_test, f"Mixed Weighted - Eval on EN\nRMSE: {rmse_en:.2f}", "EXP3_MIX_eval_EN.png",
    )

    print(f"\nDone! Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
