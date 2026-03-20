import torch
import numpy as np
import random
import gc

# Backward compatibility re-exports
from src.config import Config, load_config  # noqa: F401


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_memory():
    """Free GPU memory and run garbage collector."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
