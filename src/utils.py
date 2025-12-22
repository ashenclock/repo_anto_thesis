import yaml
import torch
import numpy as np
import random
import os
import gc
class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def load_config(path):
    with open(path, 'r') as f:
        return Config(yaml.safe_load(f))

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
    """Libera la memoria della GPU e il garbage collector."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()