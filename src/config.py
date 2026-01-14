import yaml
from pathlib import Path

class Config:
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    # --- FIX: Aggiunto metodo get per compatibilità ---
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        return str(self.__dict__)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def load_config(config_path: str | Path) -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config non trovato: {path.resolve()}")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config(data)