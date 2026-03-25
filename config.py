import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    d_model: int
    vocab_size: int

@dataclass
class InferenceConfig:
    batch_size: int
    max_seq_len: int
    use_triton: bool
    use_kv_cache: bool

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self._raw = yaml.safe_load(f)
            
        self.model = ModelConfig(**self._raw.get("model", {}))
        self.inference = InferenceConfig(**self._raw.get("inference", {}))
        
    @classmethod
    def load(cls, config_path: str = "config.yaml") -> 'Config':
        return cls(config_path)

# Global configuration instance
system_config = Config.load()
