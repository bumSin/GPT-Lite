from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 768
    eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    n_heads: int = 12
    d_head: int = 64
    d_mlp: int = 3072

class ConfigManager:
    _instance = None

    @classmethod
    def get_config(cls):
        if cls._instance is None:
            cls._instance = Config()  # Instantiates only once
        return cls._instance


'''
Usage:
class SomeModel:
    def __init__(self):
        self.cfg = ConfigManager.get_config()
        print(self.cfg.d_model)  # Use the config values

model = SomeModel()
'''