import torch
import torch.nn as nn

from config import ConfigManager


class Unembed(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = ConfigManager.get_config()

        self.unembed = nn.Linear(self.cfg.d_model, self.cfg.d_vocab)
        nn.init.normal_(self.unembed.weight, std=self.cfg.init_range)  # Use torch.nn.init.normal_ to apply normal (Gaussian) initialization to the layer’s weights
        nn.init.normal_(self.unembed.bias, std=self.cfg.init_range)    # and optionally to the bias.

    def forward(self, x):
        logits = self.unembed(x)
        return logits


# ue = Unembed()
# x = torch.randn(200, 10, 768)
# x = ue(x)
# print(x.shape)