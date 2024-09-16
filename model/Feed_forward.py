import torch
import torch.nn as nn
from fancy_einsum import einsum

from config import ConfigManager


class Feed_forward(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = ConfigManager.get_config()

        self.mlp_expand = nn.Linear(self.cfg.d_model, self.cfg.d_mlp)     # Weights are initialized uniformly
        nn.init.normal_(self.mlp_expand.weight, std=self.cfg.init_range)  # Use torch.nn.init.normal_ to apply normal (Gaussian) initialization to the layer’s weights
        nn.init.normal_(self.mlp_expand.bias,  std=self.cfg.init_range)   # and optionally to the bias.

        self.mlp_compress = nn.Linear(self.cfg.d_mlp, self.cfg.d_model)
        nn.init.normal_(self.mlp_compress.weight, std=self.cfg.init_range)  # Use torch.nn.init.normal_ to apply normal (Gaussian) initialization to the layer’s weights
        nn.init.normal_(self.mlp_compress.bias, std=self.cfg.init_range)  # and optionally to the bias.

    def forward(self, x):
        '''
        Reads in from the residual stream, hence the shape of input is (batch, seq_len, d_model)
        :param x: (batch, seq_len, d_model)
        :return: (batch, seq_len, d_model)
        '''

        x = self.mlp_expand(x)
        x = nn.functional.gelu(x)                 # FYI, GPT-2 uses something called gelu_new, I will stick to our regular gelu here
        x = self.mlp_compress(x)

        return x

# mlp = Feed_forward()
# x = torch.randn(200, 10, 768)
# x = mlp(x)
# print(x.shape)