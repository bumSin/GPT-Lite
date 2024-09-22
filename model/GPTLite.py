import torch
import torch.nn as nn

from config import ConfigManager
from model.Block import Block
from model_layers.Embedding import Embedding
from model_layers.LayerNorm import LayerNorm
from model_layers.PositionalEmbedding import PositionalEmbedding
from model_layers.Unembed import Unembed


class GPTLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = ConfigManager.get_config()

        self.embed = Embedding()
        self.pos_embed = PositionalEmbedding()
        self.blocks = nn.ModuleList([Block() for _ in range(self.cfg.n_layers)])
        self.layerNorm_final = LayerNorm()
        self.unembed = Unembed()

    def forward(self, tokens):
        '''
        x is set of tokens of shape, (batch, seq_len)

        :param x:
        :return:
        '''

        embeddings = self.embed(tokens)
        pos_embeddings = self.pos_embed(tokens)

        residual = embeddings + pos_embeddings        # Initiating residual stream

        for block in self.blocks:
            residual = block(residual)

        normalised_residual = self.layerNorm_final(residual)
        logits = self.unembed(normalised_residual)      # Expected dim: batch, seq_len, d_vocab

        return logits


# transformer = GPTLite()
# x = torch.randint(0, 512, (4, 10))
#
# logits = transformer(x)
#
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# # Example usage:
# total_params = count_parameters(GPTLite)
# print(f"Total number of trainable parameters: {total_params}")
# Output = 163,087,441    # Real GPT-2 had 1.5 billion params