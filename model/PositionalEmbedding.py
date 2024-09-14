import einops
import torch
import torch.nn as nn

from config import ConfigManager

class PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.cfg = ConfigManager.get_config()
        self.positional_embedddings_map = nn.Parameter(torch.empty(self.cfg.n_ctx, self.cfg.d_model))
        nn.init.normal_(self.positional_embedddings_map, std=self.cfg.init_range)  # To learn more about this, refer to my blog, link in README.md file


    def forward(self, x):
        '''
        Returns learned pos embeddings for tokens. positional embeddings are function of only position of token in sequence
        :param x: (batch, seq_len)     # max allowed value of seq_len is n_ctx here
        :return: (batch, seq_len, d_model) Positional Embeddings for all the tokens
        '''

        batch_size = x.size(0)
        seq_len = x.size(1)

        # Model learns pos embedding from pos 0 till pos: 1024 or n_ctx. Since seq_len might be less than 1024, we slice out pos embeddings for positions 0 to seq_len
        pos_embeddings = self.positional_embedddings_map[:seq_len, :]   # pos_embeddings dim: seq_len, d_model

        # Adding batch dim by Repeating across batch
        pos_embeddings = einops.repeat(pos_embeddings, 'seq_len d_model -> batch seq_len d_model', batch = batch_size)  # batch seq_len d_model

        return pos_embeddings     # pos_embeddings dim: batch seq_len d_model


x = torch.randn(32, 20)
pe = PositionalEmbedding()

em = pe(x)
print(em.shape)   # expected shape: batch, seq_len, d_model
