import math

import torch
import torch.nn as nn
from fancy_einsum import einsum

from config import ConfigManager

'''
1. Calculate Attention Pattern
   * Linearly transform x to get Q and K vectors
       - Remember every token has it's own set of Q, K and V vectors
   * Split them across token dimension for each head
       - If there are 24 words, each head will all 24 words but partially, each head will get d_model/num_heads part of the token
       - use einsum to do the above two steps in one line
   * Within each head, multiply all Q vectors with all K vectors to get the atention score
   * Apply causal mask
   * Softmax row-wise to convert scores into probabilities --> These probabilities represents which tokens attend to which other tokens and how much

2. Information flow among tokens within each head
   * Linealy transform x to get V vector
   * User attention pattern as weights and take the weighted average of V for every token to get z
   * Add outputs of all heads [Attention is additive here]
       - NOTE: This is a little different from actual implementation of GPT-2. 
       - In GPT-2’s architecture, the outputs of the attention heads are concatenated and then linearly projected, which is different from adding the outputs.
       - But apparently it's mathematically equivalent, so let it be.
   * Linearly transform to match the dimensions of residual stream

'''


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = ConfigManager.get_config()

        self.W_Q = nn.Parameter(torch.empty((self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)))
        self.b_Q = nn.Parameter(torch.zeros((self.cfg.n_heads, self.cfg.d_head)))

        self.W_K = nn.Parameter(torch.empty((self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)))
        self.b_K = nn.Parameter(torch.zeros((self.cfg.n_heads, self.cfg.d_head)))

        self.W_V = nn.Parameter(torch.empty((self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)))
        self.b_V = nn.Parameter(torch.zeros((self.cfg.n_heads, self.cfg.d_head)))

        self.W_O = nn.Parameter(torch.empty((self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)))
        self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model))

        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32))

    def forward(self, x):
        '''


        :param x: (batch seq_len d_model)
        :return:
        '''

        Q = einsum('batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head', x, self.W_Q) + self.b_Q
        K = einsum('batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head', x, self.W_K) + self.b_K

        attn_scores = einsum('batch q_len n_heads d_head, batch k_len n_heads d_head -> batch n_heads q_len k_len', Q, K)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        
        masked_attn_scores = self.apply_mask(attn_scores)
        attn_pattern = masked_attn_scores.softmax(dim=-1)

        V = einsum('batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head', x, self.W_V) + self.b_V
        Z = einsum('batch n_heads seq_len seq_len, batch seq_len n_heads d_head -> batch seq_len n_heads d_head', attn_pattern, V)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", Z, self.W_O) + self.b_O
        return attn_out

    def apply_mask(self, attn_scores):
        # dim of attn_scores: batch n_heads seq_len seq_len
        mask = torch.triu(torch.ones(attn_scores.size(-1), attn_scores.size(-1)), diagonal = 1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


# attn = MultiHeadAttention()
# x = torch.randn(100, 10, 768)
# attn_out = attn(x)
# print(attn_out.shape)