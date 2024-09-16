import torch
import torch.nn as nn

from model_layers.Feed_forward import Feed_forward
from model_layers.LayerNorm import LayerNorm
from model_layers.MultiHeadAttention import MultiHeadAttention


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_norm1 = LayerNorm()
        self.attn = MultiHeadAttention()

        self.layer_norm2 = LayerNorm()
        self.feedForward = Feed_forward()

    def forward(self, resid_in):
        '''
        resid_in is read from residual stream with dim: (batch, seq_len, d_model)

        :param x:
        :return:
        '''

        norm_x = self.layer_norm1(resid_in)
        attn_x = self.attn(norm_x)

        resid_mid = resid_in + attn_x         # Writing to residual stream

        norm_x = self.layer_norm2(resid_mid)
        ff_x = self.feedForward(norm_x)

        resid_out = resid_mid + ff_x          # Writing to residual stream

        return resid_out
#
# blck = Block()
# x = torch.randn(200, 10, 768)
# x = blck(x)
# print(x.shape)