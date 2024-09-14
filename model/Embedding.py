import torch
import torch.nn as nn

from config import ConfigManager


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.cfg = ConfigManager.get_config()
        self.token_to_embed_map = nn.Parameter(torch.empty(self.cfg.d_vocab, self.cfg.d_model))    # the lookup table which has to be learned, think of it as stack of embeddings of all token
        nn.init.normal_(self.token_to_embed_map, std=self.cfg.init_range)   # To learn more about this, refer to my blog, link in README.md file

    def forward(self, tokens):
        '''
        Just to remind the reader, tokens are integers --> indexes of words in vocab, google "byte pair encoding" to know more

        :param tokens: (batch, seq_len)  Input is batch of sentences (as tokens ofcourse).
        :return: (batch, seq_len, d_model) Embeddings of all the tokens in the batch of sentences
        '''

        return self.token_to_embed_map[tokens, :]      # advanced indexing, it just works :|


# em = Embedding()
# x = torch.randint(100, (512, 20))
#
# embeddings = em(x)
# print(embeddings.shape)
