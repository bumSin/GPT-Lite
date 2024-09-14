import torch
import torch.nn as nn
from einops import einops

from config import ConfigManager

'''
Layer Normalization is a technique used in neural networks to stabilize and speed up training by normalizing the activations within a layer.

How It Works:

	1.	Mean and Variance: For each input feature (across the dimensions), you compute the mean and variance across all features for a given sample. This allows you to capture the distribution of values within a layer.
	•	Mean: The average of the activations across the features.
	•	Variance: The spread or deviation of the activations from the mean.
	2.	Normalization: You normalize each feature by subtracting the mean and dividing by the square root of the variance (with a small epsilon added to avoid division by zero). This makes the activations have zero mean and unit variance.
	3.	Affine Transformation: After normalization, a learned scaling factor (gamma) and bias (beta) are applied to allow the model to adjust the scale and shift of the normalized values. This makes layer normalization flexible and allows the network to recover any distribution of activations it needs.

Here, we normalise every token independently.
'''

class LayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = ConfigManager.get_config()

        self.w = nn.Parameter(torch.ones(self.cfg.d_model))    # gamma; scale
        self.b = nn.Parameter(torch.zeros(self.cfg.d_model))   # beta:  bias

    def forward(self, x):
        # expected dim of x: batch, seq_len, d_model

        token_mean = einops.reduce(x, "batch seq_len d_model -> batch seq_len 1", "mean")
        #          = torch.mean(x, dim = -1, keepdim=True)

        token_var = einops.reduce((x - token_mean).pow(2), "batch seq_len d_model -> batch seq_len 1", "mean")
        #         = torch.var(x, dim = -1, keepdim=True)

        # To normalise, we divide by std deviation, which is sqrt of var
        normalised_x = (x - token_mean) / torch.sqrt(token_var + self.cfg.eps)    # eps is a very small value added to prevent division by zero if by chance var is 0
        normalised_x = normalised_x * self.w + self.b

        return normalised_x



# ln = LayerNorm()
# x = torch.randn(512, 20, 768)
#
# x = ln(x)
# print(x.shape)