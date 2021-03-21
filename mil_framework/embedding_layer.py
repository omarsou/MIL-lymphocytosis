import torch.nn as nn
import torch

# EMBEDDING BLOCK


class MaskedNorm(nn.Module):
    def __init__(self, num_features, mask_on):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features=num_features)
        self.num_features = num_features
        self.mask_on = mask_on

    def forward(self, x, mask=None, mask_on=True):
        if self.training and self.mask_on:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])


class SpecialEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        input_size: int,
        num_heads: int,
        mask_on: bool,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.ln = nn.Linear(self.input_size, self.embedding_dim)
        self.norm = MaskedNorm(num_features=embedding_dim, mask_on=mask_on)
        self.activation = nn.ReLU()

    # pylint: disable=arguments-differ
    def forward(self, x, mask=None):
        x = self.ln(x)
        x = self.norm(x, mask)
        x = self.activation(x)
        return x