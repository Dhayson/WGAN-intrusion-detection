import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, heads, batch_first=True, dropout=dropout)

    def forward(self, x):
        # No MultiheadAttention, Q, K e V são a própria entrada no Self-Attention
        attn_output, _ = self.mha(x, x, x, need_weights=False)
        return attn_output

class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32) \
            .reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, x):
        # print("POSITIONAL ENCODING")
        # print(" ", x.shape, self.P.shape)
        x = x + self.P[:, :x.shape[1], :].to(x.device)
        # print(" ", x.shape)
        return self.dropout(x)