import torch.nn as nn
from model.LayerNorm import LayerNorm
from model.FeedForward import FeedForward
from model.MultiHeadAttention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attn_weights
