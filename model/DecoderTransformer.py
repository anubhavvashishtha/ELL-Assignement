import torch , math
import torch.nn as nn
import torch
from model.PositionalEncoding import PositionalEncoding
from model.TransformerBlock import TransformerBlock
from model.LayerNorm import LayerNorm

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads,
                 d_ff, max_seq_len, dropout=0.1, pretrained_embeddings=None):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Store the original embedding dimension (300 for FastText)
        self.embedding_dim = pretrained_embeddings.shape[1] if pretrained_embeddings is not None else d_model

        # Create embedding layer with original FastText dimension
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # Add projection layer to convert from FastText dim to d_model
            self.embedding_proj = nn.Linear(self.embedding_dim, d_model)
        else:
            self.embedding_proj = nn.Identity()  # No projection needed if no pretrained embeddings

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, x, return_attention=False):
        batch_size, seq_len = x.shape
        mask = self.create_causal_mask(seq_len, x.device)

        # Get embeddings in original dimension (300)
        x = self.embedding(x) * math.sqrt(self.embedding_dim)

        # Project to d_model (which is divisible by num_heads)
        x = self.embedding_proj(x)

        x = self.pos_encoding(x)
        x = self.dropout(x)
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            if return_attention:
                attention_weights.append(attn_weights)
        x = self.norm(x)
        logits = self.output_projection(x)
        if return_attention:
            return logits, attention_weights
        return logits

