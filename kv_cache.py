import torch
import torch.nn as nn
import numpy as np
import math
import time


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        if past_kv is not None:
            past_keys, past_values = past_kv
            K = torch.cat([past_keys, K], dim=2)
            V = torch.cat([past_values, V], dim=2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)
        
        present_kv = (K, V) if use_cache else None
        
        return output, attn_weights, present_kv


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        attn_output, attn_weights, present_kv = self.attention(x, mask, past_kv, use_cache)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights, present_kv


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads,
                 d_ff, max_seq_len, dropout=0.1, pretrained_embeddings=None):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding_dim = pretrained_embeddings.shape[1] if pretrained_embeddings is not None else d_model

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding_proj = nn.Linear(self.embedding_dim, d_model)
        else:
            self.embedding_proj = nn.Identity()

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

    def forward(self, x, return_attention=False, past_kvs=None, use_cache=False):
        batch_size, seq_len = x.shape
        mask = self.create_causal_mask(seq_len, x.device)

        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.embedding_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        attention_weights = []
        present_kvs = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, attn_weights, present_kv = layer(x, mask, past_kv=past_kv, use_cache=use_cache)
            
            if return_attention:
                attention_weights.append(attn_weights)
            if use_cache:
                present_kvs.append(present_kv)
        
        x = self.norm(x)
        logits = self.output_projection(x)
        
        if return_attention and use_cache:
            return logits, attention_weights, present_kvs
        elif return_attention:
            return logits, attention_weights
        elif use_cache:
            return logits, present_kvs
        return logits


# ============================================================================
# GENERATION WITH KV CACHE
# ============================================================================

def generate_with_kv_cache(model, prompt, vocab, max_length=50, device='cuda', use_cache=True):
    """Generate text with optional KV caching for faster autoregressive generation"""
    model.eval()
    start_time = time.time()

    tokens = [vocab.word2idx[vocab.SOS_TOKEN]] + vocab.encode(prompt)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    # Initialize cache
    cache = None
    generated_tokens = 0

    with torch.no_grad():
        for step in range(max_length):
            if tokens_tensor.size(1) >= model.max_seq_len:
                break

            # For cached generation, only pass the last token after first step
            if use_cache and step > 0 and cache is not None:
                input_tokens = tokens_tensor[:, -1:]
            else:
                input_tokens = tokens_tensor

            # Forward pass with cache
            if use_cache:
                outputs = model(input_tokens, past_kvs=cache, use_cache=True)
                if isinstance(outputs, tuple):
                    logits, cache = outputs
                else:
                    logits = outputs
            else:
                logits = model(input_tokens, use_cache=False)

            # Sample next token
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)

            if next_token.item() == vocab.word2idx[vocab.EOS_TOKEN]:
                break

            tokens_tensor = torch.cat([tokens_tensor, next_token], dim=1)
            generated_tokens += 1

    end_time = time.time()
    generation_time = end_time - start_time

    return {
        'text': vocab.decode(tokens_tensor.squeeze(0).tolist()),
        'time': generation_time,
        'tokens_per_second': generated_tokens / generation_time if generation_time > 0 else 0,
        'num_tokens': generated_tokens
    }


def compare_generation_speed(model, prompt, vocab, max_length=50, device='cuda'):
    """Compare generation speed with and without KV caching"""
    
    print("\n" + "="*70)
    print("KV CACHE GENERATION COMPARISON")
    print("="*70)
    
    # Without cache
    print("\n🐢 Without KV Cache:")
    result_no_cache = generate_with_kv_cache(model, prompt, vocab, max_length, device, use_cache=False)
    print(f"Text: {result_no_cache['text']}")
    print(f"Time: {result_no_cache['time']:.4f}s")
    print(f"Tokens/sec: {result_no_cache['tokens_per_second']:.2f}")
    print(f"Total tokens: {result_no_cache['num_tokens']}")
    
    # With cache
    print("\n🚀 With KV Cache:")
    result_cache = generate_with_kv_cache(model, prompt, vocab, max_length, device, use_cache=True)
    print(f"Text: {result_cache['text']}")
    print(f"Time: {result_cache['time']:.4f}s")
    print(f"Tokens/sec: {result_cache['tokens_per_second']:.2f}")
    print(f"Total tokens: {result_cache['num_tokens']}")
    
    # Speedup
    if result_no_cache['time'] > 0:
        speedup = result_no_cache['time'] / result_cache['time']
        print(f"\n⚡ Speedup: {speedup:.2f}x faster with KV cache!")
    
    print("="*70 + "\n")
    
    return result_no_cache, result_cache