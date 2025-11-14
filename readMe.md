### 🚀 ELL Assignment – Decoder-Only Transformer Project
This project implements a Decoder-Only Transformer Language Model inspired by the TinyStories paper. The entire model, including the tokenizer, training pipeline, and inference engine, is built from scratch using PyTorch.

It includes:

- Custom tokenizer and vocabulary
- Full decoder-only transformer architecture
- Training pipeline with checkpoints
- Support for gradient accumulation
- Inference engine with KV-cache optimization and beam search

Notebook-based workflows for training and generation

📂 Repository Structure
bash
```
ELL Assignment/
│
├── Dataset/
│   ├── TinyStories.py
│   ├── Vocabulary.py
│   └── load_fasttext_model.py
│
├── model/
│   ├── DecoderTransformer.py
│   ├── MultiHeadAttention.py
│   ├── FeedForward.py
│   ├── LayerNorm.py
│   └── PositionalEncoding.py
│
├── inference/
│   └── inference.py
│
├── checkpoints/
├── results/
│
├── train.ipynb
├── training_checkpoint.ipynb
├── training_accumulation.ipynb
├── chat_with_me.ipynb
└── kv_cache.ipynb
```
## Key Components

### Vocab (`checkpoints/baseline/vocab.JSON`)

### Model Architecture (`model/`)
- **DecoderTransformer**: Complete transformer decoder implementation
- **MultiHeadAttention**: Multi-head self-attention with causal masking
- **TransformerBlock**: Single transformer layer with attention and FFN
- **PositionalEncoding**: Sinusoidal positional embeddings
- **FeedForward**: Position-wise feedforward network
- **LayerNorm**: Layer normalization implementation

### Dataset & Vocabulary (`Dataset/`)
- **TinyStories.py**: Handles the TinyStories dataset loading and preprocessing
- **Vocabulary.py**: Tokenization, encoding/decoding, and vocabulary management
- **load_fasttext_model.py**: Utility for loading pre-trained FastText embeddings

### Training Notebooks
- **train.jpynb**: Main training script
- **training_accumulation.jpynb**: Gradient accumulation experiments (steps 1, 2, 4, 8)
- **training_checkpoint.jpynb**: Gradient checkpointing for memory optimization
- **kv_cache.jpynb**: Key-Value cache implementation analysis
- **chat_with_me.jpynb**: Interactive chat interface with trained model

### Inference & Evaluation (`inference/`)
- **inference.py**: Contains:
  - `evaluate_model()`: Comprehensive evaluation with perplexity and BLEU scores
  - `generate()`: Basic autoregressive generation
  - `kv_generate()`: Generation with optional KV caching
  - `interactive_generation()`: Live text generation interface
- **beam_search.py**: Beam search implementation for improved generation 
  - **`beam_generate()`**: Implements beam search decoding with configurable beam width
  - **`evaluate_decoding_strategies()`**: Compares greedy vs beam search (width 5, 10)
  - **`get_validation_prompts_and_targets()`**: Extracts prompts (first 5 words) and targets from validation data
  - **`save_beam_results()`**: Generates professional comparison plots

## Key Features

### 🚀 Performance Optimizations
- **Gradient Accumulation**: Support for steps 1, 2, 4, 8 to simulate larger batch sizes
- **Gradient Checkpointing**: Memory-efficient training via activation recomputation
- **KV Caching**: Faster inference by caching key-value pairs in attention

### 📊 Evaluation Metrics
- **Perplexity**: Measure of model confidence and language modeling quality
- **BLEU Score**: Text generation quality assessment
- **Attention Visualization**: Heatmaps for model interpretability
- **Performance Benchmarks**: Tokens/second, memory usage, timing comparisons

## Before running the notebook please add these two .pt in root directory 
https://drive.google.com/file/d/1xItvsyY7Xgr_kXHGPdsHqdIwQwIhXYSK/view?usp=sharing

https://drive.google.com/file/d/1wC8JE1vXnjXyjlbeVlaz512ZjNlvkMPM/view?usp=sharing

### 🛠️ Install all required libraries:

```
pip install torch
pip install numpy
pip install matplotlib
pip install seaborn
pip install datasets
pip install gensim
pip install evaluate
```

### 📊 Evaluation (perplexity, accuracy, etc.) is handled using the evaluate library inside the training and inference notebooks.