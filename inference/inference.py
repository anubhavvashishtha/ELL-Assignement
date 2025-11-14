import torch
import numpy as np
from tqdm import tqdm
import evaluate

def evaluate_model(model, val_dataset, vocab, num_samples=50, prompt_length=5, 
                   max_generation_length=50, device='cuda'):
    """
    Evaluate model on validation samples using perplexity and BLEU score.
    
    Args:
        model: Trained DecoderTransformer model
        val_dataset: Validation TinyStoriesDataset
        vocab: Vocabulary object
        num_samples: Number of samples to evaluate (default: 50)
        prompt_length: Number of tokens to use as prompt (default: 5)
        max_generation_length: Maximum tokens to generate (default: 50)
        device: Device to run evaluation on
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    bleu_metric = evaluate.load("bleu")
    
    # Storage for results
    all_perplexities = []
    all_references = []
    all_predictions = []
    
    # Sample random indices from validation set
    sample_indices = np.random.choice(len(val_dataset), 
                                     size=min(num_samples, len(val_dataset)), 
                                     replace=False)
    
    print(f"Evaluating on {len(sample_indices)} samples...")
    
    with torch.no_grad():
        for idx in tqdm(sample_indices):
            # Get full sequence
            full_sequence = val_dataset[idx]
            
            # Extract prompt (first prompt_length tokens) and reference (rest)
            prompt_tokens = full_sequence[:prompt_length].tolist()
            reference_tokens = full_sequence[prompt_length:].tolist()
            
            # Remove padding and special tokens from reference
            reference_tokens = [t for t in reference_tokens 
                              if t not in [vocab.word2idx[vocab.PAD_TOKEN],
                                          vocab.word2idx[vocab.SOS_TOKEN],
                                          vocab.word2idx[vocab.EOS_TOKEN]]]
            
            if len(reference_tokens) == 0:
                continue
            
            # Decode prompt for generation
            prompt_text = vocab.decode(prompt_tokens)
            
            # Generate continuation using simple generate
            generated = generate(model, prompt_text, vocab, 
                                       max_length=max_generation_length,
                                       device=device)
            
            generated_tokens = generated['tokens']
            
            # Calculate perplexity for generated sequence
            if len(generated_tokens) > 0:
                perplexity = calculate_perplexity(model, prompt_tokens, 
                                                  generated_tokens, device)
                all_perplexities.append(perplexity)
            
            # Prepare for BLEU calculation
            reference_text = vocab.decode(reference_tokens)
            generated_text = generated['text']
            
            all_references.append([reference_text])  # BLEU expects list of references
            all_predictions.append(generated_text)
    
    # Calculate average perplexity
    avg_perplexity = np.mean(all_perplexities) if all_perplexities else float('inf')
    
    # Calculate BLEU score
    bleu_results = bleu_metric.compute(predictions=all_predictions, 
                                       references=all_references)
    
    results = {
        'avg_perplexity': avg_perplexity,
        'bleu_score': bleu_results['bleu'],
        'num_samples_evaluated': len(all_perplexities),
        'bleu_details': {
            'precisions': bleu_results['precisions'],
            'brevity_penalty': bleu_results['brevity_penalty'],
            'length_ratio': bleu_results['length_ratio'],
        }
    }
    
    return results

def generate(model, prompt, vocab, max_length=50, device='cuda'):
    """Simple generation without caching"""
    model.eval()
    
    tokens = [vocab.word2idx[vocab.SOS_TOKEN]] + vocab.encode(prompt)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_length):
            if tokens_tensor.size(1) >= model.max_seq_len:
                break
            
            # Forward pass
            logits = model(tokens_tensor)
            
            # Sample next token
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            if next_token.item() == vocab.word2idx[vocab.EOS_TOKEN]:
                break
            
            tokens_tensor = torch.cat([tokens_tensor, next_token], dim=1)
            generated_tokens.append(next_token.item())
    
    return {
        'text': vocab.decode(tokens_tensor.squeeze(0).tolist()),
        'tokens': generated_tokens
    }

def calculate_perplexity(model, prompt_tokens, generated_tokens, device):
    """
    Calculate perplexity for generated tokens given prompt.
    """
    model.eval()
    
    # Combine prompt and generated tokens
    all_tokens = prompt_tokens + generated_tokens
    input_tensor = torch.tensor(all_tokens[:-1]).unsqueeze(0).to(device)
    target_tensor = torch.tensor(all_tokens[1:]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        
        # Only calculate loss on generated portion
        prompt_len = len(prompt_tokens) - 1
        gen_logits = logits[:, prompt_len:, :]
        gen_targets = target_tensor[:, prompt_len:]
        
        # Calculate cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            gen_logits.reshape(-1, gen_logits.size(-1)),
            gen_targets.reshape(-1),
            reduction='mean'
        )
        
        perplexity = torch.exp(loss).item()
    
    return perplexity

def kv_generate(model, prompt, vocab, max_length=50, device='cuda', use_kv_cache=False):
    """Generation with optional KV caching"""
    model.eval()
    
    tokens = [vocab.word2idx[vocab.SOS_TOKEN]] + vocab.encode(prompt)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    
    generated_tokens = []
    kv_caches = None
    
    with torch.no_grad():
        for step in range(max_length):
            if tokens_tensor.size(1) >= model.max_seq_len:
                break
            
            # For KV caching, we only process the last token after the first step
            if use_kv_cache and step > 0:
                # Only use the last token as input when using KV cache
                input_tokens = tokens_tensor[:, -1:]
            else:
                # Use all tokens for first step or when not using KV cache
                input_tokens = tokens_tensor
            
            # Forward pass
            if use_kv_cache:
                logits, new_kv_caches = model(input_tokens, kv_caches=kv_caches)
                kv_caches = new_kv_caches
            else:
                logits, _ = model(input_tokens)
            
            # Sample next token
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            if next_token.item() == vocab.word2idx[vocab.EOS_TOKEN]:
                break
            
            # Append the new token
            if use_kv_cache and step > 0:
                tokens_tensor = torch.cat([tokens_tensor, next_token], dim=1)
            else:
                # For first step with KV cache or without KV cache, we already have all tokens
                tokens_tensor = torch.cat([tokens_tensor, next_token], dim=1) if step == 0 else tokens_tensor
            
            generated_tokens.append(next_token.item())
    
    return {
        'text': vocab.decode(tokens_tensor.squeeze(0).tolist()),
        'tokens': generated_tokens
    }

def interactive_generation(model, vocab, device='cuda', max_length=50):
    """Interactive text generation - type 'quit' to exit"""
    
    while True:
        prompt = input("\nEnter prompt (or 'quit' to exit): ").strip()
        
        if prompt.lower() == 'quit':
            print("\nExiting interactive generation. Goodbye!")
            break
        
        if not prompt:
            print("Please enter a valid prompt.")
            continue
        
        print("\n" + "-"*70)
        print(f"Prompt: {prompt}")
        print("-"*70)
        
        try:
            result = kv_generate(
                model=model,
                prompt=prompt,
                vocab=vocab,
                max_length=max_length,
                device=device,
                use_kv_cache=True
            )
            
            print(f"\nGenerated text:\n{result['text']}")
            print(f"\nTokens generated: {len(result['tokens'])}/{max_length}")
            
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
            print("Please try again with a different prompt.")

# Run with forced 50 token generation
# # Example usage:
# if __name__ == "__main__":
#     # Load model, vocab, and validation dataset
#     # model = ...
#     # vocab = ...
#     # val_dataset = ...
    
#     # Run evaluation
#     results = evaluate_model(
#         model=model,
#         val_dataset=val_dataset,
#         vocab=vocab,
#         num_samples=50,
#         prompt_length=5,
#         max_generation_length=50,
#         device='cuda'
#     )
    