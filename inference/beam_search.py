import torch
import time
import evaluate 
import os
import json
import matplotlib.pyplot as plt
from inference.inference import generate

def beam_generate(model, prompt, vocab, max_length=50, beam_width=5, device='cuda'):
    """
    Generate text using beam search decoding.
    
    Args:
        model: The language model
        prompt: Input text prompt
        vocab: Vocabulary object
        max_length: Maximum generation length
        beam_width: Number of beams to maintain
        device: Device to run on
    
    Returns:
        Dictionary with generated text and tokens
    """
    model.eval()
    
    # Initialize with SOS token + prompt
    initial_tokens = [vocab.word2idx[vocab.SOS_TOKEN]] + vocab.encode(prompt)
    
    # Initialize beams: (token_sequence, log_prob_score, is_completed)
    beams = [(
        torch.tensor(initial_tokens, device=device).unsqueeze(0),
        0.0,  # log probability score
        False  # not completed
    )]
    
    with torch.no_grad():
        for step in range(max_length):
            # If all beams are completed, break early
            if all(completed for _, _, completed in beams):
                break
            
            # Collect all active beams for processing
            candidates = []
            
            for beam_tokens, beam_score, beam_completed in beams:
                if beam_completed:
                    # Keep completed beams as they are
                    candidates.append((beam_tokens, beam_score, True))
                    continue
                
                # Forward pass
                if beam_tokens.size(1) >= getattr(model, 'max_seq_len', 1024):  # Added fallback
                    candidates.append((beam_tokens, beam_score, True))
                    continue
                
                logits = model(beam_tokens)
                
                # Get next token probabilities
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                log_probs = torch.log(probs + 1e-8)
                
                # Get top-k candidates
                top_log_probs, top_indices = torch.topk(log_probs.squeeze(0), beam_width)
                
                for i in range(beam_width):
                    next_token = top_indices[i].unsqueeze(0).unsqueeze(0)
                    new_score = beam_score + top_log_probs[i].item()
                    
                    # Check if sequence is completed
                    is_completed = (next_token.item() == vocab.word2idx[vocab.EOS_TOKEN])
                    
                    # Append new token to sequence
                    new_tokens = torch.cat([beam_tokens, next_token], dim=1)
                    
                    candidates.append((new_tokens, new_score, is_completed))
            
            # Sort candidates by score and select top beam_width
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
    
    # Return the best beam
    best_tokens = beams[0][0].squeeze(0).tolist()
    
    # Extract only the generated tokens (after initial prompt + SOS)
    generated_tokens = best_tokens[len(initial_tokens):]
    
    # Decode only the generated portion
    generated_text = vocab.decode(generated_tokens)
    
    return {
        'text': generated_text,
        'tokens': generated_tokens,
        'score': beams[0][1],
        'full_sequence': best_tokens  # For debugging
    }

def evaluate_decoding_strategies(model, val_prompts, val_targets, vocab, device='cuda'):
    """
    Compare decoding strategies (greedy, beam_5, beam_10) 
    in terms of speed and BLEU score.

    Args:
        model: Language model
        val_prompts: List of prompt strings
        val_targets: List of ground-truth target strings
        vocab: Vocabulary object
        device: 'cuda' or 'cpu'
    """
    bleu = evaluate.load("bleu")

    strategies = [
        ('greedy', lambda prompt: generate(model, prompt, vocab, device=device)),
        ('beam_5', lambda prompt: beam_generate(model, prompt, vocab, beam_width=5, device=device)),
        ('beam_10', lambda prompt: beam_generate(model, prompt, vocab, beam_width=10, device=device))
    ]

    results = {}

    for strategy_name, generate_fn in strategies:
        print(f"\nEvaluating {strategy_name} decoding...")

        times = []
        all_outputs = []
        references = []
        predictions = []

        for prompt, target in zip(val_prompts, val_targets):
            start_time = time.time()
            output = generate_fn(prompt)
            end_time = time.time()

            generation_time = end_time - start_time
            times.append(generation_time)
            all_outputs.append(output)

            predictions.append(output['text'])
            references.append([target])  # BLEU expects list of lists

        # Compute BLEU
        bleu_score = bleu.compute(predictions=predictions, references=references)['bleu']

        # Compute speed metrics
        avg_time = sum(times) / len(times)
        total_tokens = sum(len(o['tokens']) for o in all_outputs)
        tokens_per_second = total_tokens / sum(times)

        results[strategy_name] = {
            'avg_time': avg_time,
            'tokens_per_second': tokens_per_second,
            'bleu': bleu_score
        }

    return results

def get_validation_prompts_and_targets(val_loader, vocab, num_samples=20):
    """
    Extract the first 5 words as prompts and the rest as ground-truth continuations 
    from the validation dataset.
    
    Args:
        val_loader: DataLoader for validation data
        vocab: Vocabulary object
        num_samples: Number of samples to retrieve (default: 20)
    
    Returns:
        prompts: List[str] - first 5 words of each sequence
        targets: List[str] - remaining words of each sequence
    """
    prompts = []
    targets = []
    
    for batch in val_loader:
        for seq in batch:
            if len(prompts) >= num_samples:
                return prompts, targets
            
            seq = seq.numpy()
            
            # Decode full sequence (excluding special tokens)
            words = []
            for token_id in seq:
                word = vocab.idx2word.get(token_id, vocab.UNK_TOKEN)
                if word not in [vocab.PAD_TOKEN, vocab.SOS_TOKEN, vocab.EOS_TOKEN]:
                    words.append(word)
            
            if len(words) == 0:
                continue  # skip empty sequences
            
            # Prompt = first 5 words
            prompt_words = words[:5]
            # Target = remaining words
            target_words = words[5:] if len(words) > 5 else [""]

            prompt = ' '.join(prompt_words)
            target = ' '.join(target_words)
            
            prompts.append(prompt)
            targets.append(target)
    
    return prompts, targets

def save_beam_results(results, save_dir="results/inference/beam"):
    """
    Save beam search evaluation results and create comparative plots with proper spacing for labels.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Save JSON ---
    result_path = os.path.join(save_dir, "beam_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Results saved to {result_path}")

    # --- Prepare data for plotting ---
    strategies = list(results.keys())
    tokens_per_sec = [results[s]["tokens_per_second"] for s in strategies]
    avg_times = [results[s]["avg_time"] for s in strategies]
    bleu_scores = [results[s]["bleu"] for s in strategies]

    plt.style.use('default')
    plt.rcParams["font.size"] = 11

    def add_value_labels(ax, bars, fmt="{:.4f}"):
        """Add labels above bars without overlapping top."""
        y_max = max(bar.get_height() for bar in bars)
        ax.set_ylim(0, y_max * 1.2)  # Add headroom
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + y_max * 0.02,
                fmt.format(bar.get_height()),
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

    # --- 1️⃣ Tokens per Second Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, tokens_per_sec, color=['#E74C3C', '#2ECC71', '#3498DB'], alpha=0.85)
    ax.set_title("Tokens per Second (Speed)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Tokens/s", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    add_value_labels(ax, bars, fmt="{:.1f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tokens_per_second.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # --- 2️⃣ Average Time Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, avg_times, color=['#E74C3C', '#2ECC71', '#3498DB'], alpha=0.85)
    ax.set_title("Average Generation Time per Sample", fontsize=15, fontweight="bold")
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    add_value_labels(ax, bars, fmt="{:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_time.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # --- 3️⃣ BLEU Score Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, bleu_scores, color=['#E74C3C', '#2ECC71', '#3498DB'], alpha=0.85)
    ax.set_title("BLEU Score Comparison", fontsize=15, fontweight="bold")
    ax.set_ylabel("BLEU Score", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    add_value_labels(ax, bars, fmt="{:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bleu_score.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print("📊 Plots saved:")
    print(f"   • {save_dir}/tokens_per_second.png")
    print(f"   • {save_dir}/avg_time.png")
    print(f"   • {save_dir}/bleu_score.png")

def run_evaluation(model, vocab, val_prompts, val_targets):
    """
    Run evaluation and print results in a formatted table.
    """
    results = evaluate_decoding_strategies(model, val_prompts, val_targets, vocab)

    print("\n" + "=" * 75)
    print("DECODING STRATEGY COMPARISON")
    print("=" * 75)
    print(f"{'Strategy':<12} {'Tokens/s':<12} {'Avg Time (s)':<15} {'BLEU Score':<10}")
    print("-" * 75)

    for strategy, metrics in results.items():
        print(f"{strategy:<12} {metrics['tokens_per_second']:<12.2f} "
              f"{metrics['avg_time']:<15.4f} {metrics['bleu']:<10.4f}")

    print("=" * 75)

    return results


# results = run_evaluation(
#     model=model,
#     vocab=vocab,
#     val_prompts=val_prompts,
#     val_targets=val_targets
# )

# save_beam_results(results, save_dir="results/inference/beam")