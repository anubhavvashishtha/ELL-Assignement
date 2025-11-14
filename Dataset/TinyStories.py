
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class TinyStoriesDataset(Dataset):
    def __init__(self, texts, vocab, context_length, max_samples=None):
        self.vocab = vocab
        self.context_length = context_length
        self.sequences = []

        print("Preparing dataset...")
        for idx, text in enumerate(tqdm(texts)):
            if max_samples and idx >= max_samples:
                break

            tokens = [vocab.word2idx[vocab.SOS_TOKEN]] + vocab.encode(text) + [vocab.word2idx[vocab.EOS_TOKEN]]

            for i in range(len(tokens) - 1):
                end_idx = min(i + context_length + 1, len(tokens))
                seq = tokens[i:end_idx]

                if len(seq) < context_length + 1:
                    seq = seq + [vocab.word2idx[vocab.PAD_TOKEN]] * (context_length + 1 - len(seq))

                self.sequences.append(seq)

        print(f"Created {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

