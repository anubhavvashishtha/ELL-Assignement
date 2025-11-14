import  re , json
from collections import  Counter
import torch

class Vocabulary:
    def __init__(self, fasttext_model=None):
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.PAD_TOKEN = '<pad>'
        self.SOS_TOKEN = '<sos>'
        self.EOS_TOKEN = '<eos>'
        self.UNK_TOKEN = '<unk>'
        self.add_word(self.PAD_TOKEN)
        self.add_word(self.SOS_TOKEN)
        self.add_word(self.EOS_TOKEN)
        self.add_word(self.UNK_TOKEN)
        self.fasttext_model = fasttext_model

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_counts[word] += 1

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
                for token in tokens]

    def decode(self, indices):
        words = []
        for idx in indices:
            if idx in [self.word2idx[self.PAD_TOKEN], self.word2idx[self.SOS_TOKEN]]:
                continue
            if idx == self.word2idx[self.EOS_TOKEN]:
                break
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return ' '.join(words)

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
        return tokens

    def create_embedding_matrix(self):
        embedding_matrix = torch.randn(len(self.word2idx), 300) * 0.01
        if self.fasttext_model is not None:
            found = 0
            for word, idx in self.word2idx.items():
                if word in self.fasttext_model:
                    embedding_matrix[idx] = torch.tensor(self.fasttext_model[word])
                    found += 1
            print(f"Found {found}/{len(self.word2idx)} words in FastText")
        return embedding_matrix

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': {int(k): v for k, v in self.idx2word.items()},
                'word_counts': dict(self.word_counts)
            }, f)

    @classmethod
    def load(cls, path, fasttext_model=None):
        vocab = cls(fasttext_model)
        with open(path, 'r') as f:
            data = json.load(f)
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        vocab.word_counts = Counter(data['word_counts'])
        return vocab
