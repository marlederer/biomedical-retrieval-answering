'''
Utility functions for tokenization, vocabulary building, and padding.
'''
import json
import nltk
from collections import Counter
import torch
from nltk.tokenize import word_tokenize

import config

class Vocabulary:
    def __init__(self, pad_token=config.PAD_TOKEN, unk_token=config.UNK_TOKEN):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        self.word_counts = Counter()

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_counts[word] += 1

    def build_vocab(self, texts, min_freq=config.MIN_WORD_FREQ):
        for text in texts:
            for word in text: # Assuming texts are already tokenized
                self.add_word(word)
        
        # Prune vocabulary based on min_freq
        # Keep PAD and UNK tokens
        pruned_word2idx = {self.pad_token: 0, self.unk_token: 1}
        pruned_idx2word = {0: self.pad_token, 1: self.unk_token}
        
        current_idx = 2
        for word, count in self.word_counts.items():
            if count >= min_freq:
                if word not in pruned_word2idx: # Avoid re-adding PAD/UNK if they met freq
                    pruned_word2idx[word] = current_idx
                    pruned_idx2word[current_idx] = word
                    current_idx += 1
        
        self.word2idx = pruned_word2idx
        self.idx2word = pruned_idx2word
        # Update word_counts to reflect pruned vocab (optional, mainly for inspection)
        self.word_counts = Counter({word: self.word_counts[word] for word in self.word2idx if word in self.word_counts})

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = data['word2idx']
        # Convert loaded idx2word keys from string back to int
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        return vocab

def tokenize_text(text, tokenizer_type=config.TOKENIZER_TYPE):
    '''Tokenizes a single text string.'''
    text = text.lower() # Convert to lowercase
    if tokenizer_type == "nltk":
        return word_tokenize(text)
    elif tokenizer_type == "basic_whitespace":
        return text.split()
    # Add other tokenizers like SpaCy if needed
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

def texts_to_sequences(texts_tokenized, vocab):
    '''Converts a list of tokenized texts to sequences of indices.'''
    sequences = []
    for tokens in texts_tokenized:
        seq = [vocab.word2idx.get(token, vocab.word2idx[vocab.unk_token]) for token in tokens]
        sequences.append(seq)
    return sequences

def pad_sequence(sequence, max_len, pad_value=0):
    '''Pads a single sequence to max_len.'''
    padded_seq = sequence[:max_len] + [pad_value] * (max_len - len(sequence))
    return padded_seq

def create_mask_from_sequence(sequence, pad_idx=0):
    '''Creates a mask for a padded sequence. 1 for non-pad, 0 for pad.'''
    return [1 if token_id != pad_idx else 0 for token_id in sequence]


if __name__ == '__main__':
    # Test scenario for the utility functions )
    sample_texts = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    ]

    # Tokenize
    tokenized_texts = [tokenize_text(text) for text in sample_texts]
    print("Tokenized Texts:", tokenized_texts)

    # Build Vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(tokenized_texts, min_freq=1) # min_freq=1 for small example
    print("Vocabulary (word2idx):", vocab.word2idx)
    print("Vocabulary size:", len(vocab))

    # Save and Load Vocabulary
    vocab.save("data/sample_vocab.json")
    loaded_vocab = Vocabulary.load("data/sample_vocab.json")
    print("Loaded Vocabulary size:", len(loaded_vocab))

    # Convert to Sequences
    sequences = texts_to_sequences(tokenized_texts, vocab)
    print("Sequences:", sequences)

    # Pad Sequences
    max_len_example = 10
    padded_sequences = [pad_sequence(seq, max_len_example) for seq in sequences]
    print("Padded Sequences:", padded_sequences)

    # Create Masks
    masks = [create_mask_from_sequence(seq) for seq in padded_sequences]
    print("Masks:", masks)

    # Convert to Tensors (example for one sequence)
    query_tensor = torch.tensor(padded_sequences[0], dtype=torch.long)
    query_mask_tensor = torch.tensor(masks[0], dtype=torch.float32)
    print("Query Tensor:", query_tensor)
    print("Query Mask Tensor:", query_mask_tensor)

