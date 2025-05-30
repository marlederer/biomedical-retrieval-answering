'''
Configuration settings for the BioASQ Reranker.
'''
import torch

# Data paths
TRAIN_DATA_PATH = "data/training13b.json" # Path to the main BioASQ training data
HARD_NEGATIVES_PATH = "data/hard_negatives.json" # Optional: Path to pre-computed hard negatives
BM25_OUTPUT_PATH = "data/bm25.json" # Path to BM25 output for inference
DENSE_OUTPUT_PATH = "data/dense.json" # Path to Dense retriever output for inference
VOCAB_PATH = "data/vocab.json" # Path to save/load vocabulary

# Model parameters
EMBEDDING_DIM = 100 # Dimension of word embeddings
N_KERNELS = 15 # Number of RBF kernels in KNRM
MAX_QUERY_LEN = 30 # Max length for query sequences
MAX_DOC_LEN = 200 # Max length for document sequences
MAX_DOCS_PER_QUERY = 100 # Max number of documents to consider per query

# Training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
MARGIN_LOSS = 0.5 # Margin for the triplet margin loss
SAVE_MODEL_PATH = "models/knrm_model.pth" # Path to save trained models

# Evaluation parameters
MRR_K = 10 # K for MRR@K calculation

# Tokenizer settings
TOKENIZER_TYPE = "nltk" # or "spacy"
MIN_WORD_FREQ = 5 # Minimum word frequency to be included in vocabulary
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
