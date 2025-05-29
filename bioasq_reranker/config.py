'''
Configuration settings for the BioASQ Reranker.
'''
import torch

# Data paths
TRAIN_DATA_PATH = "bioasq_reranker/data/training13b.json" 
HARD_NEGATIVES_PATH = "bioasq_reranker/data/hard_negatives.json" 
BM25_OUTPUT_PATH = "bioasq_reranker/data/bm25.json" 
DENSE_OUTPUT_PATH = "bioasq_reranker/data/dense.json"
VOCAB_PATH = "bioasq_reranker/data/vocab.json" 

# Model parameters
EMBEDDING_DIM = 100 # Dimension of word embeddings
N_KERNELS = 21 # Number of RBF kernels in KNRM
MAX_QUERY_LEN = 30 # Max length for query sequences
MAX_DOC_LEN = 200 # Max length for document sequences

# Training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
# 1. 1e-5; 2. 5e - 5; 3.
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
MARGIN_LOSS = 0.5 # Margin for the triplet margin loss
SAVE_MODEL_PATH = "models/knrm_model.pth" # Path to save trained models

# Evaluation parameters
MRR_K = 10 # K for MRR@K calculation

# Tokenizer settings
TOKENIZER_TYPE = "basic_whitespace" # or "nltk", "spacy"
MIN_WORD_FREQ = 5 # Minimum word frequency to be included in vocabulary
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# API Client specific
BIOASQ_API_ENDPOINT_URL = "http://bioasq.org:8000/pubmed"
API_NUM_INITIAL_CANDIDATES = 100 # Number of candidates to fetch from BioASQ API
RERANK_TOP_K_OUTPUT = 10 # Number of top documents to output after reranking for single query
