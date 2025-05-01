# dense_retrieval.py
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import torch.nn.functional as F

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models globally (you can move to init if you want faster runs)
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

def encode_query(query):
    inputs = question_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(question_encoder.device)
    with torch.no_grad():
        embedding = question_encoder(**inputs).pooler_output
    return embedding.cpu()

def encode_contexts(contexts):
    embeddings = []
    for context in contexts:
        inputs = context_tokenizer(context, return_tensors="pt", truncation=True, max_length=512).to(context_encoder.device)
        with torch.no_grad():
            embedding = context_encoder(**inputs).pooler_output
        embeddings.append(embedding.cpu())
    return torch.vstack(embeddings)

def rank_with_dense_retrieval(query_embedding, context_embeddings, top_k=10):
    # Calculate similarities
    scores = torch.matmul(context_embeddings, query_embedding.squeeze(0))
    top_k_indices = torch.topk(scores, top_k).indices
    return top_k_indices.tolist()
