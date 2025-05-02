# dense_retrieval.py
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models globally (you can move to init if you want faster runs)
question_encoder = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
context_encoder = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
question_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
context_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

def encode_query(query):
    inputs = question_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        embedding = question_encoder(**inputs).last_hidden_state[:, 0, :]
    return embedding

def encode_contexts(contexts, batch_size=32):
    all_embeddings = []
    context_encoder.eval()
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i+batch_size]
            inputs = context_tokenizer(
                batch_contexts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)
            
            batch_embeddings = context_encoder(**inputs).last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
            
            # Move to CPU to save GPU memory if needed
            all_embeddings.append(batch_embeddings.cpu())
    # After batching, stack all embeddings into one big tensor
    return torch.vstack(all_embeddings)  # (num_contexts, hidden_dim)

def rank_with_dense_retrieval(query_embedding, context_embeddings, top_k=10):
    # Calculate similarities
    scores = torch.matmul(context_embeddings.to(device), query_embedding.to(device).squeeze(0))
    top_k_indices = torch.topk(scores, top_k).indices
    return top_k_indices.tolist()
