from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load question encoder and tokenizer
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Load context encoder and tokenizer
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Example question and context
question = "What causes hypertension?"
# Example title and abstract
title = "Hypertension and its causes"
abstract = "Hypertension is often caused by a combination of genetic and lifestyle factors. It can lead to serious health complications if untreated."

# Concatenate title and abstract
context = "Title: "+ title + " Abstract: " + abstract

# Tokenize and encode the question
question_inputs = question_tokenizer(question, return_tensors="pt").to(device)
question_embedding = question_encoder(**question_inputs).pooler_output  # Shape: [1, hidden_size]

# Tokenize and encode the passage
context_inputs = context_tokenizer(context, return_tensors="pt").to(device)
context_embedding = context_encoder(**context_inputs).pooler_output  # Shape: [1, hidden_size]

# Compute cosine similarity
cos_sim = F.cosine_similarity(question_embedding, context_embedding)
print(f"Cosine Similarity: {cos_sim.item()}")
