import argparse
import os
import json
import logging

import torch
from transformers import AutoTokenizer

from model_arch import CrossEncoderReRanker # Assuming model_arch.py contains your CrossEncoderReRanker class

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def rerank_passages(query, candidate_passages, model, tokenizer, device, max_seq_length=512, batch_size=16):
    """
    Re-ranks a list of candidate passages for a given query using the trained model.
    Returns a list of (passage, score) tuples, sorted by score in descending order.
    """
    if not candidate_passages:
        return []

    model.eval()
    scores = []
    
    # Process in batches for efficiency if many candidate_passages
    for i in range(0, len(candidate_passages), batch_size):
        batch_passages = candidate_passages[i:i+batch_size]
        
        inputs = tokenizer(
            [query] * len(batch_passages), # Repeat query for each passage in batch
            batch_passages,
            padding=True,
            truncation='only_second', # Truncate passage if query+passage > max_length
            max_length=max_seq_length,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            # If using BCEWithLogitsLoss during training, logits are raw scores.
            # Higher logit = more relevant. Sigmoid can convert to probability if needed.
            current_scores_tensor = logits.squeeze(-1) # Squeeze to remove last dim if it's 1
            
            # Option 1: Use raw logits for ranking (often fine)
            current_scores = current_scores_tensor.cpu().numpy()
            
            # Option 2: Use probabilities (sigmoid of logits)
            # current_scores = torch.sigmoid(current_scores_tensor).cpu().numpy()

            scores.extend(current_scores)

    scored_passages = list(zip(candidate_passages, scores))
    # Sort by score in descending order (higher score means more relevant)
    scored_passages.sort(key=lambda x: x[1], reverse=True)
    return scored_passages

def main():
    parser = argparse.ArgumentParser(description="Re-rank passages using a trained BioASQ model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the saved model (pytorch_model.bin, config.json, tokenizer files)")
    parser.add_argument("--query", type=str, required=True, help="The query string")
    parser.add_argument("--passages", nargs='+', required=True, help="List of candidate passages (strings)")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length for tokenization. Tries to load from training_args.json if not set.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    if not os.path.exists(args.model_path) or \
       not os.path.exists(os.path.join(args.model_path, "pytorch_model.bin")) or \
       not os.path.exists(os.path.join(args.model_path, "config.json")): # Check for essential files
        logger.error(f"Model path {args.model_path} does not seem to contain a valid saved model. "
                     "Ensure pytorch_model.bin, config.json, and tokenizer files are present.")
        return

    try:
        # The tokenizer should have been saved with `save_pretrained` in the model_path
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        # We need the original model name or path to initialize AutoModel within CrossEncoderReRanker correctly
        # Let's try to load it from training_args.json if available
        original_model_name = None
        training_args_path = os.path.join(args.model_path, "training_args.json")
        if os.path.exists(training_args_path):
            with open(training_args_path, 'r') as f:
                train_args = json.load(f)
            original_model_name = train_args.get("model_name")
            if args.max_seq_length is None: # If not set by user, try to use from training
                args.max_seq_length = train_args.get("max_seq_length", 512)


        if not original_model_name:
            logger.warning("Could not determine original base model name from training_args.json. "
                           "The CrossEncoderReRanker might not load the BERT base correctly if it's not "
                           "part of the state_dict. Ensure your model was saved to include base BERT weights "
                           "or specify --bert_model_name if needed for a different loading strategy.")
            # Fallback: assume the model_path itself can be used if it's a full Hugging Face model save
            # This might not be true if only the state_dict of the CrossEncoderReRanker was saved
            # without the base BERT model's name.
            # For the current CrossEncoderReRanker, it needs the name of the BERT base.
            # A robust way is to save the 'model_name' used during training.
            # Let's assume the user provides a valid HF model identifier if training_args is missing.
            # This part is tricky if only a state_dict is saved.
            # The `model_arch.CrossEncoderReRanker` expects a HuggingFace model name for `AutoModel.from_pretrained`.
            # So, the `model_path` for `AutoModel` should be the original base model name,
            # and then we load the `state_dict` for the whole `CrossEncoderReRanker`.
            # This is why saving `training_args.json` with `model_name` is crucial.
            logger.error("Cannot determine the base BERT model name (e.g., from training_args.json). "
                         "The CrossEncoderReRanker class needs this to initialize the BERT component. "
                         "Please ensure 'model_name' is in training_args.json or modify inference script.")
            return

        model_state_dict_path = os.path.join(args.model_path, "pytorch_model.bin")
        
        model = CrossEncoderReRanker(model_name_or_path=original_model_name)
        model.load_state_dict(torch.load(model_state_dict_path, map_location=device))
        model.to(device)
        logger.info(f"Model and tokenizer loaded successfully from {args.model_path} (using base: {original_model_name}).")

    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        return

    # --- Perform Re-ranking ---
    if args.max_seq_length is None: # Fallback if still not set
        args.max_seq_length = 512
        logger.info(f"Using default max_seq_length: {args.max_seq_length}")

    reranked_results = rerank_passages(
        args.query,
        args.passages,
        model,
        tokenizer,
        device,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size
    )

    # --- Display Results ---
    print(f"\nQuery: {args.query}")
    print("Re-ranked Passages (higher score is more relevant):")
    if reranked_results:
        for i, (passage, score) in enumerate(reranked_results):
            print(f"{i+1}. Score: {score:.4f}\tPassage: {passage}")
    else:
        print("No passages were re-ranked (perhaps input was empty).")

if __name__ == "__main__":
    main()