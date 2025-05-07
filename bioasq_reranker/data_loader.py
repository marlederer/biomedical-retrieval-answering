import json
import random
from tqdm import tqdm
import logging

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def load_bioasq_json(filepath):
    """Loads BioASQ JSON data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded BioASQ data from {filepath}")
        return data.get('questions', [])
    except FileNotFoundError:
        logger.error(f"Error: BioASQ file not found at {filepath}. Please provide the correct path.")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {filepath}. File might be corrupted.")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {filepath}: {e}")
        return []

def build_passage_corpus_from_bioasq(questions_data, min_snippet_len=10):
    """
    Creates a list of unique passages (from snippets) to serve as a corpus for negative sampling.
    A more robust corpus would involve all actual documents from BioASQ.
    """
    all_passages = set()
    if not questions_data:
        logger.warning("No questions data provided to build passage corpus.")
        return []

    for q_entry in questions_data:
        if 'snippets' in q_entry:
            for snippet in q_entry['snippets']:
                passage_text = snippet.get('text', '').strip()
                if len(passage_text) >= min_snippet_len: # Basic quality check
                    all_passages.add(passage_text)
    
    corpus = list(all_passages)
    if not corpus:
        logger.warning("Passage corpus is empty. This might happen if snippets are missing or too short.")
        # Fallback to very basic dummy passages if corpus is empty to prevent downstream errors
        # This indicates a problem with the input data or its structure for snippets.
        return ["This is a fallback passage example.", "Another fallback text for corpus generation."]
    
    logger.info(f"Built a passage corpus with {len(corpus)} unique snippets.")
    return corpus

def create_training_samples(questions_data, passage_corpus, num_neg_samples_per_positive=1, max_questions=None):
    """
    Creates training samples (query, passage, label) for the re-ranker.
    Labels: 1 for positive (relevant), 0 for negative (irrelevant).
    """
    train_samples = []
    if not questions_data:
        logger.warning("No questions data provided for creating training samples.")
        return []
    if not passage_corpus:
        logger.warning("Passage corpus is empty. Cannot generate negative samples effectively.")
        # Potentially use a very small default corpus if this happens, though it's not ideal
        passage_corpus = ["Default negative sample A.", "Default negative sample B."]


    question_subset = questions_data
    if max_questions is not None and max_questions < len(questions_data):
        logger.info(f"Using a subset of {max_questions} questions for training sample generation.")
        question_subset = random.sample(questions_data, max_questions)


    for q_entry in tqdm(question_subset, desc="Generating Training Samples"):
        query_text = q_entry.get('body', '').strip()
        if not query_text:
            logger.warning(f"Skipping question ID {q_entry.get('id', 'N/A')} due to empty body.")
            continue

        positive_passages_texts = []
        if 'snippets' in q_entry:
            for snippet in q_entry['snippets']:
                passage_text = snippet.get('text', '').strip()
                # Consider checking snippet document URL against relevant documents if available
                # For now, assume all snippets in a question are relevant to that question
                if passage_text:
                    positive_passages_texts.append(passage_text)
        
        if not positive_passages_texts:
            # logger.debug(f"No positive snippets found for question ID {q_entry.get('id', 'N/A')}: {query_text[:50]}...")
            continue # Skip if no positive examples

        for pos_passage in positive_passages_texts:
            train_samples.append({'query': query_text, 'passage': pos_passage, 'label': 1})

            # Generate negative samples
            # Current strategy: Random negatives from the corpus.
            # TODO: Implement hard negative mining for better performance.
            # Hard negatives would be passages retrieved by a first-stage retriever (e.g., BM25)
            # for this query_text, but are NOT in positive_passages_texts.
            negs_added = 0
            attempts = 0
            max_attempts = num_neg_samples_per_positive * 10 # Try a bit harder to find distinct negatives
            
            if not passage_corpus: # Should not happen if fallback in build_passage_corpus_from_bioasq works
                logger.warning("Passage corpus is empty during negative sampling, skipping negatives for this positive.")
                continue

            while negs_added < num_neg_samples_per_positive and attempts < max_attempts:
                attempts += 1
                neg_passage = random.choice(passage_corpus)
                if neg_passage not in positive_passages_texts: # Ensure it's not accidentally a positive
                    train_samples.append({'query': query_text, 'passage': neg_passage, 'label': 0})
                    negs_added += 1
    
    if not train_samples:
        logger.warning("No training samples were generated. Check input data and parameters.")
    else:
        logger.info(f"Generated {len(train_samples)} training samples.")
    return train_samples


class ReRankerDataset(Dataset):
    def __init__(self, samples, tokenizer_name_or_path, max_length):
        self.samples = samples
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length
        logger.info(f"ReRankerDataset initialized with {len(samples)} samples.")
        if not samples:
             logger.warning("ReRankerDataset initialized with zero samples!")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        query = sample['query']
        passage = sample['passage']
        label = sample['label']

        # [CLS] query [SEP] passage [SEP]
        encoding = self.tokenizer.encode_plus(
            query,
            passage,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation='only_second', # Truncate passage if query+passage > max_length
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float) # For BCEWithLogitsLoss
        }

# --- Example Usage (for testing this file standalone) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy BioASQ JSON file for testing
    dummy_data_path = "dummy_bioasq.json"
    dummy_bioasq_content = {
        "questions": [
            {
                "id": "q1",
                "body": "What is foo?",
                "snippets": [{"text": "Foo is a metasyntactic variable."}, {"text": "Foo Fighters is a band."}]
            },
            {
                "id": "q2",
                "body": "Tell me about bar.",
                "snippets": [{"text": "Bar is often used with foo."}]
            },
            {
                "id": "q3", # Question with no snippets
                "body": "What about baz?",
                "documents": ["some_doc_url"]
            }
        ]
    }
    with open(dummy_data_path, 'w') as f:
        json.dump(dummy_bioasq_content, f)

    questions = load_bioasq_json(dummy_data_path)
    if questions:
        corpus = build_passage_corpus_from_bioasq(questions)
        if corpus:
            training_samples = create_training_samples(questions, corpus, num_neg_samples_per_positive=1)
            logger.info(f"First sample: {training_samples[0] if training_samples else 'None'}")

            # Test Dataset
            tokenizer_name = "bert-base-uncased" # Using a generic one for quick test
            dataset = ReRankerDataset(training_samples, tokenizer_name, max_length=128)
            if len(dataset) > 0:
                logger.info(f"Dataset sample 0: {dataset[0]}")
            else:
                logger.warning("Dataset created but is empty.")
        else:
            logger.warning("Corpus is empty, cannot proceed with training sample generation for test.")
    else:
        logger.warning("No questions loaded, cannot proceed with test.")

    # Clean up dummy file
    import os
    os.remove(dummy_data_path)