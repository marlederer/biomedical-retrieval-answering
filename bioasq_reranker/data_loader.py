# data_loader.py
import json
import random
from tqdm import tqdm
import logging
import os # Added os

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
    """
    all_passages = set()
    if not questions_data:
        logger.warning("No questions data provided to build passage corpus.")
        return []

    for q_entry in questions_data:
        if 'snippets' in q_entry:
            for snippet in q_entry['snippets']:
                passage_text = snippet.get('text', '').strip()
                if len(passage_text) >= min_snippet_len:
                    all_passages.add(passage_text)
    
    corpus = list(all_passages)
    if not corpus:
        logger.warning("Passage corpus is empty. This might happen if snippets are missing or too short.")
        return ["This is a fallback passage example.", "Another fallback text for corpus generation."]
    
    logger.info(f"Built a passage corpus with {len(corpus)} unique snippets.")
    return corpus

def load_hard_negatives(filepath):
    """Loads pre-computed hard negatives from a JSON file."""
    if not filepath or not os.path.exists(filepath):
        logger.info("No hard negatives file provided or file does not exist. Will use random negatives only.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            hard_negatives_map = json.load(f) # Expected format: {"question_id": ["neg_passage1", ...]}
        logger.info(f"Successfully loaded {len(hard_negatives_map)} question entries from hard negatives file: {filepath}")
        return hard_negatives_map
    except Exception as e:
        logger.error(f"Error loading hard negatives from {filepath}: {e}")
        return None

def create_training_samples(questions_data,
                            passage_corpus,
                            num_neg_samples_per_positive=1,
                            max_questions=None,
                            hard_negatives_map=None): # Added hard_negatives_map
    """
    Creates training samples (query, passage, label, type) for the re-ranker.
    Labels: 1 for positive, 0 for negative.
    Type: 'positive', 'hard_negative', 'random_negative'.
    """
    train_samples = []
    if not questions_data:
        logger.warning("No questions data provided for creating training samples.")
        return []
    
    # Fallback for passage_corpus if it's somehow empty (should be handled in build_passage_corpus_from_bioasq)
    if not passage_corpus:
        logger.warning("Passage corpus is empty. Random negative sampling will be ineffective.")
        passage_corpus = ["Default random negative A.", "Default random negative B."]


    question_subset = questions_data
    if max_questions is not None and 0 < max_questions < len(questions_data):
        logger.info(f"Using a subset of {max_questions} questions for training sample generation.")
        # Ensure determinism if a seed is set elsewhere, or shuffle if desired for subset selection
        # random.shuffle(questions_data) # Uncomment if random subset desired each time
        question_subset = questions_data[:max_questions] # Or random.sample(questions_data, max_questions)


    for q_entry in tqdm(question_subset, desc="Generating Training Samples"):
        query_text = q_entry.get('body', '').strip()
        q_id = q_entry.get('id') # Need question ID for hard negatives map

        if not query_text or not q_id:
            logger.warning(f"Skipping question with missing body or ID: {q_entry}")
            continue

        positive_passages_texts = []
        if 'snippets' in q_entry:
            for snippet in q_entry['snippets']:
                passage_text = snippet.get('text', '').strip()
                if passage_text:
                    positive_passages_texts.append(passage_text)
        
        if not positive_passages_texts:
            continue

        for pos_passage in positive_passages_texts:
            train_samples.append({'query': query_text, 'passage': pos_passage, 'label': 1, 'type': 'positive'})

            # Negative samples
            negs_to_add_total = num_neg_samples_per_positive
            negs_added_count = 0

            # 1. Add Hard Negatives (if available)
            current_query_hard_negatives = []
            if hard_negatives_map:
                current_query_hard_negatives = hard_negatives_map.get(q_id, [])
            
            if current_query_hard_negatives:
                # Shuffle to pick different hard negatives if more are available than needed
                # This is useful if num_neg_samples_per_positive < len(current_query_hard_negatives)
                random.shuffle(current_query_hard_negatives) 
                for hn_passage in current_query_hard_negatives:
                    if negs_added_count < negs_to_add_total:
                        # Ensure hard negative is not accidentally a positive (should be pre-filtered but double check)
                        if hn_passage not in positive_passages_texts: # Should be true by construction of HN
                             train_samples.append({'query': query_text, 'passage': hn_passage, 'label': 0, 'type': 'hard_negative'})
                             negs_added_count += 1
                    else:
                        break
            
            # 2. Fill remaining with Random Negatives
            random_negs_needed = negs_to_add_total - negs_added_count
            if random_negs_needed > 0:
                attempts = 0
                # Max attempts to find random negatives.
                # More attempts needed if passage_corpus is small or has many overlaps with positives/hard_negs.
                max_attempts_random = random_negs_needed * 20 
                
                # Create a set of already used negatives for this query to improve diversity of randoms
                used_negatives_for_this_query = set(s['passage'] for s in train_samples if s['query'] == query_text and s['label'] == 0)

                while negs_added_count < negs_to_add_total and attempts < max_attempts_random:
                    if not passage_corpus: # Should ideally not happen with fallback
                        logger.warning(f"Passage corpus depleted/empty during random negative sampling for q_id {q_id}.")
                        break
                    attempts += 1
                    rand_neg_passage = random.choice(passage_corpus)
                    
                    if rand_neg_passage not in positive_passages_texts and \
                       rand_neg_passage not in used_negatives_for_this_query:
                        train_samples.append({'query': query_text, 'passage': rand_neg_passage, 'label': 0, 'type': 'random_negative'})
                        negs_added_count += 1
                        used_negatives_for_this_query.add(rand_neg_passage)

    if not train_samples:
        logger.warning("No training samples were generated. Check input data and parameters.")
    else:
        pos_count = sum(1 for s in train_samples if s['label'] == 1)
        hard_neg_count = sum(1 for s in train_samples if s.get('type') == 'hard_negative')
        rand_neg_count = sum(1 for s in train_samples if s.get('type') == 'random_negative')
        logger.info(f"Generated {len(train_samples)} samples: "
                    f"{pos_count} positives, {hard_neg_count} hard negatives, {rand_neg_count} random negatives.")
    return train_samples


class ReRankerDataset(Dataset): # No changes needed here
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

        encoding = self.tokenizer.encode_plus(
            query,
            passage,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation='only_second',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# --- Example Usage (for testing this file standalone) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    dummy_data_path = "dummy_bioasq_loader.json"
    dummy_hard_negs_path = "dummy_hard_negs_loader.json"

    dummy_bioasq_content = {
        "questions": [
            {"id": "q1", "body": "What is foo?", "snippets": [{"text": "Foo is a metasyntactic variable."}, {"text": "Foo Fighters is a band."}]},
            {"id": "q2", "body": "Tell me about bar.", "snippets": [{"text": "Bar is often used with foo."}]},
            {"id": "q3", "body": "What about baz?", "documents": ["some_doc_url"]}
        ]
    }
    dummy_hard_negs_content = {
        "q1": ["This is a hard negative for foo.", "Another tricky one for foo."],
        "q2": ["A confusing passage about bar."]
    }

    with open(dummy_data_path, 'w') as f: json.dump(dummy_bioasq_content, f)
    with open(dummy_hard_negs_path, 'w') as f: json.dump(dummy_hard_negs_content, f)

    questions = load_bioasq_json(dummy_data_path)
    hard_negs = load_hard_negatives(dummy_hard_negs_path)

    if questions:
        corpus = build_passage_corpus_from_bioasq(questions)
        # Add some more variety to corpus for random sampling test
        corpus.extend(["Random sample text 1", "Random sample text 2", "Yet another random snippet"])
        corpus = list(set(corpus))


        if corpus:
            training_samples = create_training_samples(questions, corpus, num_neg_samples_per_positive=2, hard_negatives_map=hard_negs)
            if training_samples:
                logger.info(f"First sample: {training_samples[0]}")
                for s in training_samples[:10]: logger.info(s)

                tokenizer_name = "bert-base-uncased"
                dataset = ReRankerDataset(training_samples, tokenizer_name, max_length=128)
                if len(dataset) > 0: logger.info(f"Dataset sample 0: {dataset[0]}")
                else: logger.warning("Dataset created but is empty.")
            else:
                logger.warning("No training samples generated in test.")
        else:
            logger.warning("Corpus is empty, cannot proceed with training sample generation for test.")
    else:
        logger.warning("No questions loaded, cannot proceed with test.")

    os.remove(dummy_data_path)
    os.remove(dummy_hard_negs_path)