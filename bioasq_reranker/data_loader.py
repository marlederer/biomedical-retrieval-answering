'''
Data loading and processing for BioASQ reranking task.
Handles loading BioASQ JSON, creating training triples, and preparing batches.
'''
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader

import config
from utils import tokenize_text, texts_to_sequences, pad_sequence, create_mask_from_sequence, Vocabulary

class BioASQTripletDataset(Dataset):
    '''
    Dataset for training a reranker using query, relevant_doc, non_relevant_doc triples.
    '''
    def __init__(self, queries, relevant_docs, non_relevant_docs, vocab, 
                 max_query_len=config.MAX_QUERY_LEN, max_doc_len=config.MAX_DOC_LEN):
        self.queries = queries
        self.relevant_docs = relevant_docs
        self.non_relevant_docs = non_relevant_docs
        self.vocab = vocab
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

        if not (len(queries) == len(relevant_docs) == len(non_relevant_docs)):
            raise ValueError("All lists (queries, relevant_docs, non_relevant_docs) must have the same length.")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_text = self.queries[idx]
        rel_doc_text = self.relevant_docs[idx]
        non_rel_doc_text = self.non_relevant_docs[idx]

        # Tokenize
        query_tokens = tokenize_text(query_text)
        rel_doc_tokens = tokenize_text(rel_doc_text)
        non_rel_doc_tokens = tokenize_text(non_rel_doc_text)

        # Convert to sequences of IDs
        query_seq = texts_to_sequences([query_tokens], self.vocab)[0]
        rel_doc_seq = texts_to_sequences([rel_doc_tokens], self.vocab)[0]
        non_rel_doc_seq = texts_to_sequences([non_rel_doc_tokens], self.vocab)[0]

        # Pad sequences
        q_padded = pad_sequence(query_seq, self.max_query_len, pad_value=self.vocab.word2idx[self.vocab.pad_token])
        rel_d_padded = pad_sequence(rel_doc_seq, self.max_doc_len, pad_value=self.vocab.word2idx[self.vocab.pad_token])
        non_rel_d_padded = pad_sequence(non_rel_doc_seq, self.max_doc_len, pad_value=self.vocab.word2idx[self.vocab.pad_token])

        # Create masks
        q_mask = create_mask_from_sequence(q_padded, pad_idx=self.vocab.word2idx[self.vocab.pad_token])
        rel_d_mask = create_mask_from_sequence(rel_d_padded, pad_idx=self.vocab.word2idx[self.vocab.pad_token])
        non_rel_d_mask = create_mask_from_sequence(non_rel_d_padded, pad_idx=self.vocab.word2idx[self.vocab.pad_token])

        return {
            'query_ids': torch.tensor(q_padded, dtype=torch.long),
            'query_mask': torch.tensor(q_mask, dtype=torch.float32),
            'rel_doc_ids': torch.tensor(rel_d_padded, dtype=torch.long),
            'rel_doc_mask': torch.tensor(rel_d_mask, dtype=torch.float32),
            'non_rel_doc_ids': torch.tensor(non_rel_d_padded, dtype=torch.long),
            'non_rel_doc_mask': torch.tensor(non_rel_d_mask, dtype=torch.float32)
        }

def load_bioasq_data(filepath):
    '''Loads BioASQ JSON data.'''
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('questions', [])

def create_training_triples(bioasq_questions, hard_negatives_map=None, num_neg_samples=1):
    '''
    Creates (query, relevant_doc, non_relevant_doc) triples for training.
    
    Args:
        bioasq_questions (list): List of question objects from BioASQ data.
        hard_negatives_map (dict, optional): A map from question ID to a list of hard negative document texts.
                                            If None, random negatives will be sampled from other relevant documents.
        num_neg_samples (int): Number of negative samples to generate per positive pair.

    Returns:
        tuple: (queries, relevant_docs, non_relevant_docs)
    '''
    queries = []
    relevant_docs_text = []
    non_relevant_docs_text = []

    all_doc_texts = []
    for q_data in bioasq_questions:
        if q_data.get('snippets'):
            for snippet in q_data['snippets']:
                all_doc_texts.append(snippet['text'])
    
    if not all_doc_texts:
        print("Warning: No documents found in bioasq_questions to sample random negatives from.")
        # Fallback: use a placeholder if no documents are available at all
        all_doc_texts = ["placeholder document text for negative sampling"] 

    for q_data in bioasq_questions:
        query_text = q_data['body']
        q_id = q_data.get('id')

        # Collect positive (relevant) documents
        positive_snippets = [s['text'] for s in q_data.get('snippets', []) if s.get('text')]
        if not positive_snippets:
            continue # Skip questions with no relevant snippets

        for pos_doc_text in positive_snippets:
            # Sample non-relevant documents
            current_neg_samples = []
            if hard_negatives_map and q_id in hard_negatives_map:
                # Use provided hard negatives first
                available_hard_negs = [neg for neg in hard_negatives_map[q_id] if neg != pos_doc_text]
                take_n = min(len(available_hard_negs), num_neg_samples)
                current_neg_samples.extend(random.sample(available_hard_negs, take_n))
            
            # If more negatives are needed, sample randomly (excluding the current positive doc)
            needed_more_negs = num_neg_samples - len(current_neg_samples)
            if needed_more_negs > 0 and len(all_doc_texts) > 1:
                potential_random_negs = [doc for doc in all_doc_texts if doc != pos_doc_text]
                if potential_random_negs: # Ensure there are docs to sample from
                    take_n_random = min(len(potential_random_negs), needed_more_negs)
                    current_neg_samples.extend(random.sample(potential_random_negs, take_n_random))
            
            if not current_neg_samples and len(all_doc_texts) > 0:
                 # Fallback if no hard negatives and random sampling failed (e.g. only one doc in corpus)
                 # This is a weak negative, but ensures we have a triple.
                 current_neg_samples.append(random.choice([d for d in all_doc_texts if d != pos_doc_text] or all_doc_texts))

            for neg_doc_text in current_neg_samples:
                queries.append(query_text)
                relevant_docs_text.append(pos_doc_text)
                non_relevant_docs_text.append(neg_doc_text)

    return queries, relevant_docs_text, non_relevant_docs_text

def get_dataloaders(vocab, batch_size=config.BATCH_SIZE, num_neg_samples_train=1, num_neg_samples_val=1):
    '''
    Prepares training and validation DataLoaders.
    Assumes BioASQ data is split into train/val or uses the same for both if not specified.
    For a real setup, you'd split your data.
    '''
    # Load raw data
    # For simplicity, using the same data file. Split this for a real application.
    train_questions_raw = load_bioasq_data(config.TRAIN_DATA_PATH)
    # val_questions_raw = load_bioasq_data(config.VAL_DATA_PATH) # If you have a separate val set

    # Load hard negatives if available
    hard_negatives = None
    try:
        with open(config.HARD_NEGATIVES_PATH, 'r') as f:
            hard_negatives = json.load(f) # Assuming format {q_id: [neg_doc_text_1, ...]}
    except FileNotFoundError:
        print(f"Hard negatives file not found at {config.HARD_NEGATIVES_PATH}. Using random negatives only.")

    # Create training triples
    # You might want to split your bioasq_questions into train and validation sets first
    # For this example, we'll use a subset for validation if the data is large enough, or just reuse.
    
    # Simple split for example: 80% train, 20% val
    random.shuffle(train_questions_raw)
    split_idx = int(len(train_questions_raw) * 0.8)
    train_qs_for_triples = train_questions_raw[:split_idx]
    val_qs_for_triples = train_questions_raw[split_idx:]

    if not val_qs_for_triples and train_qs_for_triples: # If data is too small for split
        val_qs_for_triples = train_qs_for_triples 

    print(f"Using {len(train_qs_for_triples)} questions for training triple generation.")
    print(f"Using {len(val_qs_for_triples)} questions for validation triple generation.")

    train_queries, train_rel_docs, train_non_rel_docs = create_training_triples(
        train_qs_for_triples, hard_negatives, num_neg_samples=num_neg_samples_train
    )
    val_queries, val_rel_docs, val_non_rel_docs = create_training_triples(
        val_qs_for_triples, hard_negatives, num_neg_samples=num_neg_samples_val
    )

    if not train_queries:
        raise ValueError("No training triples were generated. Check data paths and content.")
    if not val_queries:
        print("Warning: No validation triples were generated. Validation will be skipped or use training data.")
        # Fallback to use training data for validation if no validation triples generated
        val_queries, val_rel_docs, val_non_rel_docs = train_queries, train_rel_docs, train_non_rel_docs

    train_dataset = BioASQTripletDataset(train_queries, train_rel_docs, train_non_rel_docs, vocab)
    val_dataset = BioASQTripletDataset(val_queries, val_rel_docs, val_non_rel_docs, vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


# --- For Inference --- #
class BioASQPredictionDataset(Dataset):
    '''Dataset for inference: takes (query, candidate_doc) pairs.'''
    def __init__(self, queries_texts, candidate_docs_texts, vocab, 
                 max_query_len=config.MAX_QUERY_LEN, max_doc_len=config.MAX_DOC_LEN):
        self.queries_texts = queries_texts
        self.candidate_docs_texts = candidate_docs_texts
        self.vocab = vocab
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

        if len(queries_texts) != len(candidate_docs_texts):
            raise ValueError("Queries and candidate documents lists must have the same length.")

    def __len__(self):
        return len(self.queries_texts)

    def __getitem__(self, idx):
        query_text = self.queries_texts[idx]
        doc_text = self.candidate_docs_texts[idx]

        query_tokens = tokenize_text(query_text)
        doc_tokens = tokenize_text(doc_text)

        query_seq = texts_to_sequences([query_tokens], self.vocab)[0]
        doc_seq = texts_to_sequences([doc_tokens], self.vocab)[0]

        q_padded = pad_sequence(query_seq, self.max_query_len, pad_value=self.vocab.word2idx[self.vocab.pad_token])
        d_padded = pad_sequence(doc_seq, self.max_doc_len, pad_value=self.vocab.word2idx[self.vocab.pad_token])

        q_mask = create_mask_from_sequence(q_padded, pad_idx=self.vocab.word2idx[self.vocab.pad_token])
        d_mask = create_mask_from_sequence(d_padded, pad_idx=self.vocab.word2idx[self.vocab.pad_token])

        return {
            'query_ids': torch.tensor(q_padded, dtype=torch.long),
            'query_mask': torch.tensor(q_mask, dtype=torch.float32),
            'doc_ids': torch.tensor(d_padded, dtype=torch.long),
            'doc_mask': torch.tensor(d_mask, dtype=torch.float32),
            'original_query': query_text, # Keep original for grouping results
            'original_doc': doc_text
        }

def load_first_stage_rerank_data(filepath):
    """
    Loads data from a first-stage ranker (e.g., BM25 output) for reranking.
    Expected format: A list of questions, each with a 'body' (query) and 'documents' (list of URLs/IDs) 
                     and 'snippets' (list of text snippets corresponding to documents).
                     The reranker will typically rerank these snippets.
    Or, a simpler list of {query_id: str, query_text: str, candidates: [{doc_id: str, doc_text: str, score: float}, ...]}
    This function adapts to the BM25/Dense output structure you provided.
    """
    queries_for_rerank = []
    candidate_docs_for_rerank = []
    query_doc_pairs_info = [] # To store (original_query_id, original_doc_id/text) for result mapping

    with open(filepath, 'r', encoding='utf-8') as f:
        first_stage_data = json.load(f)
    
    questions_data = first_stage_data.get('questions', [])

    for question_item in questions_data:
        query_text = question_item.get('body')
        query_id = question_item.get('id')
        
        # Assuming snippets are the candidates to rerank
        # And that the `documents` list in BM25/Dense output corresponds to these snippets by order/URI.
        # The `snippets` themselves contain the text.
        snippets = question_item.get('snippets', [])
        
        if not query_text or not snippets:
            continue

        for snippet_data in snippets:
            doc_text = snippet_data.get('text')
            # Try to get a unique doc identifier, e.g. from document URL in snippet
            doc_id = snippet_data.get('document', f"snippet_offset_{snippet_data.get('offsetInBeginSection')}_{snippet_data.get('offsetInEndSection')}") 
            if doc_text:
                queries_for_rerank.append(query_text)
                candidate_docs_for_rerank.append(doc_text)
                query_doc_pairs_info.append({
                    'query_id': query_id,
                    'query_text': query_text,
                    'doc_id': doc_id, # This is the URL or a generated ID
                    'doc_text': doc_text
                })
                
    return queries_for_rerank, candidate_docs_for_rerank, query_doc_pairs_info

def get_inference_dataloader(data_filepath, vocab, batch_size=config.BATCH_SIZE):
    queries, docs, pairs_info = load_first_stage_rerank_data(data_filepath)
    if not queries:
        print(f"No query-document pairs loaded from {data_filepath} for inference.")
        return None, None
    
    dataset = BioASQPredictionDataset(queries, docs, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, pairs_info


if __name__ == '__main__':
    # --- Setup --- 
    # 1. Build or Load Vocabulary from training texts
    print("Building vocabulary from training data...")
    raw_train_data = load_bioasq_data(config.TRAIN_DATA_PATH)
    all_training_texts = []
    for q_data in raw_train_data:
        all_training_texts.append(tokenize_text(q_data['body']))
        for snippet in q_data.get('snippets', []):
            all_training_texts.append(tokenize_text(snippet['text']))
    
    vocab_obj = Vocabulary()
    if all_training_texts:
        vocab_obj.build_vocab(all_training_texts, min_freq=config.MIN_WORD_FREQ)
        vocab_obj.save(config.VOCAB_PATH)
        print(f"Vocabulary built and saved to {config.VOCAB_PATH}. Size: {len(vocab_obj)}")
    else:
        print("No training texts found to build vocabulary. Attempting to load existing one or using empty.")
        try:
            vocab_obj = Vocabulary.load(config.VOCAB_PATH)
            print(f"Vocabulary loaded from {config.VOCAB_PATH}. Size: {len(vocab_obj)}")
        except FileNotFoundError:
            print(f"ERROR: {config.VOCAB_PATH} not found and no data to build new vocab. Exiting.")
            exit()

    # --- Training DataLoader Example --- 
    print("\n--- Training DataLoader Example ---")
    train_loader, val_loader = get_dataloaders(vocab_obj, batch_size=2, num_neg_samples_train=1)
    
    if train_loader:
        print(f"Train DataLoader created. Number of batches: {len(train_loader)}")
        for i, batch in enumerate(train_loader):
            print(f"Batch {i+1}:")
            print("Query IDs shape:", batch['query_ids'].shape)
            print("Rel Doc IDs shape:", batch['rel_doc_ids'].shape)
            print("Non-Rel Doc IDs shape:", batch['non_rel_doc_ids'].shape)
            # print("Sample Query:", batch['query_ids'][0, :10])
            if i == 0: # Print one batch
                break
    else:
        print("Failed to create training dataloader.")

    if val_loader:
        print(f"Validation DataLoader created. Number of batches: {len(val_loader)}")

    # --- Inference DataLoader Example --- #
    print("\n--- Inference DataLoader Example (using BM25 output) ---")
    # Ensure BM25_OUTPUT_PATH in config.py points to a valid BM25 output file
    # Create a dummy BM25 output if it doesn't exist for testing
    dummy_bm25_output_path = config.BM25_OUTPUT_PATH
    try:
        with open(dummy_bm25_output_path, 'r') as f:
            pass # Check if file exists
    except FileNotFoundError:
        print(f"Warning: BM25 output {dummy_bm25_output_path} not found. Creating a dummy one for example.")
        dummy_data = {"questions": [
            {"id": "q1_test", "body": "test query one", "snippets": [{"text": "relevant document for q1", "document": "doc1_url"}]},
            {"id": "q2_test", "body": "test query two", "snippets": [{"text": "document for q2", "document": "doc2_url"}]}
        ]}
        with open(dummy_bm25_output_path, 'w') as f:
            json.dump(dummy_data, f)

    inference_loader, inference_pairs_info = get_inference_dataloader(dummy_bm25_output_path, vocab_obj, batch_size=2)
    if inference_loader:
        print(f"Inference DataLoader created. Number of batches: {len(inference_loader)}")
        for i, batch in enumerate(inference_loader):
            print(f"Inference Batch {i+1}:")
            print("Query IDs shape:", batch['query_ids'].shape)
            print("Doc IDs shape:", batch['doc_ids'].shape)
            # print("Original Query:", batch['original_query'][0])
            # print("Original Doc:", batch['original_doc'][0])
            if i == 0: # Print one batch
                break
        # print("\nSample pair info from inference data:", inference_pairs_info[0] if inference_pairs_info else "N/A")
    else:
        print("Failed to create inference dataloader from BM25 output.")