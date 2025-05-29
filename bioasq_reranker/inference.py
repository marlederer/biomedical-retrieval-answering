'''
Inference script for the KNRM reranker.

This script loads a trained KNRM model, processes input data 
(e.g., from BM25 or a dense retriever), reranks the documents, 
and outputs the results in a structured format.
'''
import os
import torch
import json
from tqdm import tqdm
import argparse
from collections import defaultdict

# Added BioASQPredictionDataset and DataLoader
from data_loader import Vocabulary, get_inference_dataloader, BioASQPredictionDataset # Ensure BioASQPredictionDataset is imported
from torch.utils.data import DataLoader # Ensure DataLoader is imported
# Import from BM25 package
from api_client import call_bioasq_api_search

import config
from knrm import KNRM
from metrics import calculate_mrr # For potential evaluation if ground truth is available

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def rerank_documents(
    model_path,
    vocab_path,
    input_retrieval_path, # Path to the output of BM25/Dense (e.g., bioasq_output.json)
    output_reranked_path,
    ground_truth_path=None # Optional: path to BioASQ JSON with ground truth for MRR calc
):
    print(f"Loading vocabulary from {vocab_path}...")
    try:
        vocab = Vocabulary.load(vocab_path)
    except FileNotFoundError:
        print(f"ERROR: Vocabulary file not found at {vocab_path}. Please ensure it's created (e.g., by running train.py or data_loader.py first).")
        return
    print(f"Vocabulary loaded. Size: {len(vocab)}")

    print(f"Loading KNRM model from {model_path}...")
    model = KNRM(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        n_kernels=config.N_KERNELS
    ).to(config.DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please train a model first.")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("This could be due to a mismatch in model architecture or a corrupted file.")
        return
        
    model.eval()
    print("Model loaded.")

    print(f"Loading inference data from {input_retrieval_path}...")
    inference_dataloader, query_doc_pairs_info = get_inference_dataloader(
        input_retrieval_path, vocab, batch_size=config.BATCH_SIZE
    )

    if not inference_dataloader:
        print(f"Could not create inference dataloader. Check {input_retrieval_path}.")
        return

    print("Starting inference and reranking...")
    results = [] # To store (query_id, doc_id, score, original_doc_text)
    all_scores_for_pairs = [] # List of tuples (query_text, doc_text, score)

    with torch.no_grad():
        progress_bar = tqdm(inference_dataloader, desc="Reranking")
        for i, batch in enumerate(progress_bar):
            query_ids = batch['query_ids'].to(config.DEVICE)
            query_mask = batch['query_mask'].to(config.DEVICE)
            doc_ids = batch['doc_ids'].to(config.DEVICE)
            doc_mask = batch['doc_mask'].to(config.DEVICE)
            
            original_queries = batch['original_query']
            original_docs = batch['original_doc']

            scores = model(query_ids, doc_ids, query_mask, doc_mask)
            scores = scores.squeeze(-1).cpu().tolist() # Squeeze and move to CPU

            # The query_doc_pairs_info from get_inference_dataloader is flat.
            # We need to map scores back to the correct (query_id, doc_id) from that flat list.
            # The dataloader processes items sequentially, so we can use the batch index and size.
            start_idx = i * config.BATCH_SIZE
            for j in range(len(scores)):
                current_pair_idx = start_idx + j
                if current_pair_idx < len(query_doc_pairs_info):
                    pair_info = query_doc_pairs_info[current_pair_idx]
                    results.append({
                        'query_id': pair_info['query_id'],
                        'query_text': pair_info['query_text'],
                        'doc_id': pair_info['doc_id'], # This is often a URL or a generated ID
                        'doc_text': pair_info['doc_text'],
                        'rerank_score': scores[j]
                    })
                    all_scores_for_pairs.append((pair_info['query_text'], pair_info['doc_text'], scores[j]))
    
    # Group results by query_id and sort documents by rerank_score
    reranked_output_dict = defaultdict(list)
    for res_item in results:
        reranked_output_dict[res_item['query_id']].append(res_item)

    final_bioasq_structure = {"questions": []}
    for q_id, doc_infos in reranked_output_dict.items():
        # Sort documents for this query by the new rerank_score in descending order
        sorted_doc_infos = sorted(doc_infos, key=lambda x: x['rerank_score'], reverse=True)
        
        # Reconstruct the BioASQ output format
        # We need the original query body. We can get it from the first item (all items for a q_id have same query_text)
        query_body = sorted_doc_infos[0]['query_text'] if sorted_doc_infos else ""
        
        question_output = {
            "id": q_id,
            "body": query_body,
            "documents": [info['doc_id'] for info in sorted_doc_infos], # List of doc URLs/IDs
            "snippets": [
                {
                    "document": info['doc_id'],
                    "text": info['doc_text'],
                    "score_reranked": info['rerank_score'] 
                    # You might want to add original BM25/Dense score here if available in query_doc_pairs_info
                } for info in sorted_doc_infos
            ]
        }
        final_bioasq_structure["questions"].append(question_output)

    # Save the reranked results
    ensure_dir(os.path.dirname(output_reranked_path))
    with open(output_reranked_path, 'w', encoding='utf-8') as f:
        json.dump(final_bioasq_structure, f, indent=4)
    print(f"Reranked results saved to {output_reranked_path}")

    # --- Optional: Calculate MRR if ground truth is provided ---
    if ground_truth_path:
        print(f"Calculating MRR@{config.MRR_K} using ground truth from {ground_truth_path}...")
        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f).get('questions', [])
            
            relevant_docs_map_gt = {}
            for q_data in gt_data:
                # Assuming ground truth relevant documents are identified by their URLs/IDs in 'documents' field
                # and these IDs match what's in our 'doc_id' from the reranked results.
                relevant_docs_map_gt[q_data['id']] = set(s['document'] for s in q_data.get('snippets',[]))
                # Or, if your ground truth has a simpler list of doc IDs:
                # relevant_docs_map_gt[q_data['id']] = set(q_data.get('documents', []))

            # Prepare ranked_lists for MRR calculation from our reranked output
            ranked_lists_for_mrr = {}
            for q_output in final_bioasq_structure['questions']:
                ranked_lists_for_mrr[q_output['id']] = [doc_id for doc_id in q_output['documents']]

            if not ranked_lists_for_mrr:
                print("No reranked lists available to calculate MRR.")
            elif not relevant_docs_map_gt:
                print("No ground truth relevant documents loaded. Cannot calculate MRR.")
            else:
                mrr_score, _ = calculate_mrr(ranked_lists_for_mrr, relevant_docs_map_gt, k=config.MRR_K)
                print(f"MRR@{config.MRR_K}: {mrr_score:.4f}")

        except FileNotFoundError:
            print(f"Ground truth file not found at {ground_truth_path}. Skipping MRR calculation.")
        except Exception as e:
            print(f"Error during MRR calculation: {e}")

def rerank_for_single_query(
    query_string: str,
    model, # KNRM model instance
    vocab, # Vocabulary instance
    bioasq_api_url: str,
    num_initial_candidates: int,
    top_k_output: int
):
    print(f'Fetching initial candidates for query: "{query_string}" from BioASQ API...')
    api_articles = call_bioasq_api_search(
        query_keywords=query_string,
        num_articles_to_fetch=num_initial_candidates,
        api_endpoint_url=bioasq_api_url
    )

    if not api_articles:
        print("No documents retrieved from BioASQ API for the query.")
        return []

    print(f"Retrieved {len(api_articles)} initial candidates.")

    queries_for_rerank = []
    candidate_docs_for_rerank = []
    candidate_details_for_output = []
    # Use a simple query_id for this single run, or generate one if needed for specific formats
    query_id_for_this_run = f"single_query_{hash(query_string)}_run"

    for article in api_articles:
        abstract = article.get('abstract', '')
        # Ensure abstract is not empty or just whitespace, as it's used for reranking
        if not abstract or not abstract.strip(): 
            # print(f"Skipping article {article.get('id')} due to empty abstract.") # Optional: for debugging
            continue
        queries_for_rerank.append(query_string)
        candidate_docs_for_rerank.append(abstract)
        candidate_details_for_output.append({
            'query_id': query_id_for_this_run, # Use the consistent query_id
            'query_text': query_string,
            'doc_id': article.get('id') or article.get('url', 'N/A'), # Get PMID or URL
            'title': article.get('title', 'N/A'),
            'doc_text': abstract # Store the abstract that will be reranked
        })
    
    if not candidate_docs_for_rerank: # Check if any valid docs are left after filtering
        print("No valid candidate documents with abstracts to rerank after filtering.")
        return []

    print(f"Processing {len(candidate_docs_for_rerank)} candidates with abstracts for reranking...")

    # Create Dataset and DataLoader for the retrieved candidates
    prediction_dataset = BioASQPredictionDataset(
        queries_for_rerank, 
        candidate_docs_for_rerank, 
        vocab,
        max_query_len=config.MAX_QUERY_LEN, # from config
        max_doc_len=config.MAX_DOC_LEN    # from config
    )
    prediction_dataloader = DataLoader(
        prediction_dataset, 
        batch_size=config.BATCH_SIZE, # from config
        shuffle=False # No need to shuffle for inference
    )

    model.eval() # Ensure model is in evaluation mode
    all_reranked_results = []

    with torch.no_grad(): # Disable gradient calculations for inference
        progress_bar = tqdm(prediction_dataloader, desc=f"Reranking for query: {query_string[:30]}...")
        batch_start_idx = 0
        for batch in progress_bar:
            query_ids_b = batch['query_ids'].to(config.DEVICE)
            query_mask_b = batch['query_mask'].to(config.DEVICE)
            doc_ids_b = batch['doc_ids'].to(config.DEVICE)
            doc_mask_b = batch['doc_mask'].to(config.DEVICE)

            scores_b = model(query_ids_b, doc_ids_b, query_mask_b, doc_mask_b)
            scores_b = scores_b.squeeze(-1).cpu().tolist() # Get scores as a list

            # Map scores back to original candidate details
            for i, score in enumerate(scores_b):
                detail_idx = batch_start_idx + i
                if detail_idx < len(candidate_details_for_output):
                    original_candidate_info = candidate_details_for_output[detail_idx]
                    all_reranked_results.append({
                        **original_candidate_info, # Spread the original info (query_id, doc_id, text, etc.)
                        'rerank_score': score
                    })
            batch_start_idx += len(scores_b) # Correctly increment for next batch
            
    # Sort all collected results by rerank_score in descending order
    sorted_results = sorted(all_reranked_results, key=lambda x: x['rerank_score'], reverse=True)
    
    # Return only the top_k_output documents
    return sorted_results[:top_k_output]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rerank documents using a trained KNRM model.")
    parser.add_argument("--model_path", type=str, default=config.SAVE_MODEL_PATH,
                        help="Path to the trained KNRM model file (.pth)")
    parser.add_argument("--vocab_path", type=str, default=config.VOCAB_PATH,
                        help="Path to the vocabulary file (.json)")
    
    # Group for mutually exclusive --input_file or --query
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_file", type=str,
                        help="Path to the input file from a first-stage retriever (e.g., BM25 output JSON) for batch reranking.")
    group.add_argument("--query", type=str,
                        help="A single query string to process for end-to-end reranking via BioASQ API.")

    parser.add_argument("--output_file", type=str, 
                        help="Path to save the output JSON. If in --query mode and not provided, results are printed to console. If in --input_file mode and not provided, a default name is used.")
    parser.add_argument("--ground_truth_file", type=str, default=config.TRAIN_DATA_PATH,
                        help="Optional: Path to the ground truth BioASQ JSON file for MRR calculation (only used with --input_file mode).")
    
    args = parser.parse_args()

    # Ensure essential directories exist
    ensure_dir(os.path.dirname(config.VOCAB_PATH)) # For vocab if it needs to be created by dummy logic
    ensure_dir(os.path.dirname(config.SAVE_MODEL_PATH)) # For model if it needs to be created by dummy logic
    if args.output_file:
        ensure_dir(os.path.dirname(args.output_file))
    else: # Ensure default output directory for batch mode exists if no output_file is specified
        if args.input_file:
            ensure_dir("reranked_output")

    # --- Load Vocabulary and Model (common for both modes) ---
    if not os.path.exists(args.vocab_path):
        print(f"Warning: Vocabulary file {args.vocab_path} not found. Creating a dummy vocab for script execution.")
        dummy_vocab_data = {"word2idx": {"<pad>": 0, "<unk>": 1, "test": 2, "document":3 }, "idx2word": {"0": "<pad>", "1": "<unk>", "2": "test", "3": "document"}}
        with open(args.vocab_path, 'w') as f: # This will create vocab in data/ if config.VOCAB_PATH is data/vocab.json
            json.dump(dummy_vocab_data, f)
    
    print(f"Loading vocabulary from {args.vocab_path}...")
    try:
        vocab = Vocabulary.load(args.vocab_path)
    except FileNotFoundError:
        print(f"ERROR: Vocabulary file not found at {args.vocab_path} even after dummy check. Exiting.")
        exit(1)
    except Exception as e:
        print(f"Error loading vocabulary: {e}. Exiting.")
        exit(1)
    print(f"Vocabulary loaded. Size: {len(vocab)}")

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path {args.model_path} does not exist. Please train a model first. Exiting.")
        exit(1)

    print(f"Loading KNRM model from {args.model_path}...")
    model = KNRM(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        n_kernels=config.N_KERNELS
    ).to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
    except Exception as e:
        print(f"Error loading model state_dict: {e}. Exiting.")
        exit(1)
    model.eval() # Set model to evaluation mode
    print("Model loaded.")

    # --- Mode selection based on arguments ---
    if args.query:
        # Single query mode execution
        print(f"Executing in single query mode for: \"{args.query}\"")
        top_reranked_docs = rerank_for_single_query(
            query_string=args.query,
            model=model,
            vocab=vocab,
            bioasq_api_url=config.BIOASQ_API_ENDPOINT_URL,
            num_initial_candidates=config.API_NUM_INITIAL_CANDIDATES,
            top_k_output=config.RERANK_TOP_K_OUTPUT
        )

        if top_reranked_docs:
            print(f"\nTop {len(top_reranked_docs)} reranked documents for query: \"{args.query}\"")
            # Prepare for printing or saving
            output_data_single_query = {
                "query": args.query,
                "reranked_documents": top_reranked_docs
            }
            formatted_output_json = json.dumps(output_data_single_query, indent=4)
            print(formatted_output_json)
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_output_json)
                print(f"Single query results saved to {args.output_file}")
        else:
            print(f"No results to display or save for query: \"{args.query}\"")

    elif args.input_file:
        # Batch reranking from file mode (existing functionality)
        print(f"Executing in batch mode with input file: {args.input_file}")
        input_file_to_use = args.input_file
        if not os.path.exists(input_file_to_use):
            # This dummy creation logic might be too aggressive if user just mistyped.
            # For now, it ensures script can run for testing if file is missing.
            print(f"Warning: Specified input file {input_file_to_use} not found.")
            print(f"Creating dummy input retrieval data at 'dummy_retrieval_input_batch.json' for script execution.")
            dummy_retrieval_data = {"questions": [
                {"id": "q1_infer_batch_dummy", "body": "dummy query for batch", 
                 "snippets": [{"text": "dummy doc 1 batch", "document": "dummy_doc_url_1"}, {"text": "dummy doc 2 batch", "document": "dummy_doc_url_2"}]}
            ]}
            input_file_to_use = "dummy_retrieval_input_batch.json" 
            with open(input_file_to_use, 'w') as f:
                json.dump(dummy_retrieval_data, f)
        
        # Determine output file path for batch mode
        output_file_for_batch = args.output_file
        if not output_file_for_batch: # If no output file specified for batch mode, use a default
            output_file_for_batch = os.path.join("reranked_output", "knrm_reranked_bioasq_batch_default.json")
        ensure_dir(os.path.dirname(output_file_for_batch)) # Ensure directory for output file exists

        gt_file_to_use = args.ground_truth_file
        # Only attempt to use ground truth if the file actually exists
        if gt_file_to_use and not os.path.exists(gt_file_to_use):
            print(f"Warning: Ground truth file {gt_file_to_use} not found. MRR calculation will be skipped.")
            gt_file_to_use = None # Set to None so rerank_documents skips MRR

        rerank_documents(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            input_retrieval_path=input_file_to_use,
            output_reranked_path=output_file_for_batch,
            ground_truth_path=gt_file_to_use 
        )
