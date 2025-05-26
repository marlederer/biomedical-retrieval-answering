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

import config
from knrm import KNRM
from data_loader import Vocabulary, get_inference_dataloader
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rerank documents using a trained KNRM model.")
    parser.add_argument("--model_path", type=str, default=config.SAVE_MODEL_PATH,
                        help="Path to the trained KNRM model file (.pth)")
    parser.add_argument("--vocab_path", type=str, default=config.VOCAB_PATH,
                        help="Path to the vocabulary file (.json)")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input file from a first-stage retriever (e.g., BM25 output JSON)")
    parser.add_argument("--output_file", type=str, default="reranked_output/knrm_reranked_bioasq.json",
                        help="Path to save the reranked output JSON")
    parser.add_argument("--ground_truth_file", type=str, default=config.TRAIN_DATA_PATH, # Example: use train data for GT structure
                        help="Optional: Path to the ground truth BioASQ JSON file for MRR calculation.")
    
    args = parser.parse_args()

    # Create dummy files for a quick test if they don't exist
    ensure_dir("data")
    ensure_dir("models")
    ensure_dir(os.path.dirname(args.output_file))

    if not os.path.exists(args.vocab_path):
        print(f"Warning: Dummy vocabulary created at {args.vocab_path} for inference script execution.")
        dummy_vocab_data = {"word2idx": {"<pad>": 0, "<unk>": 1, "test": 2, "document":3 }, "idx2word": {"0": "<pad>", "1": "<unk>", "2": "test", "3": "document"}}
        with open(args.vocab_path, 'w') as f:
            json.dump(dummy_vocab_data, f)

    if not os.path.exists(args.model_path):
        print(f"Warning: Model path {args.model_path} does not exist. Inference will likely fail unless a model is placed there.")
        # Cannot create a dummy model easily, user must train one.

    # Check if input file exists, if not, try to use a default from config or make a dummy one
    input_file_to_use = args.input_file
    if not os.path.exists(input_file_to_use):
        print(f"Warning: Specified input file {input_file_to_use} not found.")
        # Try to use one of the default config paths if they exist
        if os.path.exists(config.BM25_OUTPUT_PATH):
            print(f"Using BM25 output from config: {config.BM25_OUTPUT_PATH}")
            input_file_to_use = config.BM25_OUTPUT_PATH
        elif os.path.exists(config.DENSE_OUTPUT_PATH):
            print(f"Using Dense retriever output from config: {config.DENSE_OUTPUT_PATH}")
            input_file_to_use = config.DENSE_OUTPUT_PATH
        else:
            print(f"Warning: Dummy input retrieval data created at 'dummy_retrieval_input.json' for inference script execution.")
            dummy_retrieval_data = {"questions": [
                {"id": "q1_infer", "body": "test query for inference", 
                 "snippets": [{"text": "sample document one", "document": "doc_url_1"}, {"text": "sample document two", "document": "doc_url_2"}]}
            ]}
            input_file_to_use = "dummy_retrieval_input.json"
            with open(input_file_to_use, 'w') as f:
                json.dump(dummy_retrieval_data, f)
    
    # Ensure ground truth file exists if specified, or create a dummy one for testing structure
    gt_file_to_use = args.ground_truth_file
    if gt_file_to_use and not os.path.exists(gt_file_to_use):
        if gt_file_to_use == config.TRAIN_DATA_PATH and os.path.exists(config.TRAIN_DATA_PATH):
            pass # It will use the one from config
        else:
            print(f"Warning: Dummy ground truth data created at {gt_file_to_use} for MRR calculation structure test.")
            dummy_gt_data = {"questions": [
                {"id": "q1_infer", "body": "test query for inference", "documents": ["doc_url_1"], 
                 "snippets": [{"text": "sample document one", "document": "doc_url_1"}] # Assuming doc_url_1 is relevant
                }
            ]}
            with open(gt_file_to_use, 'w') as f:
                json.dump(dummy_gt_data, f)

    rerank_documents(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        input_retrieval_path=input_file_to_use,
        output_reranked_path=args.output_file,
        ground_truth_path=gt_file_to_use
    )
