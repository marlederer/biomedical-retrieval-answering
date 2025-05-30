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
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed 

import config
from knrm import KNRM
from data_loader import fetch_document_content 
from utils import Vocabulary, texts_to_sequences, pad_sequence, create_mask_from_sequence, tokenize_text 
from metrics import calculate_mrr

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to fetch document content (moved here for simplicity for this script)
def fetch_document_content(url: str):
    """
    Fetches the title and abstract of a PubMed article from its URL.
    Returns a concatenated string "title\\nabstract" or None if fetching fails.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        soup = BeautifulSoup(response.content, 'html.parser')

        title = ""
        title_tag = soup.find('h1', class_='heading-title')
        if title_tag:
            title = title_tag.get_text(separator=' ', strip=True)
        
        abstract_text = ""
        abstract_div = soup.find('div', class_='abstract-content')
        if abstract_div:
            # Try to get structured abstract if present
            abstract_sections = abstract_div.find_all(['strong', 'p'], recursive=False)
            if any(sec.name == 'strong' for sec in abstract_sections): # Likely structured abstract
                current_section_text = []
                for sec in abstract_sections:
                    current_section_text.append(sec.get_text(separator=' ', strip=True))
                abstract_text = "\\n".join(current_section_text)
            else: # Unstructured or simple <p> tags
                paragraphs = abstract_div.find_all('p')
                abstract_text = "\\n".join([p.get_text(separator=' ', strip=True) for p in paragraphs])
        
        if not title and not abstract_text: # Fallback if specific tags are not found
            title_tag_meta = soup.find('meta', attrs={'name': 'citation_title'})
            if title_tag_meta and title_tag_meta.get('content'):
                title = title_tag_meta.get('content')
            
            abstract_tag_meta = soup.find('meta', attrs={'name': 'citation_abstract'})
            if abstract_tag_meta and abstract_tag_meta.get('content'):
                abstract_text = abstract_tag_meta.get('content')

        if title or abstract_text:
            return f"{title}\\n{abstract_text}".strip()
        else:
            # print(f"Could not extract title/abstract from {url}")
            return None

    except requests.exceptions.Timeout:
        print(f"Timeout while fetching {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing content from {url}: {e}")
        return None

def rerank_documents(
    model_path,
    vocab_path,
    input_retrieval_path, # Path to the output of BM25/Dense (e.g., bioasq_output.json)
    output_reranked_path, # Path to save reranked output (currently commented out)
    ground_truth_path="bioasq_reranker\data\training13b.json" # Optional path to ground truth BioASQ JSON for MRR calculation
):
    print(f"Loading vocabulary from {vocab_path}...")
    try:
        vocab = Vocabulary.load(vocab_path)
    except FileNotFoundError:
        print(f"ERROR: Vocabulary file not found at {vocab_path}. Please ensure it's created.")
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
        return
    model.eval()
    print("Model loaded.")

    print(f"Loading inference data from {input_retrieval_path}...")
    try:
        with open(input_retrieval_path, 'r', encoding='utf-8') as f:
            all_questions_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input retrieval file not found at {input_retrieval_path}.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {input_retrieval_path}.")
        return

    questions_to_process = all_questions_data.get('questions', [])
    if not questions_to_process:
        print("No questions found in the input file.")
        return

    # Load ground truth data 
    relevant_docs_map_gt = {}
    if ground_truth_path:
        print(f"Loading ground truth from {ground_truth_path}...")
        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                gt_data_full = json.load(f)
            for q_gt in gt_data_full.get('questions', []):
                # BioASQ GT usually has relevant doc URLs in the 'documents' field
                relevant_docs_map_gt[q_gt['id']] = set(q_gt.get('documents', []))
            if not relevant_docs_map_gt:
                print("Warning: Ground truth file loaded, but no relevant documents mapped. Check GT format.")
        except FileNotFoundError:
            print(f"Warning: Ground truth file {ground_truth_path} not found. MRR cannot be calculated.")
            ground_truth_path = None # Disable MRR calculation
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from ground truth file {ground_truth_path}. MRR cannot be calculated.")
            ground_truth_path = None # Disable MRR calculation
        except Exception as e:
            print(f"Warning: Error loading ground truth data: {e}. MRR cannot be calculated.")
            ground_truth_path = None

    total_mrr_sum = 0.0
    questions_with_gt_and_docs_count = 0

    print("Starting per-question inference, reranking, and MRR calculation...")
    for question_data in tqdm(questions_to_process, desc="Processing Questions"):
        query_id = question_data.get('id')
        query_text = question_data.get('body')

        if not query_id or not query_text:
            print("Skipping question due to missing ID or body.")
            continue

        # Get document URLs directly from the 'documents' field of the question
        document_urls = question_data.get('documents', [])

        if not document_urls:
            print(f"No documents found in 'documents' list for question {query_id}. Skipping MRR calculation for this question.")
            if ground_truth_path and query_id in relevant_docs_map_gt:
                 print(f"MRR@{config.MRR_K} for question {query_id}: 0.0000 (no documents to rank)")
            continue

        current_question_docs_for_model = []
        # Parallel fetching of document content
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_url = {executor.submit(fetch_document_content, doc_url): doc_url for doc_url in document_urls}
            # Inner tqdm for document fetching progress for the current question
            for future in tqdm(as_completed(future_to_url), total=len(document_urls), desc=f"Fetching docs for Q {query_id[:10]}...", leave=False):
                doc_url = future_to_url[future]
                try:
                    content = future.result()
                    if content: 
                        current_question_docs_for_model.append({'url': doc_url, 'text': content, 'query_id': query_id})
                    else:
                        # Log that content couldn't be fetched and this doc is skipped
                        print(f"Warning: No content fetched for document {doc_url} for question {query_id}. Document will not be included in reranking.")
                except Exception as exc:
                    print(f"Warning: Document {doc_url} generated an exception during fetching: {exc}. Document will not be included in reranking.")


        if not current_question_docs_for_model:
            print(f"No document content could be prepared for question {query_id}. Skipping MRR calculation for this question.")
            if ground_truth_path and query_id in relevant_docs_map_gt:
                print(f"MRR@{config.MRR_K} for question {query_id}: 0.0000 (no document content)")
            continue
            
        # Tokenize query
        query_tokens = tokenize_text(query_text, config.TOKENIZER_TYPE) 
        # q_ids_tensor, q_mask_tensor = vocab.prepare_tensor(query_tokens, config.MAX_QUERY_LEN, config.DEVICE) # Old problematic line
        query_seq = texts_to_sequences([query_tokens], vocab)[0]
        padded_query_seq = pad_sequence(query_seq, config.MAX_QUERY_LEN, pad_value=vocab.word2idx[vocab.pad_token])
        query_mask = create_mask_from_sequence(padded_query_seq, pad_idx=vocab.word2idx[vocab.pad_token])
        
        q_ids_tensor = torch.tensor(padded_query_seq, dtype=torch.long).to(config.DEVICE)
        q_mask_tensor = torch.tensor(query_mask, dtype=torch.float32).to(config.DEVICE) # KNRM expects float mask
        
        num_docs_for_query = len(current_question_docs_for_model)
        batch_query_ids = q_ids_tensor.unsqueeze(0).repeat(num_docs_for_query, 1)
        batch_query_mask = q_mask_tensor.unsqueeze(0).repeat(num_docs_for_query, 1)

        # Tokenize documents for the current query
        doc_ids_list = []
        doc_mask_list = []
        for doc_content in current_question_docs_for_model:
            doc_tokens = tokenize_text(doc_content['text'], config.TOKENIZER_TYPE)
            # d_ids_tensor, d_mask_tensor = vocab.prepare_tensor(doc_tokens, config.MAX_DOC_LEN, config.DEVICE) # Old problematic line
            doc_seq = texts_to_sequences([doc_tokens], vocab)[0]
            padded_doc_seq = pad_sequence(doc_seq, config.MAX_DOC_LEN, pad_value=vocab.word2idx[vocab.pad_token])
            doc_mask = create_mask_from_sequence(padded_doc_seq, pad_idx=vocab.word2idx[vocab.pad_token])

            d_ids_tensor = torch.tensor(padded_doc_seq, dtype=torch.long).to(config.DEVICE)
            d_mask_tensor = torch.tensor(doc_mask, dtype=torch.float32).to(config.DEVICE) # KNRM expects float mask
            
            doc_ids_list.append(d_ids_tensor)
            doc_mask_list.append(d_mask_tensor)
        
        batch_doc_ids = torch.stack(doc_ids_list)
        batch_doc_mask = torch.stack(doc_mask_list)

        # Get scores from model
        with torch.no_grad():
            scores_tensor = model(batch_query_ids, batch_doc_ids, batch_query_mask, batch_doc_mask)
        
        scores_list = scores_tensor.squeeze(-1).cpu().tolist()

        # Combine scores with document info for sorting
        scored_docs_for_this_question = []
        for i, doc_data in enumerate(current_question_docs_for_model):
            scored_docs_for_this_question.append({
                'doc_id': doc_data['url'], 
                'doc_text': doc_data['text'], 
                'rerank_score': scores_list[i]
            })
        
        # Sort documents by new rerank_score
        sorted_docs = sorted(scored_docs_for_this_question, key=lambda x: x['rerank_score'], reverse=True)

        # Calculate and print MRR for this question if ground truth is available
        if ground_truth_path and query_id in relevant_docs_map_gt:
            ranked_list_ids = [d['doc_id'] for d in sorted_docs]
            gt_relevant_set_for_query = relevant_docs_map_gt[query_id]

            if not gt_relevant_set_for_query: # No relevant docs in GT for this query
                print(f"MRR@{config.MRR_K} for question {query_id}: N/A (no relevant documents in ground truth)")
            elif not ranked_list_ids: # No documents were ranked (e.g. all failed fetching)
                 print(f"MRR@{config.MRR_K} for question {query_id}: 0.0000 (no documents were ranked)")
            else:
                # calculate_mrr expects dicts, so wrap the single query's data
                _, per_query_mrr_dict = calculate_mrr(
                    {query_id: ranked_list_ids},
                    {query_id: gt_relevant_set_for_query},
                    k=config.MRR_K
                )
                
                q_mrr_score = per_query_mrr_dict.get(query_id, 0.0) # Default to 0 if not calculable
                print(f"MRR@{config.MRR_K} for question {query_id}: {q_mrr_score:.4f}")
                total_mrr_sum += q_mrr_score
                questions_with_gt_and_docs_count += 1
        elif ground_truth_path: # GT was provided, but not for this specific query_id
            print(f"No ground truth found for question {query_id}. Skipping MRR calculation for this question.")
        # If no ground_truth_path, MRR calculation is skipped silently for each question.

    # After processing all questions, print average MRR
    if ground_truth_path: # Only print average if GT was attempted
        if questions_with_gt_and_docs_count > 0:
            average_mrr_overall = total_mrr_sum / questions_with_gt_and_docs_count
            print(f"\\nAverage MRR@{config.MRR_K} over {questions_with_gt_and_docs_count} questions: {average_mrr_overall:.4f}")
        else:
            print(f"\\nNo MRR scores were calculated for any question (or no questions had ground truth and processable documents). Average MRR@{config.MRR_K} is not applicable.")
    else:
        print("\\nGround truth path not provided. MRR calculation was skipped.")


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
        if os.path.exists(config.BM25_OUTPUT_PATH):
            print(f"Using BM25 output from config: {config.BM25_OUTPUT_PATH}")
            input_file_to_use = config.BM25_OUTPUT_PATH
        elif os.path.exists(config.DENSE_OUTPUT_PATH):
            print(f"Using Dense retriever output from config: {config.DENSE_OUTPUT_PATH}")
            input_file_to_use = config.DENSE_OUTPUT_PATH
        else:
            print(f"Warning: Dummy input retrieval data created at 'dummy_retrieval_input.json' for inference script execution.")
            # Ensure dummy data has 'documents' field for URLs
            dummy_retrieval_data = {"questions": [
                {"id": "q1_infer", "body": "test query for inference", 
                 "documents": ["http://example.com/doc1", "http://example.com/doc2"], # Added documents field
                 "snippets": [{"text": "sample document one", "document": "http://example.com/doc1"}, 
                              {"text": "sample document two", "document": "http://example.com/doc2"}]}
            ]}
            input_file_to_use = "dummy_retrieval_input.json"
            with open(input_file_to_use, 'w') as f:
                json.dump(dummy_retrieval_data, f)
    
    gt_file_to_use = args.ground_truth_file
    if gt_file_to_use and not os.path.exists(gt_file_to_use):
        if gt_file_to_use == config.TRAIN_DATA_PATH and os.path.exists(config.TRAIN_DATA_PATH):
            pass 
        else:
            print(f"Warning: Dummy ground truth data created at {gt_file_to_use} for MRR calculation structure test.")
            dummy_gt_data = {"questions": [
                {"id": "q1_infer", "body": "test query for inference", "documents": ["doc_url_1"], 
                 "snippets": [{"text": "sample document one", "document": "doc_url_1"}] 
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
