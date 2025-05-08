# inference.py (with BioASQ API for candidate docs & document re-ranking)
import os
import json
import logging
from collections import defaultdict
import time  # For polite API usage

import torch
from transformers import AutoTokenizer
from api_client import call_bioasq_api_search  # Updated import

from model_arch import CrossEncoderReRanker  # Assuming model_arch.py contains your class

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Hardcoded Inference Configuration ---
INFERENCE_CONFIG = {
    "model_path": "bioasq_reranker/saved_models/my_biomedbert_reranker_hardcoded",  # Path to your trained re-ranker model
    "query": "What are the effective treatments for metastatic melanoma?",

    # --- BioASQ API Parameters for Candidate Document Retrieval ---
    "bioasq_api_endpoint": "http://bioasq.org:8000/pubmed",  
    "pubmed_fetch_count": 10,  # Number of documents to fetch from PubMed as candidates

    # --- Passage Generation Parameters ---
    "passage_field_from_pubmed": "abstract",  # Use 'abstract' or 'title_abstract'
                                             # 'full_text' is usually not available directly via simple Entrez fetch

    # --- Re-ranker Model Parameters ---
    "max_seq_length_passage": 256,
    "batch_size_passage": 16,

    # --- Document Score Aggregation ---
    "aggregation_method": "max",  # 'max', 'avg_top_k_passages'
    # "k_for_avg": 3
}
# --- End of Hardcoded Configuration ---

def fetch_pubmed_articles(query_term, count=10, api_endpoint_url=None):
    """
    Fetches article details (PMID, Title, Abstract) from PubMed for a given query
    using the call_bioasq_api_search function.
    """
    logger.info(f"Fetching {count} articles from BioASQ service for query: '{query_term}' using endpoint: {api_endpoint_url}")
    
    if not api_endpoint_url:
        logger.error("BioASQ API endpoint URL not provided.")
        return []

    try:
        # Directly call the API function
        # The call_bioasq_api_search function is expected to return a list of dictionaries:
        # [{"id": pmid, "url": url, "title": title, "abstract": abstract_text}, ...]
        articles_data_from_api = call_bioasq_api_search(
            query_keywords=query_term,
            num_articles_to_fetch=count,
            api_endpoint_url=api_endpoint_url
        )

        if not articles_data_from_api:
            logger.warning(f"No articles found via BioASQ service for query: '{query_term}'")
            return []

        # Transform the data to the expected format for the rest of the script
        candidate_docs_from_pubmed = []
        for article in articles_data_from_api:
            candidate_docs_from_pubmed.append({
                "doc_id": article.get("id"),  # Map 'id' to 'doc_id'
                "title": article.get("title", ""),
                "abstract": article.get("abstract", ""),
                "url": article.get("url", "")  # Keep URL if needed later
            })

    except Exception as e:
        logger.error(f"Error fetching from BioASQ service: {e}")
        return []  # Return empty list on error
    
    logger.info(f"Fetched {len(candidate_docs_from_pubmed)} candidate documents from BioASQ service.")
    return candidate_docs_from_pubmed


def split_document_into_passages(doc_text, max_passage_tokens=200, stride=100, tokenizer_for_len_check=None):
    passages = []
    if not doc_text: return passages
    if tokenizer_for_len_check:
        tokens = tokenizer_for_len_check.tokenize(doc_text)
        current_pos = 0
        while current_pos < len(tokens):
            end_pos = min(current_pos + max_passage_tokens, len(tokens))
            passage_tokens = tokens[current_pos:end_pos]
            passages.append(tokenizer_for_len_check.convert_tokens_to_string(passage_tokens))
            if end_pos == len(tokens): break
            current_pos += (max_passage_tokens - stride)
            if current_pos >= end_pos: current_pos = end_pos
    else:
        words = doc_text.split()
        current_pos = 0
        while current_pos < len(words):
            end_pos = min(current_pos + max_passage_tokens, len(words))
            passage_words = words[current_pos:end_pos]
            passages.append(" ".join(passage_words))
            if end_pos == len(words): break
            current_pos += (max_passage_tokens - stride)
            if current_pos >= end_pos: current_pos = end_pos
    return [p for p in passages if p.strip()]


def rerank_individual_passages(query, passages, model, tokenizer, device, max_seq_length, batch_size):
    if not passages: return []
    model.eval()
    all_passage_scores = []
    for i in range(0, len(passages), batch_size):
        batch_passage_texts = passages[i:i+batch_size]
        inputs = tokenizer(
            [query] * len(batch_passage_texts), batch_passage_texts, padding=True,
            truncation='only_second', max_length=max_seq_length, return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            scores_tensor = logits.squeeze(-1)
            scores_numpy = scores_tensor.cpu().numpy()
            for passage_text, score in zip(batch_passage_texts, scores_numpy):
                all_passage_scores.append({"text": passage_text, "score": float(score)})
    return all_passage_scores


def run_document_inference_with_pubmed(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_path_config = config["model_path"]
    max_seq_length_passage_config = config["max_seq_length_passage"]
    bioasq_api_endpoint_config = config.get("bioasq_api_endpoint")  # Get BioASQ API endpoint

    # --- Load Re-ranker Model and Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_config)
        original_model_name = None
        training_args_data = None
        training_args_path_v1 = os.path.join(model_path_config, "training_args.json")
        training_args_path_v2 = os.path.join(model_path_config, "training_config_hardcoded.json")
        if os.path.exists(training_args_path_v1):
            with open(training_args_path_v1, 'r') as f: training_args_data = json.load(f)
        elif os.path.exists(training_args_path_v2):
            with open(training_args_path_v2, 'r') as f: training_args_data = json.load(f)
        if training_args_data: original_model_name = training_args_data.get("model_name")
        if not original_model_name:
            logger.error("Cannot determine base BERT model name from training config.")
            return

        model_state_dict_path = os.path.join(model_path_config, "pytorch_model.bin")
        passage_reranker_model = CrossEncoderReRanker(model_name_or_path=original_model_name)
        passage_reranker_model.load_state_dict(torch.load(model_state_dict_path, map_location=device))
        passage_reranker_model.to(device)
        logger.info(f"Passage re-ranker model loaded successfully from {model_path_config}.")
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        return

    # --- Fetch Candidate Documents from PubMed (via BioASQ service) ---
    query_text = config["query"]
    candidate_documents = fetch_pubmed_articles(
        query_text,
        count=config["pubmed_fetch_count"],
        api_endpoint_url=bioasq_api_endpoint_config  # Pass BioASQ API endpoint
    )
    if not candidate_documents:
        logger.warning(f"No candidate documents fetched from BioASQ service for query '{query_text}'. Cannot proceed.")
        print(f"\nQuery: {query_text}")
        print("No documents found via BioASQ service for this query.")
        return

    # --- Process Candidate Documents (Re-ranking Logic) ---
    passage_field = config["passage_field_from_pubmed"]  # Field to get text from PubMed results
    document_scores = defaultdict(lambda: {"doc_id": "", "passages_scores": [], "aggregated_score": -float('inf'), "title": "", "original_doc_data": None})

    logger.info(f"Re-ranking {len(candidate_documents)} documents fetched from BioASQ service...")

    for doc_data in candidate_documents:
        doc_id = doc_data["doc_id"]
        doc_title = doc_data.get("title", "N/A")
        document_scores[doc_id]["doc_id"] = doc_id
        document_scores[doc_id]["title"] = doc_title
        document_scores[doc_id]["original_doc_data"] = doc_data

        text_to_process = ""
        if passage_field == "abstract":
            text_to_process = doc_data.get("abstract", "")
        elif passage_field == "title_abstract":
            title_text = doc_data.get("title", "")
            abstract_text = doc_data.get("abstract", "")
            text_to_process = f"{title_text}. {abstract_text}".strip()
        else:
            logger.warning(f"Unsupported passage_field_from_pubmed: {passage_field}. Defaulting to abstract.")
            text_to_process = doc_data.get("abstract", "")

        if not text_to_process.strip():
            logger.warning(f"Document PMID {doc_id} ('{doc_title}') has no text for field '{passage_field}'. Assigning very low score.")
            document_scores[doc_id]["aggregated_score"] = -float('inf')
            document_scores[doc_id]["passages_scores"] = []
            continue

        doc_passages = [text_to_process]

        if not doc_passages:
            logger.warning(f"No passages generated for document {doc_id}. Skipping.")
            document_scores[doc_id]["aggregated_score"] = -float('inf')
            document_scores[doc_id]["passages_scores"] = []
            continue

        ranked_passages_for_doc = rerank_individual_passages(
            query_text,
            doc_passages,
            passage_reranker_model,
            tokenizer,
            device,
            max_seq_length=max_seq_length_passage_config,
            batch_size=config["batch_size_passage"]
        )
        document_scores[doc_id]["passages_scores"] = ranked_passages_for_doc

        if ranked_passages_for_doc:
            if config["aggregation_method"] == "max":
                document_scores[doc_id]["aggregated_score"] = max(p["score"] for p in ranked_passages_for_doc)
            else:
                document_scores[doc_id]["aggregated_score"] = max(p["score"] for p in ranked_passages_for_doc)
        else:
            document_scores[doc_id]["aggregated_score"] = -float('inf')

    sorted_documents = sorted(document_scores.values(), key=lambda x: x["aggregated_score"], reverse=True)

    # --- Display Results ---
    print(f"\nQuery: {query_text}")
    print(f"Top {len(sorted_documents)} documents from BioASQ service, re-ranked (higher score is more relevant):")
    if sorted_documents:
        for i, doc_info in enumerate(sorted_documents):
            pubmed_url = doc_info.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{doc_info['doc_id']}/")
            print(f"{i+1}. PMID: {doc_info['doc_id']} (Score: {doc_info['aggregated_score']:.4f}) - URL: {pubmed_url}")
            print(f"    Title: {doc_info['title']}")
    else:
        print("No documents were re-ranked.")

if __name__ == "__main__":
    if not INFERENCE_CONFIG.get("model_path") or not os.path.exists(INFERENCE_CONFIG["model_path"]):
        logger.error(f"Model path '{INFERENCE_CONFIG.get('model_path')}' not configured or does not exist.")
    elif not INFERENCE_CONFIG.get("bioasq_api_endpoint"):
        logger.error("Please set your BioASQ API endpoint URL in the INFERENCE_CONFIG (bioasq_api_endpoint).")
    else:
        run_document_inference_with_pubmed(INFERENCE_CONFIG)
