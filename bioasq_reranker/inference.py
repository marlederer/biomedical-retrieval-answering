# inference.py (with BioASQ API for candidate docs & document re-ranking)
import os
import json
import logging
from collections import defaultdict
import time  # For polite API usage

import torch
from transformers import AutoTokenizer
from api_client import call_bioasq_api_search  # Updated import

# Attempt to import keyword_extractor and pmid extraction utility
try:
    from keyword_extractor import extract_keyword_combinations
except ImportError:
    from keyword_extractor import extract_keyword_combinations as local_extract_keyword_combinations
    extract_keyword_combinations = local_extract_keyword_combinations


def extract_pmid_from_url(url_string):
    """
    Extracts PubMed ID (PMID) from a URL string or if the string itself is a PMID.
    Replace with your robust implementation.
    """
    if not url_string or not isinstance(url_string, str):
        return None
    url_lower = url_string.lower()
    if "ncbi.nlm.nih.gov/pubmed/" in url_lower:
        try:
            return url_string.split("pubmed/")[1].split("/")[0].split("?")[0]
        except IndexError:
            pass
    if url_string.isdigit():  # If the ID itself is passed
        return url_string
    return None


from model_arch import CrossEncoderReRanker  # Assuming model_arch.py contains your class

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_documents_with_keyword_combinations(query, config):
    """
    Fetches candidate documents from BioASQ using keyword combinations extracted from the query.
    """
    api_endpoint_url = config.get("bioasq_api_endpoint")
    num_candidates_per_combo = config.get("num_candidates_per_combination")

    if not api_endpoint_url:
        logger.error("BioASQ API endpoint ('bioasq_api_endpoint') not configured.")
        return []
    if num_candidates_per_combo is None:
        logger.error("Number of candidates per combination ('num_candidates_per_combination') not configured.")
        return []
    if not isinstance(num_candidates_per_combo, int) or num_candidates_per_combo <= 0:
        logger.error("'num_candidates_per_combination' must be a positive integer.")
        return []

    logger.info(f"Extracting keyword combinations for query: {query[:100]}...")
    try:
        keyword_combinations_list = extract_keyword_combinations(query)
    except Exception as e:
        logger.error(f"Error during keyword extraction: {e}", exc_info=True)  # Log the full traceback
        return []

    if not keyword_combinations_list:
        logger.warning(f"No keyword combinations extracted for query: '{query}'.")
        return []

    all_fetched_articles = []
    seen_article_pmids = set()

    logger.info(f"Fetching up to {num_candidates_per_combo} candidates per keyword combination from {api_endpoint_url} for {len(keyword_combinations_list)} combinations.")

    for combo_idx, combo_keywords in enumerate(keyword_combinations_list):
        current_query_str = " ".join(combo_keywords) if isinstance(combo_keywords, list) else combo_keywords
        logger.info(f"Processing combination {combo_idx + 1}/{len(keyword_combinations_list)}: '{current_query_str}'")

        try:
            articles_data_from_api = call_bioasq_api_search(
                query_keywords=current_query_str,
                num_articles_to_fetch=num_candidates_per_combo,
                api_endpoint_url=api_endpoint_url
            )

            if articles_data_from_api:
                logger.debug(f"Retrieved {len(articles_data_from_api)} articles for combination '{current_query_str}'.")
                for article_from_api in articles_data_from_api:
                    if not isinstance(article_from_api, dict):
                        logger.warning(f"Skipping non-dictionary item from API for combo '{current_query_str}': {article_from_api}")
                        continue

                    pmid_source_url = article_from_api.get('url', article_from_api.get('uri'))
                    pmid_source_id = str(article_from_api.get('id', ''))

                    pmid = extract_pmid_from_url(pmid_source_url) or extract_pmid_from_url(pmid_source_id)

                    if pmid and pmid not in seen_article_pmids:
                        fetched_article = {
                            "doc_id": pmid,
                            "title": article_from_api.get("title", ""),
                            "abstract": article_from_api.get("abstract", ""),
                            "url": article_from_api.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                            "original_api_response": article_from_api
                        }
                        all_fetched_articles.append(fetched_article)
                        seen_article_pmids.add(pmid)
                    elif pmid and pmid in seen_article_pmids:
                        logger.debug(f"PMID {pmid} already seen, skipping.")
                    elif not pmid:
                        logger.warning(f"Could not extract PMID for combo '{current_query_str}'. URL: '{pmid_source_url}', ID: '{pmid_source_id}'. Data: {article_from_api}")
            else:
                logger.debug(f"No articles found for keyword combination: '{current_query_str}'")
        except Exception as e:
            logger.error(f"Error fetching or processing articles for combination '{current_query_str}': {e}")

    logger.info(f"Fetched a total of {len(all_fetched_articles)} unique candidate articles after processing all keyword combinations.")
    return all_fetched_articles


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

    query_text = config["query"]

    candidate_documents = fetch_documents_with_keyword_combinations(
        query_text,
        config
    )

    if not candidate_documents:
        logger.warning(f"No candidate documents fetched using keyword combinations for query '{query_text}'. Cannot proceed.")
        print(f"\nQuery: {query_text}")
        print("No documents found using keyword combinations for this query.")
        return

    passage_field = config.get("passage_field_from_pubmed", "abstract")
    document_scores = defaultdict(lambda: {"doc_id": "", "passages_scores": [], "aggregated_score": -float('inf'), "title": "", "original_doc_data": None})

    logger.info(f"Re-ranking {len(candidate_documents)} documents fetched using keyword combinations...")

    for doc_data in candidate_documents:
        doc_id = doc_data.get("doc_id")
        if not doc_id:
            logger.warning(f"Skipping document with no doc_id: {doc_data.get('title', 'N/A')}")
            continue

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

        doc_passages = split_document_into_passages(
            text_to_process,
            max_passage_tokens=config.get("max_passage_tokens_split", 200),  # From INFERENCE_CONFIG
            stride=config.get("passage_split_stride", 100),                 # From INFERENCE_CONFIG
            tokenizer_for_len_check=tokenizer                               # Pass the loaded tokenizer
        )

        if not doc_passages:
            logger.warning(f"No passages generated for document {doc_id}. Skipping.")
            document_scores[doc_id]["aggregated_score"] = -float('inf')
            document_scores[doc_id]["passages_scores"] = []
            continue

        batch_size_passage_config = config.get("batch_size_passage", 16)

        ranked_passages_for_doc = rerank_individual_passages(
            query_text,
            doc_passages,
            passage_reranker_model,
            tokenizer,
            device,
            max_seq_length=max_seq_length_passage_config,
            batch_size=batch_size_passage_config
        )
        document_scores[doc_id]["passages_scores"] = ranked_passages_for_doc

        aggregation_method = config.get("aggregation_method", "max")
        if ranked_passages_for_doc:
            if aggregation_method == "max":
                document_scores[doc_id]["aggregated_score"] = max(p["score"] for p in ranked_passages_for_doc)
            else:
                logger.warning(f"Unsupported aggregation_method: {aggregation_method}. Defaulting to 'max'.")
                document_scores[doc_id]["aggregated_score"] = max(p["score"] for p in ranked_passages_for_doc)
        else:
            document_scores[doc_id]["aggregated_score"] = -float('inf')

    sorted_documents = sorted(document_scores.values(), key=lambda x: x["aggregated_score"], reverse=True)

    print(f"\nQuery: {query_text}")
    print(f"Top {len(sorted_documents)} documents fetched using keyword combinations, re-ranked (higher score is more relevant):")
    if sorted_documents:
        for i, doc_info in enumerate(sorted_documents):
            pubmed_url = doc_info.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{doc_info['doc_id']}/")
            print(f"{i+1}. PMID: {doc_info['doc_id']} (Score: {doc_info['aggregated_score']:.4f}) - URL: {pubmed_url}")
            print(f"    Title: {doc_info['title']}")
    else:
        print("No documents were re-ranked.")


if __name__ == "__main__":
    INFERENCE_CONFIG = {
        "model_path": "bioasq_reranker/saved_models/my_biomedbert_reranker_hardcoded",
        "query": "What is the genetic basis of addiction?",
        "bioasq_api_endpoint": "http://bioasq.org:8000/pubmed", 
        "num_candidates_per_combination": 5,
        "passage_field_from_pubmed": "abstract",
        "max_seq_length_passage": 512,
        "batch_size_passage": 16,
        "aggregation_method": "max",
        "max_passage_tokens_split": 200,
        "passage_split_stride": 100,
    }

    if not INFERENCE_CONFIG.get("bioasq_api_endpoint") or INFERENCE_CONFIG.get("bioasq_api_endpoint") == "YOUR_BIOASQ_API_ENDPOINT_URL":
        logger.error("Please configure 'bioasq_api_endpoint' in INFERENCE_CONFIG.")
    elif not INFERENCE_CONFIG.get("num_candidates_per_combination"):
        logger.error("Please configure 'num_candidates_per_combination' in INFERENCE_CONFIG.")
    else:
        run_document_inference_with_pubmed(INFERENCE_CONFIG)
