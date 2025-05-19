import os
import sys # Import sys for exit
import json # Import json for loading config

# Import functions from our refactored modules using relative imports
from keyword_extractor import extract_keyword_combinations
from api_client import call_bioasq_api_search
from ranker import rank_articles_bm25
from evaluation import calculate_precision_recall_f1, extract_pmid_from_url # Removed load_ground_truth as we load raw data here

# --- Configuration Loading ---
script_dir = os.path.dirname(__file__) # Get the directory where the script is located
config_filepath = os.path.join(script_dir, 'config.json')
# The ground_truth_filepath is now the input file for questions
# input_questions_filepath = os.path.join(script_dir, '..', 'training13b.json') # Remove this line

try:
    with open(config_filepath, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Configuration file not found at {config_filepath}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {config_filepath}")
    sys.exit(1)

# --- Use loaded configuration ---
BIOASQ_API_ENDPOINT = config.get("api_endpoint", "http://bioasq.org:8000/pubmed")
num_candidates_per_combination = config.get("num_candidates_per_combination", 100)
num_final_results = config.get("num_final_results", 10)
debug_mode = config.get("debug_mode", False)
debug_limit = config.get("debug_limit", 5)
output_mode = config.get("output_mode", "evaluation") # New setting: "evaluation" or "json"
input_questions_filepath_config = config.get("input_questions_filepath", "../training13b.json") # Default if not in config

# Resolve the input questions filepath relative to the script directory
input_questions_filepath = os.path.abspath(os.path.join(script_dir, input_questions_filepath_config))


# --- Main Orchestration ---

if __name__ == "__main__":

    # Load raw input questions data
    print(f"Loading input questions from: {input_questions_filepath}")
    try:
        with open(input_questions_filepath, 'r', encoding='utf-8') as f:
            raw_input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input questions file not found at {input_questions_filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_questions_filepath}")
        sys.exit(1)

    input_questions_list = raw_input_data.get('questions', [])

    if not input_questions_list:
        print("Warning: No questions found in the input file. Exiting.")
        sys.exit(1)

    if debug_mode and debug_limit > 0 and len(input_questions_list) > debug_limit:
        print(f"Debug mode: Limiting to first {debug_limit} questions.")
        input_questions_list = input_questions_list[:debug_limit]


    if output_mode == "json":
        all_output_question_data = []
        print(f"\\nStarting JSON generation for {len(input_questions_list)} questions...")

        for idx, input_question_obj in enumerate(input_questions_list):
            question_id = input_question_obj['id']
            question_type = input_question_obj['type']
            question_body = input_question_obj['body']

            print("-" * 40)
            print(f"Processing question for JSON output ({idx + 1}/{len(input_questions_list)}): {question_body[:100]}...")

            keyword_combinations_list = extract_keyword_combinations(question_body)
            
            all_candidate_articles = []
            seen_article_pmids = set()

            for combo in keyword_combinations_list:
                candidate_articles_from_api = call_bioasq_api_search(combo, num_candidates_per_combination, BIOASQ_API_ENDPOINT)
                if candidate_articles_from_api:
                    for article in candidate_articles_from_api:
                        pmid = extract_pmid_from_url(article.get('url')) or extract_pmid_from_url(article.get('id'))
                        if pmid and pmid not in seen_article_pmids:
                            article['pmid'] = pmid
                            if 'url' not in article or not article['url']: # Ensure URL is present
                                article['url'] = f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
                            all_candidate_articles.append(article)
                            seen_article_pmids.add(pmid)
            
            output_snippets_list = []
            output_documents_urls_list = []

            if all_candidate_articles:
                ranked_pmids = rank_articles_bm25(question_body, all_candidate_articles, top_k=num_final_results)
                
                pmid_to_article_map = {art['pmid']: art for art in all_candidate_articles if 'pmid' in art}
                
                ranked_articles_for_output = []
                for pmid in ranked_pmids:
                    if pmid in pmid_to_article_map:
                        ranked_articles_for_output.append(pmid_to_article_map[pmid])

                output_documents_urls_list = [art.get('url') for art in ranked_articles_for_output if art.get('url')]

                if ranked_articles_for_output:
                    first_ranked_article = ranked_articles_for_output[0]
                    # Prefer abstract, then title, then empty string for snippet text
                    doc_text_content = first_ranked_article.get('abstract', '')
                    if not doc_text_content: # If abstract is empty or not present
                        doc_text_content = first_ranked_article.get('title', '')
                    
                    snippet_text = doc_text_content[:150]
                    snippet_document_url = first_ranked_article.get('url')

                    if snippet_document_url: # Only add snippet if document URL exists
                        output_snippets_list.append({
                            "document": snippet_document_url,
                            "text": snippet_text,
                            "offsetInBeginSection": 0,
                            "offsetInEndSection": 150,
                            "beginSection": "abstract",
                            "endSection": "abstract"
                        })
            
            question_output_entry = {
                "id": question_id,
                "type": question_type,
                "body": question_body,
                "ideal_answer": "",
                "exact_answer": [],
                "documents": output_documents_urls_list,
                "snippets": output_snippets_list
            }
            all_output_question_data.append(question_output_entry)

        output_json_filename = "bioasq_output.json"
        output_json_filepath = os.path.join(script_dir, output_json_filename)
        with open(output_json_filepath, 'w', encoding='utf-8') as outfile:
            json.dump({"questions": all_output_question_data}, outfile, indent=4)
        print(f"\\nJSON output successfully written to {output_json_filepath}")

    elif output_mode == "evaluation":
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        processed_questions_count = 0
        questions_with_results = 0

        print(f"\\nStarting evaluation for {len(input_questions_list)} questions...")

        for idx, input_question_obj in enumerate(input_questions_list):
            question_body = input_question_obj['body']
            relevant_doc_urls = input_question_obj.get('documents', [])
            # Extract PMIDs from ground truth URLs, ensuring they are strings and non-empty
            relevant_pmids_str = {str(pmid) for url in relevant_doc_urls if (pmid := extract_pmid_from_url(url))}


            print("-" * 40)
            print(f"Processing question ({idx + 1}/{len(input_questions_list)}): {question_body[:100]}...")
            processed_questions_count += 1

            keyword_combinations_list = extract_keyword_combinations(question_body)
            # ... (rest of the original code for fetching and ranking, adapted) ...
            all_candidate_articles = []
            seen_article_ids = set() 
            # print(f"Fetching top {num_candidates_per_combination} candidates per keyword combination...")

            for combo in keyword_combinations_list:
                # print(f"  Fetching for combination: '{combo}'")
                candidate_articles_from_api = call_bioasq_api_search(combo, num_candidates_per_combination, BIOASQ_API_ENDPOINT)
                if candidate_articles_from_api:
                    # print(f"    Retrieved {len(candidate_articles_from_api)} candidates for '{combo}'.")
                    for article in candidate_articles_from_api:
                        pmid = extract_pmid_from_url(article.get('url')) or extract_pmid_from_url(article.get('id'))
                        if pmid and pmid not in seen_article_ids:
                            article['pmid'] = pmid
                            all_candidate_articles.append(article)
                            seen_article_ids.add(pmid)
            
            if all_candidate_articles:
                # print(f"Retrieved {len(all_candidate_articles)} unique candidate articles in total.")
                # print("Ranking candidates locally using BM25...")
                final_top_pmids = rank_articles_bm25(question_body, all_candidate_articles, top_k=num_final_results)
                # Ensure final_top_pmids are strings for comparison
                final_top_pmids_str = set(map(str, final_top_pmids))

                precision, recall, f1 = calculate_precision_recall_f1(final_top_pmids_str, relevant_pmids_str)

                # ... (rest of the original pretty print and accumulation logic) ...
                print("-" * 40)
                print(f"Final Top {num_final_results} ranked article PMIDs for question: {question_body[:100]}...")
                final_top_urls = [f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" for pmid in final_top_pmids_str]
                for i, url in enumerate(final_top_urls, start=1):
                    print(f"{i}. {url}")
                print("-" * 40)
                print(f"Ground truth PMIDs ({len(relevant_pmids_str)}):")
                relevant_urls = {f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" for pmid in relevant_pmids_str}
                for i, url in enumerate(sorted(list(relevant_urls)), start=1):
                    print(f"{i}. {url}")
                print("-" * 40)
                overlap_top_k = final_top_pmids_str.intersection(relevant_pmids_str)
                print(f"Overlap between ranked top K and relevant PMIDs: {len(overlap_top_k)}")
                # ... (further details if needed) ...
                print("-" * 40)
                print(f"Precision@{num_final_results}: {precision:.4f}")
                print(f"Recall@{num_final_results}: {recall:.4f}")
                print(f"F1-Score@{num_final_results}: {f1:.4f}")

                total_precision += precision
                total_recall += recall
                total_f1 += f1
                questions_with_results += 1
            else:
                print("Failed to retrieve any candidate articles from the API for this question.")
                print(f"Precision@{num_final_results}: 0.0000")
                print(f"Recall@{num_final_results}: 0.0000")
                print(f"F1-Score@{num_final_results}: 0.0000")

        # Calculate average metrics
        print("-" * 40)
        print("\\n--- Overall Evaluation Results ---")
        if questions_with_results > 0:
            avg_precision = total_precision / questions_with_results
            avg_recall = total_recall / questions_with_results
            avg_f1 = total_f1 / questions_with_results
            print(f"Average Precision@{num_final_results}: {avg_precision:.4f}")
            print(f"Average Recall@{num_final_results}: {avg_recall:.4f}")
            print(f"Average F1-Score@{num_final_results}: {avg_f1:.4f}")
            print(f"Metrics averaged over {questions_with_results} questions (out of {processed_questions_count} total processed) for which candidates were retrieved.")
        elif processed_questions_count > 0:
            print("No candidate articles were retrieved for any processed question where evaluation was attempted. Cannot calculate average metrics.")
        else:
            print("No questions were processed (check input file or debug settings).")
    else:
        print(f"Error: Unknown output_mode '{output_mode}'. Please use 'json' or 'evaluation'.")
        sys.exit(1)