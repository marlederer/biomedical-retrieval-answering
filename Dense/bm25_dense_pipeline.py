import os
import sys # Import sys for exit
import json # Import json for loading config

# Add BM25 folder to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
bm25_dir = os.path.join(current_dir, '..', 'BM25')
sys.path.append(bm25_dir)

# Import functions from our refactored modules using relative imports
from keyword_extractor import extract_keyword_combinations
from api_client import search_pubmed, fetch_pubmed_details
from ranker import rank_articles_bm25
from evaluation import load_ground_truth, calculate_precision_recall_f1, extract_pmid_from_url
from dense_retrieval import encode_contexts, encode_query, rank_with_dense_retrieval, select_top_k_snippets_from_articles

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
BIOASQ_API_ENDPOINT = config.get("api_endpoint", "http://bioasq.org:8000/pubmed") # Provide default
num_candidates_per_combination = config.get("num_candidates_per_combination", 100)
num_initial_bm25 = config.get("num_initial_bm25", 1000)  # NEW: how many BM25 results to keep
print("num_initial_bm25: " + str(num_initial_bm25))
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

            # 1. Extract Keyword Combinations
            keyword_combinations_list = extract_keyword_combinations(question_body)
            
            # 2. Call API for Candidates
            all_candidate_articles = []
            seen_article_pmids = set()

            for combo in keyword_combinations_list:
                pmids = search_pubmed(combo, max_results=num_candidates_per_combination)
                candidate_articles_from_api = fetch_pubmed_details(pmids)
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
                print(f"Retrieved {len(all_candidate_articles)} unique candidate articles in total.")

                # Step 3: Rank Candidates with BM25
                print("Ranking candidates locally using BM25...")
                bm25_top_pmids = rank_articles_bm25(question_body, all_candidate_articles, top_k=num_initial_bm25)
                
                # Filter candidate articles to only BM25 top results
                filtered_articles = [article for article in all_candidate_articles if article['pmid'] in bm25_top_pmids]

                if not filtered_articles:
                    print("No articles passed BM25 filtering.")
                    continue

                # Step 4: Re-rank with Dense Retrieval
                print("Re-ranking candidates using Dense Retrieval...")
                passages = [[article['title'],article['abstract']] for article in filtered_articles]
                passage_embeddings = encode_contexts(passages)
                query_embedding = encode_query(question_body)
                dense_top_indices = rank_with_dense_retrieval(query_embedding, passage_embeddings, top_k=num_final_results)

                # Final top PMIDs after dense re-ranking
                final_top_pmids = [filtered_articles[idx]['pmid'] for idx in dense_top_indices]

                pmid_to_article_map = {art['pmid']: art for art in all_candidate_articles if 'pmid' in art}
                
                ranked_articles_for_output = []
                for pmid in final_top_pmids:
                    if pmid in pmid_to_article_map:
                        ranked_articles_for_output.append(pmid_to_article_map[pmid])

                output_documents_urls_list = [art.get('url') for art in ranked_articles_for_output if art.get('url')]

                if ranked_articles_for_output:
                    top_k_snippets = select_top_k_snippets_from_articles(
                    query_embedding=query_embedding,
                    articles=ranked_articles_for_output
                    )  

                    output_snippets_list.extend(top_k_snippets)
                            
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
        questions_with_results = 0 # Count questions where we actually got results to rank

        print(f"\\nStarting evaluation for {len(input_questions_list)} questions...")

        for idx, input_question_obj in enumerate(input_questions_list):
            question_body = input_question_obj['body']
            relevant_doc_urls = input_question_obj.get('documents', [])
            # Extract PMIDs from ground truth URLs, ensuring they are strings and non-empty
            relevant_pmids_str = {str(pmid) for url in relevant_doc_urls if (pmid := extract_pmid_from_url(url))}

            print("-" * 40)
            print(f"Processing question ({idx + 1}/{len(input_questions_list)}): {question_body[:100]}...")
            processed_questions_count += 1

            # 1. Extract Keyword Combinations
            keyword_combinations_list = extract_keyword_combinations(question_body)
            print(f"Extracted keyword combinations: {keyword_combinations_list}") # Optional debug

            # 2. Call API for Candidates
            all_candidate_articles = []
            seen_article_ids = set() # Use PMIDs here for uniqueness tracking
            print(f"Fetching top {num_candidates_per_combination} candidates per keyword combination...") # Optional debug

            for combo in keyword_combinations_list:
                print(f"  Fetching for combination: '{combo}'") # Optional debug
                candidate_articles = call_bioasq_api_search(combo, num_candidates_per_combination, BIOASQ_API_ENDPOINT)
                if candidate_articles:
                    print(f"    Retrieved {len(candidate_articles)} candidates for '{combo}'.") # Optional debug
                    for article in candidate_articles:
                        # Extract PMID consistently using the helper
                        pmid = extract_pmid_from_url(article.get('url')) or extract_pmid_from_url(article.get('id'))

                        if pmid and pmid not in seen_article_ids:
                            # Store the definitive PMID back into the article dict for easier access later if needed
                            article['pmid'] = pmid # Ensure pmid is stored if found
                            all_candidate_articles.append(article)
                            seen_article_ids.add(pmid)
                # else:
                    # print(f"    Failed to retrieve candidates for combination '{combo}'.") # Optional debug


            if all_candidate_articles:
                print(f"Retrieved {len(all_candidate_articles)} unique candidate articles in total.")

                # Step 3: Rank Candidates with BM25
                print("Ranking candidates locally using BM25...")
                bm25_top_pmids = rank_articles_bm25(question_body, all_candidate_articles, top_k=num_initial_bm25)

                # Filter candidate articles to only BM25 top results
                filtered_articles = [article for article in all_candidate_articles if article['pmid'] in bm25_top_pmids]

                if not filtered_articles:
                    print("No articles passed BM25 filtering.")
                    continue

                # Step 4: Re-rank with Dense Retrieval
                print("Re-ranking candidates using Dense Retrieval...")
                passages = [[article['title'],article['abstract']] for article in filtered_articles]
                passage_embeddings = encode_contexts(passages)
                query_embedding = encode_query(question_body)
                dense_top_indices = rank_with_dense_retrieval(query_embedding, passage_embeddings, top_k=num_final_results)

                # Final top PMIDs after dense re-ranking
                final_top_pmids = [filtered_articles[idx]['pmid'] for idx in dense_top_indices]

                # Ensure ground truth PMIDs are strings for comparison
                relevant_pmids_str = set(map(str, relevant_pmids_str))

                # 5. Evaluate Results
                precision, recall, f1 = calculate_precision_recall_f1(final_top_pmids, relevant_pmids_str)

                # 6. Pretty Print Results & Evaluation
                print("-" * 40)
                print(f"Final Top {num_final_results} ranked article PMIDs for question: {question}")
                print("-" * 40)
                # Add pubmed URLs to the final results for clarity
                final_top_urls = [f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" for pmid in final_top_pmids]
                for i, url in enumerate(final_top_urls, start=1):
                    print(f"{i}. {url}")
                print("-" * 40)
                print(f"Ground truth PMIDs ({len(relevant_pmids_str)}):")
                relevant_urls = {f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" for pmid in relevant_pmids_str}
                for i, url in enumerate(sorted(list(relevant_urls)), start=1): # Sort for consistent display
                    print(f"{i}. {url}")
                print("-" * 40)
                # Print overlap between retrieved top K and relevant PMIDs
                print("Overlap between ranked top K and relevant PMIDs:")
                overlap_top_k = set(map(str, final_top_pmids)).intersection(relevant_pmids_str)
                overlap_top_k_urls = {f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" for pmid in overlap_top_k}
                if overlap_top_k_urls:
                    for i, url in enumerate(sorted(list(overlap_top_k_urls)), start=1):
                        print(f"{i}. {url}")
                else:
                    print("None")
                print("-" * 40)
                # Print overlap of *all* retrieved unique PMIDs with relevant PMIDs
                overlap_all = seen_article_ids.intersection(relevant_pmids_str)
                print(f"Overlap of *all* retrieved unique PMIDs ({len(seen_article_ids)}) with relevant PMIDs: {len(overlap_all)}")
                print("-" * 40)

                print(f"Precision@{num_final_results}: {precision:.4f}")
                print(f"Recall@{num_final_results}: {recall:.4f}")
                print(f"F1-Score@{num_final_results}: {f1:.4f}")

                total_precision += precision
                total_recall += recall
                total_f1 += f1
                questions_with_results += 1 # Increment count for averaging later
            else:
                print("Failed to retrieve any candidate articles from the API for this question.")
                # Assign 0 scores if no articles were retrieved, don't count towards average
                precision, recall, f1 = 0.0, 0.0, 0.0
                print(f"Precision@{num_final_results}: {precision:.4f}")
                print(f"Recall@{num_final_results}: {recall:.4f}")
                print(f"F1-Score@{num_final_results}: {f1:.4f}")


        # Calculate average metrics based only on questions where results were obtained
        print("-" * 40)
        print("\n--- Overall Evaluation Results ---")
        if questions_with_results > 0:
            avg_precision = total_precision / questions_with_results
            avg_recall = total_recall / questions_with_results
            avg_f1 = total_f1 / questions_with_results
            print(f"Average Precision@{num_final_results}: {avg_precision:.4f}")
            print(f"Average Recall@{num_final_results}: {avg_recall:.4f}")
            print(f"Average F1-Score@{num_final_results}: {avg_f1:.4f}")
            print(f"Metrics averaged over {questions_with_results} questions (out of {processed_questions_count} total processed) for which candidates were retrieved.")
        elif processed_questions_count > 0:
            print("No candidate articles were retrieved for any processed question. Cannot calculate average metrics.")
        else:
            print("No questions were processed (check ground truth file or debug settings).")