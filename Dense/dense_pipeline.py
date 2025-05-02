import os
import sys # Import sys for exit
import json # Import json for loading config

# Add BM25 folder to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
bm25_dir = os.path.join(current_dir, '..', 'BM25')
sys.path.append(bm25_dir)

# Import functions from our refactored modules using relative imports
from keyword_extractor import extract_keyword_combinations
from api_client import call_bioasq_api_search
from ranker import rank_articles_bm25
from evaluation import load_ground_truth, calculate_precision_recall_f1, extract_pmid_from_url
from dense_retrieval import encode_contexts, encode_query, rank_with_dense_retrieval

# --- Configuration Loading ---
script_dir = os.path.dirname(__file__) # Get the directory where the script is located
config_filepath = os.path.join(script_dir, 'config.json')
ground_truth_filepath = os.path.join(script_dir, '..', 'training13b.json')

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
num_final_results = config.get("num_final_results", 10)
debug_mode = config.get("debug_mode", True)
debug_limit = config.get("debug_limit", 2)

# --- Main Orchestration ---

if __name__ == "__main__":

    # Load ground truth data
    print(f"Loading ground truth from: {ground_truth_filepath}")
    ground_truth_data = load_ground_truth(ground_truth_filepath, debug_mode, debug_limit)

    if ground_truth_data is None: # Check if loading failed
        print("Exiting due to critical error loading ground truth data.")
        sys.exit(1) # Exit script if ground truth is essential and failed to load
    if not ground_truth_data:
        print("Warning: No ground truth data loaded. Evaluation might be empty.")
        # Decide if you want to exit or continue without evaluation data
        # sys.exit(1)

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    processed_questions = 0
    questions_with_results = 0 # Count questions where we actually got results to rank

    print(f"\nStarting evaluation for {len(ground_truth_data)} questions...")

    for question, relevant_pmids in ground_truth_data.items():
        print("-" * 40)
        print(f"Processing question ({processed_questions + 1}/{len(ground_truth_data)}): {question}")
        processed_questions += 1

        # 1. Extract Keyword Combinations
        keyword_combinations_list = extract_keyword_combinations(question)
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

            # 3. Rank Candidates Locally using Dense Retrieval
            print("Ranking candidates locally using Dense Retrieval...")
            # 1. Encode candidates
            passages = [[article['title'],article['abstract']] for article in all_candidate_articles]
            passage_embeddings = encode_contexts(passages)

            # 2. Encode question
            query_embedding = encode_query(question)

            # 3. Rank
            final_top_indices = rank_with_dense_retrieval(query_embedding, passage_embeddings, top_k=num_final_results)

            # 4. Map back indices to PMIDs
            final_top_pmids = [all_candidate_articles[idx]['pmid'] for idx in final_top_indices]

            # Ensure ground truth PMIDs are strings for comparison
            relevant_pmids_str = set(map(str, relevant_pmids))

            # 4. Evaluate Results
            precision, recall, f1 = calculate_precision_recall_f1(final_top_pmids, relevant_pmids_str)

            # 5. Pretty Print Results & Evaluation
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
        print(f"Metrics averaged over {questions_with_results} questions (out of {processed_questions} total processed) for which candidates were retrieved.")
    elif processed_questions > 0:
         print("No candidate articles were retrieved for any processed question. Cannot calculate average metrics.")
    else:
        print("No questions were processed (check ground truth file or debug settings).")