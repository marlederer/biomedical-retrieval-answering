import requests
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import combinations # Import combinations
from rank_bm25 import BM25Okapi # Popular BM25 library
import os # Import os for path joining

# --- 1. Keyword Extraction ---

def extract_keyword_combinations(question_text):
    """Extracts single keywords, pairs, and triplets after simple splitting and stopword removal."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(question_text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Generate single keywords
    single_keywords = filtered_tokens

    # Single keywords combined with spaces " ".join(filtered_tokens)
    string_of_keywords = [" ".join(filtered_tokens)]

    # Generate all unique pairs of keywords
    keyword_pairs = list(combinations(filtered_tokens, 2))
    pair_strings = [" ".join(pair) for pair in keyword_pairs]

    # Generate all unique triplets of keywords
    keyword_triplets = list(combinations(filtered_tokens, 3))
    triplet_strings = [" ".join(triplet) for triplet in keyword_triplets]


    # Combine all combinations
    all_combinations = single_keywords + string_of_keywords + pair_strings + triplet_strings
    return all_combinations

# --- 2. API Call (Candidate Retrieval) ---

def call_bioasq_api_search(query_keywords, num_articles_to_fetch, api_endpoint_url, session_id=None):
    """
    Calls the BioASQ PubMed search API.
    First, it retrieves a session endpoint, then uses it for the actual search.
    """
    try:
        # Step 1: Get session endpoint
        session_response = requests.get(api_endpoint_url, timeout=60)
        # Check if the session endpoint was retrieved successfully
        if session_response.status_code != 200:
            print(f"Error: Failed to retrieve session endpoint. Status code: {session_response.status_code}")
            print(f"Response content: {session_response.text}")
            return []

        session_endpoint_url = session_response.text
        
        # Step 2: Perform the search using the session endpoint
        payload = {
            "findPubMedCitations": [query_keywords, 0, num_articles_to_fetch]
        }

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        #Print request for debugging
        print(f"Requesting session endpoint: {session_endpoint_url}")
        daten = {"json": json.dumps(payload)}
        print(daten) # Optional: uncomment for debugging
        response = requests.post(session_endpoint_url, headers=headers, data={"json": json.dumps(payload)}, timeout=60)

        if response.status_code != 200:
            print(f"Error: Failed to retrieve search results. Status code: {response.status_code}")
            print(f"Response content: {response.text}")
            return []

        try:
            results = response.json()
        except json.JSONDecodeError:
            print(f"Error decoding search results response: {response.text}")
            return []

        # --- 3. Local Data Preparation ---
        # Extract relevant info (pmid/url, abstract, title)
        if "result" in results and "documents" in results["result"]:
            articles_data = []
            for doc in results["result"]["documents"]:
                pmid = doc.get('pmid')
                doc_url = f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" if pmid else None
                articles_data.append({
                    "id": pmid or doc_url,
                    "url": doc_url,
                    "title": doc.get('title', ''),
                    "abstract": doc.get('documentAbstract', '')
                })
            return articles_data
        else:
            print("Warning: Unexpected API response format:", results)
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error calling BioASQ API: {e}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding API response: {response.text}")
        return []


# --- 4. Preprocessing ---

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text_for_bm25(text):
    """Basic preprocessing: tokenize, lower, remove stops & non-alphanum."""
    if not isinstance(text, str): # Handle potential None or non-string abstracts
        return []
    tokens = word_tokenize(text.lower())
    processed_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return processed_tokens

# --- 5. Local BM25 Ranking ---

def rank_articles_bm25(query_text, articles_data, top_k=10):
    """Ranks articles using BM25 on the local set."""
    if not articles_data:
        return []

    # Create the corpus for BM25 - list of tokenized documents
    corpus = []
    doc_ids = [] # Keep track of original IDs/URLs
    for article in articles_data:
        # Combine title and abstract for richer content
        content = f"{article.get('title', '')} {article.get('abstract', '')}"
        processed_content = preprocess_text_for_bm25(content)
        corpus.append(processed_content)
        # Use the id from the URL as doc_ids example :http://www.ncbi.nlm.nih.gov/pubmed/39436382 we want the last digits
        doc_id = article.get('id', '').split('/')[-1] if article.get('url') else None
        doc_ids.append(doc_id)


    # Preprocess the query
    tokenized_query = preprocess_text_for_bm25(query_text)

    # Initialize and run BM25
    # You might tune k1 and b parameters based on validation data if available
    bm25 = BM25Okapi(corpus)
    doc_scores = bm25.get_scores(tokenized_query)

    # Combine IDs and scores, then sort
    scored_docs = list(zip(doc_ids, doc_scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # --- 6. Final Selection ---
    top_ranked_urls = [doc[0] for doc in scored_docs[:top_k]]
    return top_ranked_urls

# --- Helper Functions for Evaluation ---

# Function to extract PMID from a PubMed URL
def extract_pmid_from_url(url):
    """Extracts the PubMed ID (PMID) from a PubMed URL."""
    if isinstance(url, str) and 'pubmed/' in url:
        # Handle potential query parameters or fragments
        base_url = url.split('?')[0].split('#')[0]
        return base_url.split('/')[-1]
    # If it's already just a number (potentially a PMID string)
    elif isinstance(url, str) and url.isdigit():
        return url
    return None

# Function to load ground truth data with DEBUG mode to only load first N items
def load_ground_truth(filepath, debug_mode=False, debug_limit=5):
    """Loads questions and their relevant document PMIDs from a JSON file."""
    ground_truth = {}
    if debug_mode:
        print(f"DEBUG MODE: Loading only the first {debug_limit} items from the ground truth file.")    
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data.get('questions', [])[:debug_limit]:
                    question_body = item.get('body')
                    doc_urls = item.get('documents', [])
                    if question_body:
                        # Extract PMIDs and store as a set for efficient lookup
                        pmids = {extract_pmid_from_url(url) for url in doc_urls if extract_pmid_from_url(url)}
                        if pmids: # Only add if there are valid PMIDs
                            ground_truth[question_body] = pmids
        except FileNotFoundError:
            print(f"Error: Ground truth file not found at {filepath}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filepath}")
        except Exception as e:
            print(f"An unexpected error occurred while loading ground truth: {e}")
        return ground_truth
    else:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data.get('questions', []):
                    question_body = item.get('body')
                    doc_urls = item.get('documents', [])
                    if question_body:
                        # Extract PMIDs and store as a set for efficient lookup
                        pmids = {extract_pmid_from_url(url) for url in doc_urls if extract_pmid_from_url(url)}
                        if pmids: # Only add if there are valid PMIDs
                            ground_truth[question_body] = pmids
        except FileNotFoundError:
            print(f"Error: Ground truth file not found at {filepath}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filepath}")
        except Exception as e:
            print(f"An unexpected error occurred while loading ground truth: {e}")
        return ground_truth
# Function to calculate precision, recall, and F1-score
def calculate_precision_recall_f1(retrieved_pmids, relevant_pmids):
    """Calculates Precision@k, Recall@k, and F1@k."""
    if not relevant_pmids: # Avoid division by zero if no relevant documents exist
        return 0.0, 0.0, 0.0

    # Ensure retrieved_pmids are strings for comparison
    retrieved_set = set(map(str, retrieved_pmids))
    relevant_set = set(map(str, relevant_pmids)) # Ensure ground truth are strings too
    true_positives = len(retrieved_set.intersection(relevant_set))

    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


# --- Orchestration Example with Evaluation ---

if __name__ == "__main__":
    # Define file paths relative to the script location
    script_dir = os.path.dirname(__file__) # Get the directory where the script is located
    # Construct the path assuming training13b.json is one level up from the script's directory
    ground_truth_filepath = os.path.join(script_dir, '..', 'training13b.json')

    # !! REPLACE WITH ACTUAL TASK 13b ENDPOINT !!
    BIOASQ_API_ENDPOINT = "http://bioasq.org:8000/pubmed" # Placeholder

    num_candidates_per_combination = 100
    num_final_results = 10 # This is 'k' for Precision@k, Recall@k

    # Load ground truth data
    debug_mode = True # Set to True to load only the first N items for debugging
    print(f"Loading ground truth from: {ground_truth_filepath}")
    ground_truth_data = load_ground_truth(ground_truth_filepath, debug_mode)

    if not ground_truth_data:
        print("Exiting due to issues loading ground truth data.")
        # Consider using sys.exit(1) after importing sys if this is critical
        exit()

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
        print(f"Extracted keyword combinations: {keyword_combinations_list}") # Optional: uncomment for debugging

        # 2. Call API for Candidates
        all_candidate_articles = []
        seen_article_ids = set() # Use PMIDs here for uniqueness tracking
        # print(f"Fetching top {num_candidates_per_combination} candidates per keyword combination...") # Optional: uncomment for debugging
        for combo in keyword_combinations_list:
            # print(f"  Fetching for combination: '{combo}'") # Optional: uncomment for debugging
            candidate_articles = call_bioasq_api_search(combo, num_candidates_per_combination, BIOASQ_API_ENDPOINT)
            if candidate_articles:
                print(f"    Retrieved {len(candidate_articles)} candidates for '{combo}'.") # Optional: uncomment for debugging
                for article in candidate_articles:
                    # Prioritize PMID from URL, fall back to 'id' if necessary
                    pmid_from_url = extract_pmid_from_url(article.get('url'))
                    pmid_from_id = extract_pmid_from_url(article.get('id')) # Handle case where 'id' might be a URL or PMID
                    pmid = pmid_from_url or pmid_from_id

                    if pmid and pmid not in seen_article_ids:
                        # Store the definitive PMID back into the article dict for ranking
                        article['pmid'] = pmid
                        all_candidate_articles.append(article)
                        seen_article_ids.add(pmid)
            # else:
                 # print(f"    Failed to retrieve candidates for combination '{combo}'.") # Optional: uncomment for debugging


        if all_candidate_articles:
            print(f"Retrieved {len(all_candidate_articles)} unique candidate articles in total.")

            # 5. Rank Candidates Locally using BM25
            print("Ranking candidates locally using BM25...")
            # Ensure rank_articles_bm25 returns PMIDs based on the 'pmid' field we added
            final_top_pmids = rank_articles_bm25(question, all_candidate_articles, top_k=num_final_results)

            # Add pubmed URLs to the final results for clarity
            final_top_pmids_url = [f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" for pmid in final_top_pmids]
            relevant_pmids_url = [f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" for pmid in relevant_pmids] # Ensure ground truth is URLs too

            # 6. Pretty print evaluate Results
            print("-" * 40)
            print(f"Final Top {num_final_results} ranked articles for question: {question}")
            print("-" * 40)
            for i, pmid in enumerate(final_top_pmids_url, start=1):
                print(f"{i}. {pmid}")
            print("-" * 40)
            print(f"Ground truth")
            for i, pmid in enumerate(relevant_pmids_url, start=1):
                print(f"{i}. {pmid}")
            print("-" * 40)
            # Print overlap between retrieved and relevant PMIDs
            print("Overlap between retrieved and relevant PMIDs:")
            overlap = set(final_top_pmids_url).intersection(set(relevant_pmids_url))
            for i, pmid in enumerate(overlap, start=1):
                print(f"{i}. {pmid}")
            print("-" * 40)
            # Print amount of overlap of all retrieved PMIDs with relevant PMIDs
            overlap_all = set(seen_article_ids).intersection(set(relevant_pmids))
            print(f"Overlap of all retrieved PMIDs with relevant PMIDs: {len(overlap_all)}")

            precision, recall, f1 = calculate_precision_recall_f1(final_top_pmids, relevant_pmids)
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
    if questions_with_results > 0:
        avg_precision = total_precision / questions_with_results
        avg_recall = total_recall / questions_with_results
        avg_f1 = total_f1 / questions_with_results
        print("-" * 40)
        print("\n--- Overall Evaluation Results ---")
        print(f"Average Precision@{num_final_results}: {avg_precision:.4f}")
        print(f"Average Recall@{num_final_results}: {avg_recall:.4f}")
        print(f"Average F1-Score@{num_final_results}: {avg_f1:.4f}")
        print(f"Metrics averaged over {questions_with_results} questions (out of {processed_questions} total) for which candidates were retrieved.")
    elif processed_questions > 0:
         print("\n--- Overall Evaluation Results ---")
         print("No candidate articles were retrieved for any question. Cannot calculate average metrics.")
    else:
        print("\nNo questions were processed.")