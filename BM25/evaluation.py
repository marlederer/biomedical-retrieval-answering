import json
import os

# Function to extract PMID from a PubMed URL or just a PMID string
def extract_pmid_from_url(identifier):
    """Extracts the PubMed ID (PMID) from a PubMed URL or returns if it's already a PMID."""
    if isinstance(identifier, str):
        if 'pubmed/' in identifier:
            # Handle potential query parameters or fragments
            base_url = identifier.split('?')[0].split('#')[0]
            potential_pmid = base_url.split('/')[-1]
            if potential_pmid.isdigit():
                return potential_pmid
        elif identifier.isdigit(): # Check if the identifier itself is a PMID
            return identifier
    return None # Return None if it's not a valid URL or PMID string

# Function to load ground truth data with DEBUG mode to only load first N items
def load_ground_truth(filepath, debug_mode=False, debug_limit=5):
    """Loads questions and their relevant document PMIDs from a JSON file."""
    ground_truth = {}
    limit_msg = f"Loading only the first {debug_limit} items." if debug_mode else "Loading all items."
    print(f"DEBUG MODE: {debug_mode}. {limit_msg} from the ground truth file: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions_data = data.get('questions', [])
            items_to_process = questions_data[:debug_limit] if debug_mode else questions_data

            for item in items_to_process:
                question_body = item.get('body')
                doc_urls = item.get('documents', [])
                if question_body:
                    # Extract PMIDs and store as a set for efficient lookup
                    pmids = {extract_pmid_from_url(url) for url in doc_urls if extract_pmid_from_url(url)}
                    if pmids: # Only add if there are valid PMIDs
                        ground_truth[question_body] = pmids
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {filepath}")
        return None # Indicate failure clearly
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None # Indicate failure clearly
    except Exception as e:
        print(f"An unexpected error occurred while loading ground truth: {e}")
        return None # Indicate failure clearly

    if not ground_truth:
        print("Warning: No valid ground truth data loaded.")

    return ground_truth

# Function to calculate precision, recall, and F1-score
def calculate_precision_recall_f1(retrieved_pmids, relevant_pmids):
    """Calculates Precision@k, Recall@k, and F1@k."""
    if not relevant_pmids: # Avoid division by zero if no relevant documents exist
        # If there are no relevant docs, precision/recall/F1 are arguably undefined or 0.
        # Let's return 0, but this interpretation might vary.
        return 0.0, 0.0, 0.0

    # Ensure both sets contain strings for comparison
    retrieved_set = set(map(str, retrieved_pmids))
    relevant_set = set(map(str, relevant_pmids)) # Ensure ground truth are strings too

    true_positives = len(retrieved_set.intersection(relevant_set))

    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    # Recall is based on the total number of relevant documents
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def evaluate_predictions(predictions_filepath, ground_truth_filepath, debug_mode=False, debug_limit=5):
    with open(predictions_filepath, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    ground_truth_data = load_ground_truth(ground_truth_filepath, debug_mode=debug_mode, debug_limit=debug_limit)

    if not ground_truth_data:
        print("Ground truth could not be loaded. Evaluation aborted.")
        return

    total_doc_precision = total_doc_recall = total_doc_f1 = 0.0
    total_snip_precision = total_snip_recall = total_snip_f1 = 0.0
    evaluated_count = 0

    for item in predictions_data.get('questions', []):
        question_body = item.get('body')
        predicted_urls = item.get('documents', [])
        predicted_pmids = {extract_pmid_from_url(url) for url in predicted_urls if extract_pmid_from_url(url)}
        predicted_snips = set(s["text"] for s in item.get("snippets", []) if "text" in s)

        if question_body in ground_truth_data:
            gt = ground_truth_data[question_body]
            relevant_pmids = gt["documents"]
            relevant_snips = gt["snippets"]

            doc_precision, doc_recall, doc_f1 = calculate_precision_recall_f1(predicted_pmids, relevant_pmids)
            snip_precision, snip_recall, snip_f1 = calculate_precision_recall_f1(predicted_snips, relevant_snips)

            print(f"Q: {question_body[:60]}...")
            print(f"Documents - Precision: {doc_precision:.4f}, Recall: {doc_recall:.4f}, F1: {doc_f1:.4f}")
            print(f"Snippets  - Precision: {snip_precision:.4f}, Recall: {snip_recall:.4f}, F1: {snip_f1:.4f}\n")

            total_doc_precision += doc_precision
            total_doc_recall += doc_recall
            total_doc_f1 += doc_f1
            total_snip_precision += snip_precision
            total_snip_recall += snip_recall
            total_snip_f1 += snip_f1
            evaluated_count += 1

    if evaluated_count > 0:
        avg_doc_precision = total_doc_precision / evaluated_count
        avg_doc_recall = total_doc_recall / evaluated_count
        avg_doc_f1 = total_doc_f1 / evaluated_count

        avg_snip_precision = total_snip_precision / evaluated_count
        avg_snip_recall = total_snip_recall / evaluated_count
        avg_snip_f1 = total_snip_f1 / evaluated_count

        print("=== Overall Evaluation ===")
        print(f"Document Retrieval - Avg Precision: {avg_doc_precision:.4f}, Recall: {avg_doc_recall:.4f}, F1: {avg_doc_f1:.4f}")
        print(f"Snippet Retrieval  - Avg Precision: {avg_snip_precision:.4f}, Recall: {avg_snip_recall:.4f}, F1: {avg_snip_f1:.4f}")
    else:
        print("No questions matched for evaluation.")
