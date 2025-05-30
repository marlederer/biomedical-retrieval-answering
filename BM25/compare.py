'''
Compares ground truth document relevance with predicted document relevance from BioASQ-formatted JSON files.
Calculates Accuracy, Precision, Recall, F1-score (macro-averaged), and MRR@k.
'''
import json
import os

def extract_pmid_from_url(url: str) -> str | None:
    """
    Extracts the PubMed ID (PMID) from a PubMed URL.
    Example: "http://www.ncbi.nlm.nih.gov/pubmed/34512906" -> "34512906"
    """
    if isinstance(url, str) and "pubmed" in url:
        parts = url.strip().split('/')
        if parts:
            # Handle potential trailing slashes or query parameters if any
            for part in reversed(parts):
                if part.isdigit():
                    return part
    return None

def load_questions_data(filepath: str) -> dict[str, dict[str, list[str]]]:
    """
    Loads question data from a JSON file and organizes it by question ID.
    Returns a dictionary where keys are question IDs and values are dicts
    containing 'documents' (list of PMIDs) and 'snippets' (list of snippet texts).
    Order is preserved.
    """
    data_map = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for question_obj in data.get('questions', []):
            q_id = question_obj.get('id')
            if not q_id: # Skip questions without an ID
                # print(f"Skipping question without ID: {question_obj}")
                continue

            # Process documents
            doc_urls = question_obj.get('documents', []) # Default to empty list
            doc_pmids = []
            if isinstance(doc_urls, list):
                for url in doc_urls:
                    pmid = extract_pmid_from_url(url)
                    if pmid:
                        doc_pmids.append(pmid)
            
            # Process snippets
            raw_snippets = question_obj.get('snippets', []) # Default to empty list
            processed_snippets = []
            if isinstance(raw_snippets, list):
                for s_item in raw_snippets:
                    if isinstance(s_item, dict) and 'text' in s_item: # Check for dict and 'text' key
                        s_text = str(s_item['text']).strip()
                        if s_text: # Only add non-empty snippets
                            processed_snippets.append(s_text)
                    elif isinstance(s_item, str): # Fallback for simple string snippets (e.g. from ground truth if different)
                        s_text = s_item.strip()
                        if s_text:
                            processed_snippets.append(s_text)

            # Only add to map if there's an ID. Documents/snippets can be empty lists.
            data_map[q_id] = {"documents": doc_pmids, "snippets": processed_snippets}

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return {}
    return data_map


def calculate_metrics(ground_truth_data: dict[str, set[str]], 
                      predicted_data: dict[str, list[str]], 
                      k_mrr: int = 10) -> dict[str, float]:
    """
    Calculates accuracy, precision, recall, F1-score, and MRR@k.
    - ground_truth_data: {q_id: {set of true PMIDs}}
    - predicted_data: {q_id: [list of predicted PMIDs (ordered)]}
    - k_mrr: The 'k' for MRR@k calculation.
    """
    total_precision_q = 0.0
    total_recall_q = 0.0
    total_f1_q = 0.0
    total_reciprocal_rank_q = 0.0
    accurate_questions_count = 0
    num_evaluated_questions = 0 # Will count common questions

    # Determine common question IDs for evaluation
    gt_ids = set(ground_truth_data.keys())
    pred_ids = set(predicted_data.keys())
    common_q_ids = gt_ids.intersection(pred_ids)

    if not common_q_ids:
        print("Warning: No common question IDs found between ground truth and predicted data. Cannot calculate metrics.")
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_score_macro": 0.0,
            f"mrr@{k_mrr}": 0.0,
            "evaluated_questions": 0
        }

    for q_id in common_q_ids: # Iterate over common question IDs
        num_evaluated_questions += 1
        
        true_pmids_set = ground_truth_data[q_id] # q_id is in ground_truth_data
        predicted_pmids_list = predicted_data[q_id] # q_id is in predicted_data
        
        # Consider only the top 10 predictions for comparison
        predicted_pmids_list = predicted_pmids_list[:10]
        
        retrieved_pmids_set = set(predicted_pmids_list)
        
        # True Positives
        tp_set = true_pmids_set.intersection(retrieved_pmids_set)
        tp = len(tp_set)

        # Precision, Recall, F1 for the current question
        precision_q = tp / len(retrieved_pmids_set) if len(retrieved_pmids_set) > 0 else 0.0
        
        if not true_pmids_set:
            # If ground truth is empty: recall is 1 if predictions are also empty, else 0.
            recall_q = 1.0 if not retrieved_pmids_set else 0.0
        else:
            recall_q = tp / len(true_pmids_set) # len(true_pmids_set) > 0 here
        
        f1_q = (2 * precision_q * recall_q) / (precision_q + recall_q) if (precision_q + recall_q) > 0 else 0.0
        
        total_precision_q += precision_q
        total_recall_q += recall_q
        total_f1_q += f1_q

        # Accuracy definition:
        # 1. At least one relevant doc retrieved (tp > 0)
        # OR 2. GT is empty and predictions are empty.
        current_q_accurate = False
        if tp > 0:
            current_q_accurate = True
        elif not true_pmids_set and not retrieved_pmids_set:
            current_q_accurate = True
        
        if current_q_accurate:
            accurate_questions_count += 1

        # MRR@k for the current question
        reciprocal_rank_q = 0.0
        for i, pmid in enumerate(predicted_pmids_list[:k_mrr]):
            if pmid in true_pmids_set:
                reciprocal_rank_q = 1.0 / (i + 1)
                break
        total_reciprocal_rank_q += reciprocal_rank_q

    if num_evaluated_questions == 0:
        # This case should ideally be caught by the initial check of ground_truth_data
        return {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, 
                "f1_score_macro": 0.0, f"mrr@{k_mrr}": 0.0, "evaluated_questions": 0}

    accuracy = accurate_questions_count / num_evaluated_questions
    avg_precision_macro = total_precision_q / num_evaluated_questions
    avg_recall_macro = total_recall_q / num_evaluated_questions
    avg_f1_macro = total_f1_q / num_evaluated_questions
    
    mrr_at_k = total_reciprocal_rank_q / num_evaluated_questions

    return {
        "accuracy": accuracy,
        "precision_macro": avg_precision_macro,
        "recall_macro": avg_recall_macro,
        "f1_score_macro": avg_f1_macro,
        f"mrr@{k_mrr}": mrr_at_k,
        "evaluated_questions": num_evaluated_questions
    }

# --- Jaccard Similarity Functions ---
def preprocess_snippet_for_jaccard(text: str) -> set[str]:
    """Lowercase and split text into a set of words."""
    return set(text.lower().split())

def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Calculates Jaccard similarity between two sets of words."""
    if not isinstance(set1, set) or not isinstance(set2, set): # Ensure inputs are sets
        return 0.0
    if not set1 and not set2: # Both empty
        return 1.0
    if not set1 or not set2: # One empty
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def calculate_snippet_metrics_jaccard(
    ground_truth_snippets_map: dict[str, set[str]], 
    predicted_snippets_map: dict[str, list[str]], 
    k_mrr: int = 10, 
    jaccard_threshold: float = 0.5
) -> dict[str, float]:
    """
    Calculates snippet metrics based on Jaccard similarity.
    - ground_truth_snippets_map: {q_id: {set of true snippet strings}}
    - predicted_snippets_map: {q_id: [list of predicted snippet strings (ordered)]}
    - k_mrr: The 'k' for MRR@k calculation.
    - jaccard_threshold: Minimum Jaccard similarity to consider a match.
    """
    total_precision_q = 0.0
    total_recall_q = 0.0
    total_f1_q = 0.0
    total_reciprocal_rank_q = 0.0
    accurate_questions_count = 0
    num_evaluated_questions = 0

    gt_ids = set(ground_truth_snippets_map.keys())
    pred_ids = set(predicted_snippets_map.keys())
    common_q_ids = gt_ids.intersection(pred_ids)

    if not common_q_ids:
        return {
            "accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0,
            "f1_score_macro": 0.0, f"mrr@{k_mrr}": 0.0, "evaluated_questions": 0
        }

    for q_id in common_q_ids:
        num_evaluated_questions += 1

        true_snippet_texts = ground_truth_snippets_map.get(q_id, set())
        # Already sliced to top 10 in previous step if this function is called after exact match processing
        # or needs to be sliced if called directly with full predicted_snippets_map
        pred_snippet_texts_top_k = predicted_snippets_map.get(q_id, [])[:10]


        processed_true_snippets = [preprocess_snippet_for_jaccard(s) for s in true_snippet_texts]
        processed_pred_snippets = [preprocess_snippet_for_jaccard(s) for s in pred_snippet_texts_top_k]

        # --- Precision related ---
        tp_for_precision = 0
        if processed_pred_snippets:
            for pred_snip_set in processed_pred_snippets:
                max_j_with_any_true = 0.0
                if processed_true_snippets:
                    for true_snip_set in processed_true_snippets:
                        max_j_with_any_true = max(max_j_with_any_true, jaccard_similarity(pred_snip_set, true_snip_set))
                if max_j_with_any_true >= jaccard_threshold:
                    tp_for_precision += 1
        
        if not processed_pred_snippets: # No predictions
            precision_q = 1.0 if not processed_true_snippets else 0.0
        else: # There are predictions
            precision_q = tp_for_precision / len(processed_pred_snippets)

        # --- Recall related ---
        gt_snippets_found_count = 0
        if processed_true_snippets:
            for true_snip_set in processed_true_snippets:
                found_this_gt = False
                if processed_pred_snippets:
                    for pred_snip_set in processed_pred_snippets:
                        if jaccard_similarity(true_snip_set, pred_snip_set) >= jaccard_threshold:
                            found_this_gt = True
                            break
                if found_this_gt:
                    gt_snippets_found_count += 1
        
        if not processed_true_snippets: # No true snippets
            recall_q = 1.0 if not processed_pred_snippets else 0.0
        else: # There are true snippets
            recall_q = gt_snippets_found_count / len(processed_true_snippets)
            
        # --- F1 ---
        f1_q = (2 * precision_q * recall_q) / (precision_q + recall_q) if (precision_q + recall_q) > 0 else 0.0
        
        total_precision_q += precision_q
        total_recall_q += recall_q
        total_f1_q += f1_q

        # --- Accuracy ---
        current_q_accurate = False
        if tp_for_precision > 0: # At least one predicted snippet matched a GT snippet
            current_q_accurate = True
        elif not processed_true_snippets and not processed_pred_snippets: # Both GT and pred are empty
            current_q_accurate = True
        
        if current_q_accurate:
            accurate_questions_count += 1

        # --- MRR@k ---
        reciprocal_rank_q = 0.0
        # Note: processed_pred_snippets is already top_k (or fewer)
        for i, pred_snip_set in enumerate(processed_pred_snippets): # Iterate up to k_mrr (which is len of this list)
            if i >= k_mrr: break # Ensure we don't exceed k_mrr if list was longer for some reason
            found_match_for_mrr = False
            if processed_true_snippets:
                for true_snip_set in processed_true_snippets:
                    if jaccard_similarity(pred_snip_set, true_snip_set) >= jaccard_threshold:
                        found_match_for_mrr = True
                        break
            if found_match_for_mrr:
                reciprocal_rank_q = 1.0 / (i + 1)
                break
        total_reciprocal_rank_q += reciprocal_rank_q

    if num_evaluated_questions == 0:
        return {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, 
                "f1_score_macro": 0.0, f"mrr@{k_mrr}": 0.0, "evaluated_questions": 0}

    return {
        "accuracy": accurate_questions_count / num_evaluated_questions,
        "precision_macro": total_precision_q / num_evaluated_questions,
        "recall_macro": total_recall_q / num_evaluated_questions,
        "f1_score_macro": total_f1_q / num_evaluated_questions,
        f"mrr@{k_mrr}": total_reciprocal_rank_q / num_evaluated_questions,
        "evaluated_questions": num_evaluated_questions
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to training13b.json (ground truth)
    ground_truth_filepath = os.path.abspath(os.path.join(script_dir, '..', 'data/training13b.json'))
    # Path to bioasq_output_reranker.json (predictions)
    predicted_filepath = os.path.join(script_dir, '../bioasq_reranker/data/dense.json')

    print(f"Loading ground truth from: {ground_truth_filepath}")
    raw_ground_truth_data_full = load_questions_data(ground_truth_filepath)
    
    ground_truth_docs_map = {
        qid: set(data.get('documents', []))
        for qid, data in raw_ground_truth_data_full.items()
    }
    ground_truth_snippets_map = {
        qid: set(data.get('snippets', [])) # Use set for GT snippets for direct comparison
        for qid, data in raw_ground_truth_data_full.items()
    }
    
    print(f"Loading predicted data from: {predicted_filepath}")
    raw_predicted_data_full = load_questions_data(predicted_filepath)

    predicted_docs_map = {
        qid: data.get('documents', []) # List for order (MRR) and top-k slicing
        for qid, data in raw_predicted_data_full.items()
    }
    predicted_snippets_map = {
        qid: data.get('snippets', []) # List for order (MRR) and top-k slicing
        for qid, data in raw_predicted_data_full.items()
    }

    if not ground_truth_docs_map and not ground_truth_snippets_map:
        print("No ground truth data loaded or all entries lacked IDs/data. Exiting.")
        return

    k_for_mrr = 10 

    print("\\n--- Document Evaluation Metrics ---")
    # Check if there's anything to evaluate for documents
    if not any(ground_truth_docs_map.values()) and not any(predicted_docs_map.values()):
         print("No document data found in ground truth or predictions to evaluate.")
    else:
        doc_metrics = calculate_metrics(ground_truth_docs_map, predicted_docs_map, k_mrr=k_for_mrr)
        print(f"Evaluated on {doc_metrics['evaluated_questions']} questions common for documents.")
        print(f"Accuracy: {doc_metrics['accuracy']:.4f}")
        print(f"Precision (macro): {doc_metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {doc_metrics['recall_macro']:.4f}")
        print(f"F1-score (macro): {doc_metrics['f1_score_macro']:.4f}")
        print(f"MRR@{k_for_mrr}: {doc_metrics[f'mrr@{k_for_mrr}']:.4f}")

    print("\\n--- Snippet Evaluation Metrics (Exact Match) ---")
    # Check if there's anything to evaluate for snippets
    if not any(ground_truth_snippets_map.values()) and not any(predicted_snippets_map.values()):
         print("No snippet data found in ground truth or predictions to evaluate for exact match.")
    else:
        # For exact match, predicted_snippets_map still contains lists of strings.
        # calculate_metrics will handle the set conversion for its internal logic.
        exact_snippet_metrics = calculate_metrics(
            ground_truth_snippets_map, # {qid: set of GT snippet strings}
            predicted_snippets_map,    # {qid: list of predicted snippet strings}
            k_mrr=k_for_mrr
        )
        print(f"Evaluated on {exact_snippet_metrics['evaluated_questions']} questions common for snippets (exact match).")
        print(f"Accuracy: {exact_snippet_metrics['accuracy']:.4f}")
        print(f"Precision (macro): {exact_snippet_metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {exact_snippet_metrics['recall_macro']:.4f}")
        print(f"F1-score (macro): {exact_snippet_metrics['f1_score_macro']:.4f}")
        print(f"MRR@{k_for_mrr}: {exact_snippet_metrics[f'mrr@{k_for_mrr}']:.4f}")

    print("\\n--- Snippet Evaluation Metrics (Jaccard Similarity @ 0.5) ---")
    if not any(ground_truth_snippets_map.values()) and not any(predicted_snippets_map.values()):
        print("No snippet data found in ground truth or predictions to evaluate for Jaccard match.")
    else:
        jaccard_snippet_metrics = calculate_snippet_metrics_jaccard(
            ground_truth_snippets_map, # {qid: set of GT snippet strings}
            predicted_snippets_map,    # {qid: list of predicted snippet strings}
            k_mrr=k_for_mrr,
            jaccard_threshold=0.5
        )
        print(f"Evaluated on {jaccard_snippet_metrics['evaluated_questions']} questions common for snippets (Jaccard sim >= 0.5).")
        print(f"Accuracy: {jaccard_snippet_metrics['accuracy']:.4f}")
        print(f"Precision (macro): {jaccard_snippet_metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {jaccard_snippet_metrics['recall_macro']:.4f}")
        print(f"F1-score (macro): {jaccard_snippet_metrics['f1_score_macro']:.4f}")
        print(f"MRR@{k_for_mrr}: {jaccard_snippet_metrics[f'mrr@{k_for_mrr}']:.4f}")

if __name__ == "__main__":
    main()
