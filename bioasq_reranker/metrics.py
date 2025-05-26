'''
Evaluation metrics for reranking.

Includes Mean Reciprocal Rank (MRR).
'''

def calculate_mrr(ranked_lists, relevant_docs_map, k=10):
    '''
    Calculates Mean Reciprocal Rank @ k (MRR@k).

    Args:
        ranked_lists (dict): A dictionary where keys are query_ids and values are 
                             lists of document_ids, sorted by relevance score (highest first).
                             Example: {'q1': ['docA', 'docB', 'docC'], 'q2': ['docX', 'docY']}
        relevant_docs_map (dict): A dictionary where keys are query_ids and values are 
                                  lists or sets of truly relevant document_ids for that query.
                                  Example: {'q1': ['docB'], 'q2': ['docZ']}
        k (int): The cutoff for MRR calculation (e.g., MRR@10).

    Returns:
        float: The Mean Reciprocal Rank @ k.
        dict: Reciprocal ranks for each query.
    '''
    if not ranked_lists:
        return 0.0, {}

    reciprocal_ranks = {}
    total_rr = 0.0

    for query_id, ranked_docs in ranked_lists.items():
        rr_q = 0.0
        true_relevant_docs = relevant_docs_map.get(query_id, set()) # Use set for efficient lookup
        if not isinstance(true_relevant_docs, (list, set)):
            print(f"Warning: Relevant docs for query {query_id} is not a list or set. Skipping.")
            true_relevant_docs = set()
        
        # Convert to set if it's a list for faster lookups
        if isinstance(true_relevant_docs, list):
            true_relevant_docs = set(true_relevant_docs)

        for rank, doc_id in enumerate(ranked_docs[:k]): # Consider only top k documents
            if doc_id in true_relevant_docs:
                rr_q = 1.0 / (rank + 1)
                break # Found the first relevant document
        
        reciprocal_ranks[query_id] = rr_q
        total_rr += rr_q

    mean_rr = total_rr / len(ranked_lists) if ranked_lists else 0.0
    return mean_rr, reciprocal_ranks


if __name__ == '__main__':
    # Example Usage for MRR
    example_ranked_lists = {
        'q1': ['docA', 'docB_relevant', 'docC', 'docD', 'docE_relevant'], # First relevant at rank 2
        'q2': ['docX', 'docY', 'docZ'], # No relevant doc in top k (assuming k=3 for this comment)
        'q3': ['docP_relevant', 'docQ', 'docR'] # First relevant at rank 1
    }
    example_relevant_docs_map = {
        'q1': ['docB_relevant', 'docE_relevant'],
        'q2': ['docW_relevant'], # This relevant doc is not in q2's ranked list
        'q3': ['docP_relevant']
    }
    k_val = 3

    mrr_at_k, rrs_per_query = calculate_mrr(example_ranked_lists, example_relevant_docs_map, k=k_val)
    print(f"Example MRR@{k_val}: {mrr_at_k:.4f}")
    print(f"Reciprocal Ranks per query: {rrs_per_query}")

    # Expected for q1: 1/2 = 0.5 (docB_relevant is at rank 2)
    # Expected for q2: 0.0 (docW_relevant is not in top 3 of q2's list)
    # Expected for q3: 1/1 = 1.0 (docP_relevant is at rank 1)
    # Expected MRR@3 = (0.5 + 0.0 + 1.0) / 3 = 1.5 / 3 = 0.5

    assert abs(rrs_per_query['q1'] - 0.5) < 1e-6
    assert abs(rrs_per_query['q2'] - 0.0) < 1e-6
    assert abs(rrs_per_query['q3'] - 1.0) < 1e-6
    assert abs(mrr_at_k - 0.5) < 1e-6
    print("MRR example assertions passed.")

    # Test with empty ranked list
    mrr_empty, _ = calculate_mrr({}, example_relevant_docs_map, k=k_val)
    print(f"MRR for empty ranked_lists: {mrr_empty}")
    assert mrr_empty == 0.0

    # Test with query not in relevant_docs_map
    mrr_q_not_in_rel, rrs_q_not_in_rel = calculate_mrr({'q4':['d1','d2']}, example_relevant_docs_map, k=k_val)
    print(f"MRR for query not in relevant_map: {mrr_q_not_in_rel}, RRs: {rrs_q_not_in_rel}")
    assert rrs_q_not_in_rel['q4'] == 0.0
    assert mrr_q_not_in_rel == 0.0

    print("All metric tests passed.")