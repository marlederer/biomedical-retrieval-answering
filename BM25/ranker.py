from rank_bm25 import BM25Okapi
from preprocessing import preprocess_text_for_bm25 # Use relative import
from evaluation import extract_pmid_from_url # Import helper

def rank_articles_bm25(query_text, articles_data, top_k=10):
    """Ranks articles using BM25 on the local set."""
    if not articles_data:
        return []

    # Create the corpus for BM25 - list of tokenized documents
    corpus = []
    doc_ids = [] # Keep track of original PMIDs

    for article in articles_data:
        # Combine title and abstract for richer content
        content = f"{article.get('title', '')} {article.get('abstract', '')}"
        processed_content = preprocess_text_for_bm25(content)
        corpus.append(processed_content)

        # Extract PMID consistently using the helper function
        # Assumes 'id' or 'url' field contains the PMID or URL
        pmid = extract_pmid_from_url(article.get('url')) or extract_pmid_from_url(article.get('id'))
        doc_ids.append(pmid if pmid else article.get('id')) # Fallback to original id if no PMID found


    # Preprocess the query
    tokenized_query = preprocess_text_for_bm25(query_text)

    # Initialize and run BM25
    # You might tune k1 and b parameters based on validation data if available
    bm25 = BM25Okapi(corpus)
    doc_scores = bm25.get_scores(tokenized_query)

    # Combine valid IDs and scores, then sort
    # Filter out entries where doc_id might be None if that's possible and undesirable
    scored_docs = [(doc_id, score) for doc_id, score in zip(doc_ids, doc_scores) if doc_id is not None]
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Final Selection - return only the PMIDs/IDs
    top_ranked_ids = [doc[0] for doc in scored_docs[:top_k]]
    return top_ranked_ids
