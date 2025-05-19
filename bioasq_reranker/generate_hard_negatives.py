# generate_hard_negatives.py
import argparse
import json
import logging
from tqdm import tqdm
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Assuming data_loader.py is in the same directory or accessible in PYTHONPATH
from data_loader import load_bioasq_json, build_passage_corpus_from_bioasq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_hard_negatives_tfidf(questions_data, passage_corpus, top_k_retrieval=20, num_hard_negatives_per_query=5):
    """
    Generates hard negatives using a simple TF-IDF retriever.

    For each question:
    1. Retrieves top_k_retrieval passages from the passage_corpus using TF-IDF similarity.
    2. Filters out known positive passages for that question.
    3. Selects up to num_hard_negatives_per_query from the remaining highly-ranked (but incorrect) passages.

    Args:
        questions_data (list): List of question dicts from BioASQ JSON.
        passage_corpus (list): List of all unique passage texts.
        top_k_retrieval (int): How many top passages to retrieve initially with TF-IDF.
        num_hard_negatives_per_query (int): Max number of hard negatives to save per query.

    Returns:
        dict: A dictionary mapping question_id to a list of hard negative passage texts.
              e.g., {"question_id1": ["hard_neg_text1", "hard_neg_text2"], ...}
    """
    if not passage_corpus:
        logger.error("Passage corpus is empty. Cannot generate TF-IDF vectors or hard negatives.")
        return {}
    if not questions_data:
        logger.error("Questions data is empty. Cannot generate hard negatives.")
        return {}

    logger.info("Initializing TF-IDF Vectorizer...")
    try:
        vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.9) # Tune params as needed
        corpus_tfidf_matrix = vectorizer.fit_transform(passage_corpus)
        logger.info(f"TF-IDF matrix shape for corpus: {corpus_tfidf_matrix.shape}")
    except ValueError as e:
        logger.error(f"Error initializing TF-IDF or fitting corpus (e.g. empty vocabulary): {e}. "
                     "This might happen if passage_corpus is too small or homogeneous after stopword removal.")
        logger.info("Trying TF-IDF with min_df=1 as a fallback...")
        try:
            vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
            corpus_tfidf_matrix = vectorizer.fit_transform(passage_corpus)
            logger.info(f"Fallback TF-IDF matrix shape for corpus: {corpus_tfidf_matrix.shape}")
        except ValueError as e_fallback:
            logger.error(f"Fallback TF-IDF also failed: {e_fallback}. Cannot proceed.")
            return {}


    hard_negatives_map = {}

    for q_entry in tqdm(questions_data, desc="Generating Hard Negatives"):
        query_text = q_entry.get('body', '').strip()
        q_id = q_entry.get('id')

        if not query_text or not q_id:
            logger.warning(f"Skipping question with missing body or ID: {q_entry}")
            continue

        positive_passages_texts = set()
        if 'snippets' in q_entry:
            for snippet in q_entry['snippets']:
                passage_text = snippet.get('text', '').strip()
                if passage_text:
                    positive_passages_texts.add(passage_text)

        if not positive_passages_texts:
            # logger.debug(f"No positive snippets for question ID {q_id}, cannot define hard negatives relative to them.")
            continue
        
        try:
            query_tfidf_vector = vectorizer.transform([query_text])
        except ValueError:
            logger.warning(f"Could not transform query (possibly empty after processing): '{query_text}'. Skipping for q_id {q_id}.")
            continue


        # Calculate cosine similarities
        similarities = cosine_similarity(query_tfidf_vector, corpus_tfidf_matrix).flatten()

        # Get top_k_retrieval indices, sorted by similarity
        # Adding a small epsilon to handle cases where many similarities are identical (e.g., zero)
        # This ensures that argsort behaves deterministically if scores are tied.
        # sorted_indices = np.argsort(similarities + np.random.rand(len(similarities)) * 1e-9)[::-1]
        sorted_indices = np.argsort(similarities)[::-1]


        current_hard_negatives = []
        for idx in sorted_indices[:top_k_retrieval]:
            candidate_passage = passage_corpus[idx]
            # Check if it's NOT a known positive for this query
            if candidate_passage not in positive_passages_texts:
                current_hard_negatives.append(candidate_passage)
            if len(current_hard_negatives) >= num_hard_negatives_per_query:
                break
        
        if current_hard_negatives:
            hard_negatives_map[q_id] = current_hard_negatives
            # logger.debug(f"Found {len(current_hard_negatives)} hard negatives for q_id {q_id}")

    logger.info(f"Generated hard negatives for {len(hard_negatives_map)} questions.")
    return hard_negatives_map


def main():
    parser = argparse.ArgumentParser(description="Generate hard negatives for BioASQ re-ranker training.")
    parser.add_argument("--bioasq_file", type=str, required=True, help="Path to BioASQ training JSON file (e.g., trainingSet.json)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated hard negatives JSON file.")
    parser.add_argument("--top_k_retrieval", type=int, default=50, help="Number of passages to retrieve with TF-IDF per query.")
    parser.add_argument("--num_hard_negatives_per_query", type=int, default=5, help="Max number of hard negatives to store per query.")
    parser.add_argument("--max_questions_process", type=int, default=None, help="Max questions from BioASQ to process (for quick test/debug)")


    args = parser.parse_args()

    logger.info(f"Loading BioASQ data from: {args.bioasq_file}")
    raw_questions_data_full = load_bioasq_json(args.bioasq_file)

    if not raw_questions_data_full:
        logger.error("No questions data loaded. Exiting.")
        return

    if args.max_questions_process is not None and args.max_questions_process < len(raw_questions_data_full):
        logger.info(f"Processing a subset of {args.max_questions_process} questions.")
        questions_to_process = raw_questions_data_full[:args.max_questions_process]
    else:
        questions_to_process = raw_questions_data_full

    # Build a corpus from all snippets in the dataset (can be adapted to use a larger external corpus)
    passage_corpus = build_passage_corpus_from_bioasq(raw_questions_data_full) # Use full data for corpus richness
    if not passage_corpus:
        logger.error("Failed to build passage corpus. Cannot generate hard negatives. Exiting.")
        return

    hard_negatives = generate_hard_negatives_tfidf(
        questions_to_process,
        passage_corpus,
        args.top_k_retrieval,
        args.num_hard_negatives_per_query
    )

    if hard_negatives:
        logger.info(f"Saving {len(hard_negatives)} hard negative entries to {args.output_file}")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(hard_negatives, f, indent=4)
        logger.info("Hard negatives generation complete.")
    else:
        logger.warning("No hard negatives were generated. Output file will not be created or will be empty.")

if __name__ == "__main__":
    # Create a dummy BioASQ file if it doesn't exist for quick testing
    dummy_bioasq_path = "dummy_bioasq_for_hn.json"
    if not os.path.exists(dummy_bioasq_path) and \
       not any(arg.startswith("--bioasq_file") and os.path.exists(arg.split("=")[1] if "=" in arg else arg) for arg in os.sys.argv):
        # Create dummy only if no bioasq_file is provided or if default path doesn't exist.
        # This logic is a bit complex for a simple dummy file creation.
        # Let's simplify: if a specific --bioasq_file is given and exists, use it. Otherwise, try dummy.
        create_dummy = True
        for i, arg_val in enumerate(os.sys.argv):
            if arg_val == "--bioasq_file" and i + 1 < len(os.sys.argv):
                if os.path.exists(os.sys.argv[i+1]):
                    create_dummy = False
                break
        
        if create_dummy:
            logger.info(f"Creating dummy BioASQ data at {dummy_bioasq_path} for demonstration.")
            dummy_bioasq_content = {
                "questions": [
                    {"id": f"q{i}", "body": f"What is item {i} using specific keywords like genomics and therapy?", "snippets": [{"text": f"Item {i} is a test entity about genomics."}, {"text": f"More about item {i} therapy here."}]} for i in range(1, 21)
                ] + [
                    {"id": f"q_alt{i}", "body": f"Alternative query for product {i} focusing on synthesis and pathways?", "snippets": [{"text": f"Product {i} involves complex synthesis."}, {"text": f"Understanding {i} pathways is key."}]} for i in range(1, 11)
                ]
            }
            with open(dummy_bioasq_path, 'w') as f:
                json.dump(dummy_bioasq_content, f)
            # Adjust argv to use this dummy file if no other is specified
            if "--bioasq_file" not in os.sys.argv:
                os.sys.argv.extend(["--bioasq_file", dummy_bioasq_path])
            if "--output_file" not in os.sys.argv:
                 os.sys.argv.extend(["--output_file", "data/hard_negatives_dummy.json"])


    main()

    # Clean up dummy file if created by this script's logic
    if "dummy_bioasq_for_hn.json" in os.sys.argv and os.path.exists(dummy_bioasq_path):
         if os.path.exists(dummy_bioasq_path):
            os.remove(dummy_bioasq_path)
            logger.info(f"Removed dummy data file: {dummy_bioasq_path}")
         dummy_output = "data/hard_negatives_dummy.json"
         if os.path.exists(dummy_output) and "hard_negatives_dummy.json" in os.sys.argv:
             os.remove(dummy_output)
             logger.info(f"Removed dummy output file: {dummy_output}")