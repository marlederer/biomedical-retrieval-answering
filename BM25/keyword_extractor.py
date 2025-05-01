import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import combinations

# Download necessary NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except Exception as e:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except Exception as e:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

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
