import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words('english'))

def preprocess_text_for_bm25(text):
    """Basic preprocessing: tokenize, lower, remove stops & non-alphanum."""
    if not isinstance(text, str): # Handle potential None or non-string abstracts
        return []
    tokens = word_tokenize(text.lower())
    processed_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return processed_tokens
