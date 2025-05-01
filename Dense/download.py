import nltk
from nltk.tokenize import sent_tokenize

def main():
    print(nltk.data.path)
    try:
        nltk.data.find('tokenizers/punkt')
        # Try to tokenize something small
        sent_tokenize("Test sentence.")
        print("✅ 'punkt' tokenizer is fully available and working.")
    except Exception as e:
        print("❌ 'punkt' tokenizer is not available. Attempting to download...")
        print("🔄 Downloading 'punkt' tokenizer...")
        nltk.download('punkt', download_dir=r"C:\\Users\\phili\\AppData\\Roaming\\nltk_data")
        nltk.download('punkt_tab')
if __name__ == "__main__":
    main()