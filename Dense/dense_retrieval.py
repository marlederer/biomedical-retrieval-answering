# dense_retrieval.py
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import spacy

nlp = spacy.load("en_core_web_sm")
# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models globally (you can move to init if you want faster runs)
question_encoder = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
context_encoder = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
question_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
context_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

def encode_query(query):
    inputs = question_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        embedding = question_encoder(**inputs).last_hidden_state[:, 0, :]
    return embedding

def encode_contexts(contexts, batch_size=32):
    safe_contexts = []

    for idx, c in enumerate(contexts):
        if isinstance(c, str):
            safe_contexts.append(c.strip())
        elif c is None:
            print(f"[DEBUG] Context at index {idx} is None. Replacing with empty string.")
            safe_contexts.append("")
        else:
            try:
                coerced = str(c).strip()
                safe_contexts.append(coerced)
            except Exception as e:
                print(f"[DEBUG] Failed to convert context at index {idx}: {c!r}. Error: {e}. Replacing with empty string.")
                safe_contexts.append("")

    contexts = safe_contexts
    all_embeddings = []
    context_encoder.eval()
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i+batch_size]
            inputs = context_tokenizer(
                batch_contexts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)
            
            batch_embeddings = context_encoder(**inputs).last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
            
            # Move to CPU to save GPU memory if needed
            all_embeddings.append(batch_embeddings.cpu())
    # After batching, stack all embeddings into one big tensor
    return torch.vstack(all_embeddings)  # (num_contexts, hidden_dim)

def rank_with_dense_retrieval(query_embedding, context_embeddings, top_k=10):
    # Calculate similarities
    scores = torch.matmul(context_embeddings.to(device), query_embedding.to(device).squeeze(0))
    top_k_indices = torch.topk(scores, top_k).indices
    return top_k_indices.tolist()

def select_top_k_snippets_from_articles(query_embedding, articles, top_k=10):
    candidate_snippets = []
    snippet_metadata = []

    for art in articles:
        if not art or not isinstance(art, dict):
            print("[WARNING] Skipping invalid article:", art)
            continue
        url = art.get('url')
        title = art.get('title') or ''
        title = title.strip()

        abstract = art.get('abstract') or ''
        abstract = abstract.strip()

        if not url:
            continue

        # 1. Add title as a separate snippet (if present)
        if title:
            candidate_snippets.append(title)
            snippet_metadata.append({
                "text": title,
                "document": url,
                "offsetInBeginSection": 0,
                "offsetInEndSection": len(title),
                "beginSection": "title",
                "endSection": "title"
            })

        # 2. Split abstract into sentence snippets (if present)
        if abstract:
            doc = nlp(abstract)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if sent_text:
                    candidate_snippets.append(sent_text)
                    snippet_metadata.append({
                        "text": sent_text,
                        "document": url,
                        "offsetInBeginSection": sent.start_char,
                        "offsetInEndSection": sent.end_char,
                        "beginSection": "abstract",
                        "endSection": "abstract"
                    })
        if not candidate_snippets:
            return []

        snippet_embeddings = encode_contexts(candidate_snippets)
        scores = torch.matmul(snippet_embeddings.to(device), query_embedding.squeeze(0))  # shape: (num_snippets,)
        # Get top-k ranked indices (truncate if fewer candidates)
        k = min(top_k, scores.size(0))
        top_indices = torch.topk(scores, k).indices.tolist()

    return [snippet_metadata[i] for i in top_indices]

def main():
    # Sample biomedical question
    query = "Is rheumatoid arthritis more common in men or women?"

    # Sample biomedical articles (mocked data)
    articles = [
        {
            "url": "http://www.ncbi.nlm.nih.gov/pubmed/12723987",
            "title": "Gender differences in rheumatoid arthritis outcomes.",
            "abstract": "Rheumatoid arthritis is more common in women than in men. It causes joint inflammation and pain. Hormonal factors may influence disease severity.",
        },
        {
            "url": "http://www.ncbi.nlm.nih.gov/pubmed/22853635",
            "title": "RA diagnosis and progression in males and females.",
            "abstract": "In developed countries, the prevalence of RA is about 1%. Women are three times more likely to develop RA than men. Disease onset can vary significantly.",
        },
    ]

    # Step 1: Encode the query
    query_embedding = encode_query(query)

    # Step 2: Retrieve top-k snippets from these articles
    top_k = 5
    top_snippets = select_top_k_snippets_from_articles(query_embedding, articles, top_k=top_k)

    # Step 3: Display output
    print(f"\nQuery: {query}")
    print(f"\nTop {top_k} Retrieved Snippets:\n")
    for i, snip in enumerate(top_snippets, 1):
        print(f"{i}. [From {snip['beginSection']}] {snip['text']}")
        print(f"   URL: {snip['document']}")
        print(f"   Offset: {snip['offsetInBeginSection']}–{snip['offsetInEndSection']}\n")

if __name__ == "__main__":
    main()