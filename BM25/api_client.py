import requests
import json
import requests
from xml.etree import ElementTree as ET

def search_pubmed(query, max_results=10):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "xml"
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        pmids = [id_elem.text for id_elem in root.findall(".//Id")]
        return pmids
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] PubMed search failed for query: {query!r}\nReason: {e}")
        return []  # Fail-safe: return empty list to continue pipeline
def fetch_pubmed_details(pmids, batch_size=100):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    all_articles = []

    for i in range(0, len(pmids), batch_size):
        batch_pmids = pmids[i:i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch_pmids),
            "retmode": "xml"
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            root = ET.fromstring(response.text)
            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                title_elem = article.find(".//ArticleTitle")
                abstract_elems = article.findall(".//Abstract/AbstractText")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                title = title_elem.text if title_elem is not None else ""
                # Handle case where abstract has multiple parts or is None
                if abstract_elems:
                    abstract_parts = [a.text for a in abstract_elems if a.text]
                    abstract = " ".join(abstract_parts)
                else:
                    abstract = ""
                url = f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" if pmid else None

                all_articles.append({
                    "id": pmid,
                    "url": url,
                    "title": title,
                    "abstract": abstract
                })
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] PubMed fetch failed for PMIDs {batch_pmids[:3]}... (total {len(batch_pmids)})\nReason: {e}")
            continue  # Skip this batch and keep going

    return all_articles

if __name__ == "__main__":
    query = "rheumatoid arthritis gender prevalence"
    num_articles = 5
    pmids = search_pubmed(query, max_results=num_articles)
    articles = fetch_pubmed_details(pmids)
    for article in articles:
        print(f"PMID: {article['id']}")
        print(f"Title: {article['title']}")
        print(f"Abstract: {article['abstract']}")
        print(f"URL: {article['url']}\n")

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
        # print(f"Requesting session endpoint: {session_endpoint_url}") # Keep commented unless debugging
        daten = {"json": json.dumps(payload)}
        # print(daten) # Keep commented unless debugging
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

        # Extract relevant info (pmid/url, abstract, title)
        if "result" in results and "documents" in results["result"]:
            articles_data = []
            for doc in results["result"]["documents"]:
                pmid = doc.get('pmid')
                doc_url = f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" if pmid else None
                articles_data.append({
                    "id": pmid or doc_url, # Store PMID if available, else URL
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
        # This case might be redundant if the inner try-except handles it, but keep for safety
        print(f"Error decoding API response (outer catch): {response.text if 'response' in locals() else 'No response object'}")
        return []
