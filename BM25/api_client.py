import requests
import json

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

