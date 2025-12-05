import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

def get_all_metadata(session: requests.Session, base_url: str) -> list[dict]:
    """
    Scrapes the UQ PPL browse pages to extract metadata for all policy documents.

    Args:
        session: An active requests.Session object.
        base_url: The base URL for the PPL browse section (e.g., "https://policies.uq.edu.au/browse").

    Returns:
        A list of dictionaries, where each dictionary contains the 'url',
        'title', and 'summary' for a policy document.
    """
    print("--- Fetching metadata index ---")
    all_data = []

    try:
        # Get the main browse page to find summary links (A-Z)
        response = session.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Grab all <a> tags in the browse list headers
        links = soup.select("#jump-content div.browse-list h2 a")
        
        # Filter out "Return to Top" and resolve full URLs
        summary_urls = [
            urljoin(base_url, a.get("href"))
            for a in links if a.get("href") != "#top"
        ]
        print(f"Found {len(summary_urls)} summary pages to process.")

        # Iterate through each summary page (A-Z)
        for url in tqdm(summary_urls, desc="Processing summary pages"):
            resp = session.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Select all list items
            items = soup.select("#jump-content div.browse-list ul li")
            
            for li in items:
                a_tag = li.find("a")
                span_tag = li.find("span", class_="overview")
                if not a_tag: continue
                
                entry_url = a_tag.get("href")
                # Ensure the URL is absolute
                if not entry_url.startswith('http'):
                    entry_url = urljoin(base_url, entry_url)
                    
                title = a_tag.get_text(strip=True)
                summary = span_tag.get_text(strip=True) if span_tag else ""
                
                all_data.append({"url": entry_url, "title": title, "summary": summary})

    except requests.RequestException as e:
        print(f"❌ ERROR fetching metadata: {e}")
        # Depending on requirements, you might want to raise the exception
        # or return an empty list. Returning empty list for now.
        return []

    print(f"✅ Successfully fetched metadata for {len(all_data)} documents.")
    return all_data