import requests

def setup_session() -> requests.Session:
    """
    Creates and configures a standard requests.Session for scraping.
    """
    session = requests.Session()
    # Using a generic user agent is usually sufficient
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    return session