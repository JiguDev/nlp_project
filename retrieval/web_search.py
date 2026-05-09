"""Live web scraping using Wikipedia API + requests-based search."""
from __future__ import annotations

import re
from typing import List

import requests
from bs4 import BeautifulSoup


class WebRetriever:
    """Retrieves context from Wikipedia API and web scraping."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GovChatbot/1.0 (MTech NLP Project; educational use)"
        })

    # ------------------------------------------------------------------ #
    # 1.  Wikipedia API  (reliable, fast, structured)
    # ------------------------------------------------------------------ #
    def search_wikipedia(self, query: str, sentences: int = 5, lang: str = "en") -> List[str]:
        """Fetch a Wikipedia summary for *query* via the official API."""
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": 3,
        }
        results: List[str] = []
        try:
            resp = self.session.get(api_url, params=params, timeout=10)
            resp.raise_for_status()
            search_hits = resp.json().get("query", {}).get("search", [])

            for hit in search_hits[:2]:
                title = hit.get("title", "")
                # Now fetch the actual summary of this page
                summary = self._get_wiki_summary(title, sentences=sentences, lang=lang)
                if summary:
                    results.append(summary)
        except Exception as e:
            print(f"[WARN] Wikipedia API search failed: {e}")
        return results

    def _get_wiki_summary(self, title: str, sentences: int = 5, lang: str = "en") -> str:
        """Retrieve the introductory summary for a Wikipedia article."""
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
            "utf8": 1,
            "exsentences": sentences,
        }
        try:
            resp = self.session.get(api_url, params=params, timeout=10)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "").strip()
                if extract and len(extract) > 30:
                    return extract
        except Exception as e:
            print(f"[WARN] Wiki summary fetch failed for '{title}': {e}")
        return ""

    # ------------------------------------------------------------------ #
    # 2.  Google / DuckDuckGo HTML scraping fallback
    # ------------------------------------------------------------------ #
    def _scrape_search_snippets(self, query: str, top_k: int = 3) -> List[str]:
        """Scrape search result snippets from DuckDuckGo HTML (no API key)."""
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query}
        results: List[str] = []
        try:
            resp = self.session.post(url, data=data, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for result_div in soup.select(".result__body"):
                snippet_tag = result_div.select_one(".result__snippet")
                title_tag = result_div.select_one(".result__a")
                if snippet_tag:
                    snippet = snippet_tag.get_text(strip=True)
                    title = title_tag.get_text(strip=True) if title_tag else ""
                    if snippet and len(snippet) > 20:
                        text = f"{title}: {snippet}" if title else snippet
                        results.append(text)
                        if len(results) >= top_k:
                            break
        except Exception as e:
            print(f"[WARN] HTML DDG scrape failed: {e}")
        return results

    # ------------------------------------------------------------------ #
    # 3.  Government-site targeted search
    # ------------------------------------------------------------------ #
    def search_government_web(self, query: str, top_k: int = 2) -> List[str]:
        """Search specifically for Indian government websites."""
        clean_q = self._clean_query(query)
        return self._scrape_search_snippets(f"{clean_q} site:gov.in", top_k=top_k)

    # ------------------------------------------------------------------ #
    # 4.  Combined search  (main entry point)
    # ------------------------------------------------------------------ #
    def search_all(self, query: str, top_k: int = 3) -> List[str]:
        """
        Try multiple sources in order of reliability:
          1. Wikipedia API  (best quality)
          2. DDG HTML scrape for gov.in
          3. DDG HTML scrape (general)
        Returns the first batch that succeeds.
        """
        clean_q = self._clean_query(query)
        print(f"[WebRetriever] Searching for: '{clean_q}'")

        # --- Wikipedia (most reliable) ---
        wiki = self.search_wikipedia(clean_q, sentences=6)
        if wiki:
            print(f"[WebRetriever] Wikipedia returned {len(wiki)} result(s)")
            return wiki[:top_k]

        # --- Gov.in scrape ---
        gov = self._scrape_search_snippets(f"{clean_q} site:gov.in", top_k=top_k)
        if gov:
            print(f"[WebRetriever] Gov.in scrape returned {len(gov)} result(s)")
            return gov

        # --- General web scrape ---
        general = self._scrape_search_snippets(clean_q, top_k=top_k)
        if general:
            print(f"[WebRetriever] General scrape returned {len(general)} result(s)")
            return general

        print("[WebRetriever] All sources returned 0 results")
        return []

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_query(query: str) -> str:
        """Remove language-override phrases so the search focuses on the topic."""
        q = query.lower()
        stopwords = [
            "in english", "english language", "अंग्रेजी में", "angreji me",
            "in hindi", "hindi language", "हिंदी में", "hindi me", "hindi mein",
            "in gujarati", "gujarati language", "ગુજરાતી માં", "gujrati ma",
            "tell me about same in detail", "tell me about", "tell me in detail",
            "tell me", "reply me in hindi", "reply me in gujarati", "reply me in english",
            "reply me in", "reply in", "reply me",
            "what is the", "what is a", "what is",
            "explain about", "explain",
            "in detail", "same in detail",
            "मुझे", "बताइए", "बताओ",
        ]
        for w in stopwords:
            q = q.replace(w, "")
        q = re.sub(r"\s+", " ", q).strip()
        # Remove dangling punctuation
        q = re.sub(r"^[,.\s]+|[,.\s]+$", "", q)
        q = re.sub(r",\s*,", ",", q).strip(", ")

        # Common Indian government scheme corrections (substring match)
        corrections = {
            "manrega": "MGNREGA Mahatma Gandhi National Rural Employment Guarantee Act",
            "mnrega": "MGNREGA Mahatma Gandhi National Rural Employment Guarantee Act",
            "mgnrega": "MGNREGA Mahatma Gandhi National Rural Employment Guarantee Act",
            "nrega": "MGNREGA Mahatma Gandhi National Rural Employment Guarantee Act",
            "aadhar": "Aadhaar card India",
            "adhar": "Aadhaar card India",
            "aadhaar": "Aadhaar card India",
            "pan card": "Permanent Account Number PAN card India",
            "rti": "Right to Information Act India",
        }
        q_lower = q.lower().strip()
        for key, replacement in corrections.items():
            if key in q_lower:
                q = replacement
                break

        return q if q else query.strip()
