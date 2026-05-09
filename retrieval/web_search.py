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
        Aggressive retrieval strategy:
        1. Clean and identify language.
        2. Always search English sources (Wikipedia/Gov.in) for Indian schemes.
        3. Fallback to native language Wikipedia only if English fails.
        """
        original_q = query.strip()
        clean_q = self._clean_query(query)
        
        # Identify script
        is_indic = any('\u0900' <= c <= '\u097F' for c in clean_q) or any('\u0A80' <= c <= '\u0AFF' for c in clean_q)
        
        search_queries = [clean_q]
        if is_indic:
            try:
                from deep_translator import GoogleTranslator
                translated_q = GoogleTranslator(source='auto', target='en').translate(clean_q)
                print(f"[WebRetriever] Indic script detected. Searching English fallback: '{translated_q}'")
                search_queries.insert(0, translated_q) # Prioritize English search
            except Exception:
                pass

        results = []
        for q in search_queries:
            print(f"[WebRetriever] Querying: '{q}'")
            # 1. Wikipedia English (High quality)
            wiki_en = self.search_wikipedia(q, sentences=6, lang="en")
            if wiki_en:
                results.extend(wiki_en)
            
            # 2. Gov.in Targeted Scrape
            gov = self._scrape_search_snippets(f"{q} site:gov.in", top_k=top_k)
            if gov:
                results.extend(gov)
            
            if results:
                break # Found high quality data

        # 3. Last resort: Native Wikipedia
        if not results and is_indic:
            lang = "hi" if any('\u0900' <= c <= '\u097F' for c in clean_q) else "gu"
            native = self.search_wikipedia(clean_q, sentences=6, lang=lang)
            results.extend(native)

        if not results:
            print("[WebRetriever] All sources returned 0 results")
            return []
        
        return results[:top_k]

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_query(query: str) -> str:
        """Remove language-override phrases and normalize."""
        q = query.lower()
        # Remove noisy suffixes/prefixes
        noise = [
            "in hindi", "in gujarati", "in english", "tell me about", "reply me", 
            "हिंदी में", "ગુજરાતી માં", "explain", "what is", "about", "details of"
        ]
        for n in noise:
            q = q.replace(n, "")
        
        q = re.sub(r"\s+", " ", q).strip()
        
        # Specific Indian Gov keyword boosting
        corrections = {
            "nps": "National Pension System India",
            "mgnrega": "MGNREGA Scheme India",
            "aadhar": "Aadhaar Card UIDAI",
            "pan": "Income Tax PAN Card India",
            "rti": "Right to Information Act India",
            "pm kisan": "PM-Kisan Scheme India",
        }
        for k, v in corrections.items():
            if k in q:
                return v # Return the boosted query immediately
                
        return q if len(q) > 2 else query
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
