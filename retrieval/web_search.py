"""Live web scraping fallback using DuckDuckGo."""
from __future__ import annotations

from typing import List
from duckduckgo_search import DDGS

class WebRetriever:
    """Retrieves context directly from web search."""
    
    def __init__(self):
        self.ddgs = DDGS()
        
    def search_government_web(self, query: str, top_k: int = 2) -> List[str]:
        """Perform a web search focused on Indian government sites."""
        
        # Clean query by removing common overrides that might confuse the search engine
        search_query = query.lower()
        stopwords = ["in english", "english language", "अंग्रेजी में", "angreji me", 
                     "in hindi", "hindi language", "हिंदी में", "hindi me", "hindi mein",
                     "in gujarati", "gujarati language", "ગુજરાતી માં", "gujrati ma",
                     "tell me", "मुझे", "बताइए"]
        
        for word in stopwords:
            search_query = search_query.replace(word, "")
            
        search_query = search_query.strip()
        
        # Construct specific search target
        advanced_query = f"{search_query} site:gov.in"
        
        results = []
        try:
            # We fetch up to top_k * 2 results to filter out badly formatted ones
            raw_results = self.ddgs.text(advanced_query, max_results=top_k * 2)
            
            for res in raw_results:
                title = res.get("title", "")
                body = res.get("body", "")
                if body:
                    results.append(f"{title}: {body}")
                    if len(results) >= top_k:
                        break
        except Exception as e:
            print(f"[WARN] Web scrape failed: {e}")
            
        return results
