"""Web search service."""

import httpx

from typing import List, Optional
from datetime import datetime

from app.config import settings
from app.models import SearchResult, WebSearchResult



class WebSearchService:
    """Web search using Tavily when configured."""

    def __init__(self):
        self.api_key = settings.TAVILY_API_KEY
        self.base_url = "https://api.tavily.com/search"

    async def search(self, query: str, symbols: Optional[List[str]] = None, context: Optional[str] = None, max_results: int = 5) -> WebSearchResult:
        """Advanced financial web search with query expansion and high-signal domain filtering."""
        if not self.api_key:
            return WebSearchResult(results=[], search_query=query)

        # 1. Build a search-engine-friendly query
        import re as _re
        # Extract core terms: remove question words, keep entity + event + date
        search_terms = query
        # Remove common question patterns that don't help search
        for noise in ["为什么", "为何", "什么原因", "是什么", "怎么回事", "请问", "请分析", "帮我"]:
            search_terms = search_terms.replace(noise, "")
        search_terms = search_terms.strip("？?，, 。.")

        # Add stock symbols for better targeting
        if symbols:
            symbol_str = " ".join(symbols)
            search_terms = f"{search_terms} {symbol_str} stock"

        enhanced_query = search_terms.strip()

        # 2. Market Fact Injection
        if context:
            enhanced_query = f"{enhanced_query} {context}"

        # 3. Determine search time range based on query content
        search_days = 7
        import re
        date_match = re.search(r'(\d{1,2})月(\d{1,2})日', query)
        if date_match:
            month, day = int(date_match.group(1)), int(date_match.group(2))
            try:
                target = datetime(datetime.utcnow().year, month, day)
                days_ago = (datetime.utcnow() - target).days
                if days_ago > 0:
                    search_days = min(days_ago + 7, 180)
            except ValueError:
                pass

        try:
            async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
                payload = {
                    "api_key": self.api_key,
                    "query": enhanced_query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                    "topic": "news",
                    "days": search_days,
                    "include_answer": False,
                    "include_raw_content": False,
                }
                
                response = await client.post(self.base_url, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    results = []

                    for item in data.get("results", [])[:max_results]:
                        results.append(SearchResult(
                            title=item.get("title", ""),
                            snippet=item.get("content", "")[:500],
                            url=item.get("url", ""),
                            published=item.get("published_date"),
                            source="tavily_financial_news",
                        ))

                    return WebSearchResult(results=results, search_query=enhanced_query)

        except Exception:
            pass

        return WebSearchResult(results=[], search_query=query)
