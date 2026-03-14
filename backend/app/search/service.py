"""Web search service."""

import re
import logging
import httpx

from typing import List, Optional
from datetime import datetime

from app.config import settings
from app.models import SearchResult, WebSearchResult

logger = logging.getLogger(__name__)

# Map Chinese company names to English for better Tavily results
_COMPANY_EN_NAMES = {
    "阿里巴巴": "Alibaba BABA",
    "腾讯": "Tencent",
    "百度": "Baidu BIDU",
    "京东": "JD.com JD",
    "拼多多": "Pinduoduo PDD",
    "美团": "Meituan",
    "网易": "NetEase NTES",
    "比亚迪": "BYD",
    "小米": "Xiaomi",
    "特斯拉": "Tesla TSLA",
    "苹果": "Apple AAPL",
    "谷歌": "Google GOOGL",
    "亚马逊": "Amazon AMZN",
    "微软": "Microsoft MSFT",
    "英伟达": "NVIDIA NVDA",
    "Meta": "Meta META",
}


class WebSearchService:
    """Web search using Tavily when configured."""

    def __init__(self):
        self.api_key = settings.TAVILY_API_KEY
        self.base_url = "https://api.tavily.com/search"

    async def search(self, query: str, symbols: Optional[List[str]] = None, context: Optional[str] = None, max_results: int = 5) -> WebSearchResult:
        """Financial web search with bilingual query expansion."""
        if not self.api_key:
            return WebSearchResult(results=[], search_query=query)

        # 1. Clean query: remove question words
        search_terms = query
        for noise in ["为什么", "为何", "什么原因", "是什么", "怎么回事", "请问", "请分析", "帮我", "最近"]:
            search_terms = search_terms.replace(noise, " ")
        search_terms = search_terms.strip("？?，, 。. ")

        # 2. Add English company name for better Tavily coverage
        en_parts = []
        for cn_name, en_name in _COMPANY_EN_NAMES.items():
            if cn_name in query:
                en_parts.append(en_name)
        if symbols:
            for s in symbols:
                ticker = s.split(".")[0]  # "9988.HK" -> "9988"
                if ticker not in search_terms:
                    en_parts.append(s)

        # 3. Convert date to English format for search
        date_match = re.search(r'(\d{1,2})月(\d{1,2})日', query)
        if date_match:
            month, day = int(date_match.group(1)), int(date_match.group(2))
            month_names = ["", "January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            if 1 <= month <= 12:
                en_parts.append(f"{month_names[month]} {day}")

        # 4. Build final query
        enhanced_query = search_terms
        if en_parts:
            enhanced_query = f"{search_terms} {' '.join(en_parts)}"
        enhanced_query = enhanced_query.strip()

        # 5. Determine search time range
        search_days = 30  # default to 30 days for better coverage
        if date_match:
            month, day = int(date_match.group(1)), int(date_match.group(2))
            try:
                target = datetime(datetime.utcnow().year, month, day)
                days_ago = (datetime.utcnow() - target).days
                if days_ago > 0:
                    search_days = min(days_ago + 14, 180)
            except ValueError:
                pass

        logger.info(f"[WebSearch] query={enhanced_query}, days={search_days}")

        all_results = []
        # Search with both "news" and "general" topics for better coverage
        for topic in ["news", "general"]:
            try:
                async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
                    payload = {
                        "api_key": self.api_key,
                        "query": enhanced_query,
                        "max_results": max_results,
                        "search_depth": "advanced",
                        "topic": topic,
                        "days": search_days,
                        "include_answer": False,
                        "include_raw_content": False,
                    }

                    response = await client.post(self.base_url, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        seen_urls = {r.url for r in all_results}
                        for item in data.get("results", [])[:max_results]:
                            url = item.get("url", "")
                            if url in seen_urls:
                                continue
                            seen_urls.add(url)
                            all_results.append(SearchResult(
                                title=item.get("title", ""),
                                snippet=item.get("content", "")[:500],
                                url=url,
                                published=item.get("published_date"),
                                source=f"tavily_{topic}",
                            ))
            except Exception:
                continue

            # If first search already found good results, skip second
            if len(all_results) >= 3:
                break

        logger.info(f"[WebSearch] found {len(all_results)} results")
        return WebSearchResult(results=all_results[:max_results], search_query=enhanced_query)
