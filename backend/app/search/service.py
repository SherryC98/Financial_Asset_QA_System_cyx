"""Web search service with Finnhub company news integration."""

import re
import logging
import httpx
import asyncio

from typing import List, Optional
from datetime import datetime, timedelta

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

# Map HK/CN symbols to US ADR tickers for Finnhub (which mainly covers US markets)
_HK_TO_US_TICKER = {
    "9988.HK": "BABA",
    "9618.HK": "JD",
    "9999.HK": "NTES",
    "1810.HK": "XIACY",
    "0700.HK": "TCEHY",
}


class WebSearchService:
    """Web search using Tavily + Finnhub company news."""

    def __init__(self):
        self.tavily_key = settings.TAVILY_API_KEY
        self.finnhub_key = getattr(settings, "FINNHUB_API_KEY", None)
        self.tavily_url = "https://api.tavily.com/search"

    async def search(self, query: str, symbols: Optional[List[str]] = None, context: Optional[str] = None, max_results: int = 5) -> WebSearchResult:
        """Financial web search combining Finnhub company news + Tavily."""
        # 1. Parse date from query
        date_match = re.search(r'(\d{1,2})月(\d{1,2})日', query)
        target_date = None
        if date_match:
            month, day = int(date_match.group(1)), int(date_match.group(2))
            try:
                target_date = datetime(datetime.utcnow().year, month, day)
            except ValueError:
                pass

        # 2. Run Finnhub company news + Tavily in parallel
        finnhub_coro = self._fetch_finnhub_news(symbols, target_date) if symbols and self.finnhub_key else None
        tavily_coro = self._fetch_tavily(query, symbols, target_date, max_results)

        if finnhub_coro:
            results = await asyncio.gather(finnhub_coro, tavily_coro, return_exceptions=True)
        else:
            tavily_result = await tavily_coro
            results = [[], tavily_result]

        all_results: List[SearchResult] = []

        # Finnhub results first (more targeted)
        if not isinstance(results[0], Exception) and results[0]:
            all_results.extend(results[0])
            logger.info(f"[WebSearch] Finnhub returned {len(results[0])} company news")

        # Then Tavily results
        if not isinstance(results[1], Exception) and results[1]:
            seen_urls = {r.url for r in all_results}
            for r in results[1]:
                if r.url not in seen_urls:
                    all_results.append(r)
                    seen_urls.add(r.url)

        logger.info(f"[WebSearch] Total results: {len(all_results)}")
        enhanced_query = query
        if symbols:
            enhanced_query = f"{query} ({', '.join(symbols)})"
        return WebSearchResult(results=all_results[:max_results], search_query=enhanced_query)

    async def _fetch_finnhub_news(self, symbols: List[str], target_date: Optional[datetime] = None) -> List[SearchResult]:
        """Fetch company-specific news from Finnhub."""
        results = []
        for symbol in symbols[:2]:
            # Convert HK symbols to US ADR tickers
            ticker = _HK_TO_US_TICKER.get(symbol, symbol.split(".")[0] if "." in symbol else symbol)

            # Date range: 7 days around target date, or last 30 days
            if target_date:
                from_date = (target_date - timedelta(days=3)).strftime("%Y-%m-%d")
                to_date = (target_date + timedelta(days=3)).strftime("%Y-%m-%d")
            else:
                to_date = datetime.utcnow().strftime("%Y-%m-%d")
                from_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(
                        "https://finnhub.io/api/v1/company-news",
                        params={"symbol": ticker, "from": from_date, "to": to_date, "token": self.finnhub_key},
                    )
                    if response.status_code != 200:
                        continue
                    articles = response.json()
                    if not isinstance(articles, list):
                        continue

                    for article in articles[:5]:
                        headline = article.get("headline", "")
                        summary = article.get("summary", "")
                        pub_ts = article.get("datetime", 0)
                        pub_date = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d") if pub_ts else None

                        if headline:
                            results.append(SearchResult(
                                title=headline,
                                snippet=summary[:500] if summary else headline,
                                url=article.get("url", ""),
                                published=pub_date,
                                source=f"finnhub_{article.get('source', 'news')}",
                            ))
            except Exception as e:
                logger.warning(f"[WebSearch] Finnhub news failed for {ticker}: {e}")
                continue

        return results

    async def _fetch_tavily(self, query: str, symbols: Optional[List[str]], target_date: Optional[datetime], max_results: int) -> List[SearchResult]:
        """Fetch results from Tavily search."""
        if not self.tavily_key:
            return []

        # Clean query
        search_terms = query
        for noise in ["为什么", "为何", "什么原因", "是什么", "怎么回事", "请问", "请分析", "帮我", "最近"]:
            search_terms = search_terms.replace(noise, " ")
        search_terms = search_terms.strip("？?，, 。. ")

        # Add English names
        en_parts = []
        for cn_name, en_name in _COMPANY_EN_NAMES.items():
            if cn_name in query:
                en_parts.append(en_name)
        if symbols:
            for s in symbols:
                ticker = _HK_TO_US_TICKER.get(s, s)
                if ticker not in search_terms:
                    en_parts.append(ticker)

        # Add English date
        if target_date:
            month_names = ["", "January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            en_parts.append(f"{month_names[target_date.month]} {target_date.day}")

        enhanced_query = search_terms
        if en_parts:
            enhanced_query = f"{search_terms} {' '.join(en_parts)}"
        enhanced_query = enhanced_query.strip()

        # Search time range
        search_days = 30
        if target_date:
            days_ago = (datetime.utcnow() - target_date).days
            if days_ago > 0:
                search_days = min(days_ago + 14, 180)

        logger.info(f"[WebSearch] Tavily query={enhanced_query}, days={search_days}")

        results = []
        try:
            async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
                payload = {
                    "api_key": self.tavily_key,
                    "query": enhanced_query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                    "topic": "news",
                    "days": search_days,
                    "include_answer": False,
                    "include_raw_content": False,
                }
                response = await client.post(self.tavily_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("results", [])[:max_results]:
                        results.append(SearchResult(
                            title=item.get("title", ""),
                            snippet=item.get("content", "")[:500],
                            url=item.get("url", ""),
                            published=item.get("published_date"),
                            source="tavily_news",
                        ))
        except Exception as e:
            logger.warning(f"[WebSearch] Tavily failed: {e}")

        return results
