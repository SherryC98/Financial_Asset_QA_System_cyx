"""
RAG Pipeline - Keyword search + optional SiliconFlow Embedding API vector search.
No heavy local models (no torch/transformers/chromadb).
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from app.config import settings
from app.models import KnowledgeResult, Document

_logger = logging.getLogger(__name__)


class RAGPipeline:
    """Lightweight RAG: keyword search primary, API-based vector search optional."""

    QUERY_EXPANSIONS = {
        "市盈率": {"pe", "price-to-earnings", "valuation", "估值"},
        "市净率": {"pb", "price-to-book", "book value", "估值"},
        "市销率": {"ps", "price-to-sales", "sales"},
        "波动率": {"volatility", "risk", "drawdown", "technical"},
        "最大回撤": {"drawdown", "risk", "technical"},
        "财务报表": {"balance sheet", "income statement", "cash flow", "financial statements"},
        "现金流": {"cash flow", "financial statements"},
        "技术分析": {"technical", "rsi", "macd", "support", "resistance"},
        "债券": {"bond", "fixed income", "market instruments"},
        "etf": {"fund", "market instruments"},
        "宏观": {"macro", "economics", "macro economics"},
    }

    def __init__(self):
        self._embedding_client = None
        self._local_documents = self._load_local_documents()
        _logger.info(f"[RAG] Loaded {len(self._local_documents)} local documents for keyword search")

    # ------------------------------------------------------------------
    # Embedding API (SiliconFlow, OpenAI-compatible)
    # ------------------------------------------------------------------

    def _get_embedding_client(self) -> OpenAI:
        if self._embedding_client is None:
            self._embedding_client = OpenAI(
                api_key=settings.EMBEDDING_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL,
            )
        return self._embedding_client

    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Call SiliconFlow Embedding API (OpenAI-compatible format)."""
        client = self._get_embedding_client()
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts,
            encoding_format="float",
        )
        sorted_items = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_items]

    def _embed_query(self, query: str) -> List[float]:
        embeddings = self._call_embedding_api([query])
        return embeddings[0]

    # ------------------------------------------------------------------
    # Local document loading (keyword search corpus)
    # ------------------------------------------------------------------

    def _load_local_documents(self) -> List[dict]:
        """Load from data/knowledge, raw_data/knowledge, raw_data/finance_report, dealed_data."""
        base = Path(__file__).resolve().parents[2] / "data"
        documents = []
        seen_sources: set[str] = set()

        MAX_SNIPPET = 1500  # Only keep first 1500 chars to save memory

        def add_doc(content: str, source: str, key: str) -> None:
            if not content or len(content.strip()) < 20:
                return
            if key in seen_sources:
                return
            seen_sources.add(key)
            snippet = content[:MAX_SNIPPET].strip()
            documents.append({
                "source": source,
                "content": snippet,
                "tokens": self._tokenize_text(snippet),
            })

        for rel_dir in ("knowledge", "raw_data/knowledge", "raw_data/finance_report"):
            dir_path = base / rel_dir
            if not dir_path.exists():
                continue
            for file_path in sorted(dir_path.rglob("*.md")):
                key = f"{rel_dir}/{file_path.name}"
                content = None
                for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
                    try:
                        content = file_path.read_text(encoding=encoding)
                        break
                    except Exception:
                        continue
                if content:
                    add_doc(content, file_path.name, key)

        dealed_dir = base / "dealed_data"
        if dealed_dir.exists():
            for file_path in sorted(dealed_dir.iterdir()):
                if not file_path.is_file():
                    continue
                suffix = file_path.suffix.lower()
                key = f"dealed_data/{file_path.name}"
                content = None

                if suffix == ".md":
                    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
                        try:
                            content = file_path.read_text(encoding=encoding)
                            break
                        except Exception:
                            continue
                elif suffix == ".json":
                    content = self._extract_text_from_mineru_json(file_path)
                elif suffix == ".html":
                    content = self._extract_text_from_html(file_path)

                if content:
                    add_doc(content, file_path.name, key)

        return documents

    @staticmethod
    def _extract_text_from_mineru_json(file_path: Path) -> str:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            parts = []
            for page in data.get("pdf_info", []):
                for block in page.get("para_blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            c = span.get("content", "").strip()
                            if c:
                                parts.append(c)
            return "\n".join(parts) if parts else ""
        except Exception:
            return ""

    @staticmethod
    def _extract_text_from_html(file_path: Path) -> str:
        try:
            raw = file_path.read_text(encoding="utf-8")
            raw = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", raw, flags=re.I)
            raw = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", raw, flags=re.I)
            body = re.search(r"<body[^>]*>([\s\S]*?)</body>", raw, re.I)
            if body:
                body = body.group(1)
            else:
                body = raw
            text = re.sub(r"<[^>]+>", " ", body)
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        except Exception:
            return ""

    @staticmethod
    def _tokenize_text(text: str) -> set[str]:
        lowered = text.lower()
        tokens = {
            token
            for token in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", lowered)
            if len(token) >= 2
        }
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
            for size in range(2, min(5, len(chunk) + 1)):
                for index in range(0, len(chunk) - size + 1):
                    tokens.add(chunk[index:index + size])
        return tokens

    # ------------------------------------------------------------------
    # Keyword search (primary path — no heavy deps)
    # ------------------------------------------------------------------

    def _search_local_documents(self, query: str) -> KnowledgeResult:
        query_tokens = self._tokenize_text(query)
        expanded_terms = set()
        for keyword, synonyms in self.QUERY_EXPANSIONS.items():
            if keyword in query.lower() or keyword in query:
                expanded_terms.update(synonyms)
        for term in expanded_terms:
            query_tokens.update(self._tokenize_text(term))
        if not query_tokens:
            return KnowledgeResult(documents=[], total_found=0)

        ranked = []
        for item in self._local_documents:
            overlap = len(query_tokens & item["tokens"])
            if overlap == 0 and any(term in item["content"].lower() for term in expanded_terms):
                overlap = 1
            if overlap == 0:
                continue
            snippet = item["content"][:600].strip()
            ranked.append(
                Document(
                    content=snippet,
                    source=item["source"],
                    score=float(overlap),
                )
            )

        ranked.sort(key=lambda doc: doc.score, reverse=True)
        top_results = ranked[:settings.RAG_TOP_N]
        return KnowledgeResult(documents=top_results, total_found=len(ranked))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, query: str) -> KnowledgeResult:
        """Primary search: keyword match on local documents."""
        return self._search_local_documents(query)

    async def search_grounded(self, query: str, score_threshold: float = 0.3) -> KnowledgeResult:
        """Same as search() — keyword match only (ChromaDB removed to save memory)."""
        return self._search_local_documents(query)

    def get_collection_count(self) -> int:
        """Return number of local documents available for keyword search."""
        return len(self._local_documents)
