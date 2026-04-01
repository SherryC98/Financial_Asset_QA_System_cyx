"""
RAG Pipeline - Vector retrieval via SiliconFlow Embedding API (no local models)
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List
from pathlib import Path
import re
import logging
from openai import OpenAI
from app.config import settings
from app.models import KnowledgeResult, Document

_logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG: API-based embedding retrieval + cosine distance ranking"""

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
        # Initialize ChromaDB（解析为绝对路径，确保 RAG 向量库稳定接入）
        raw_dir = Path(settings.CHROMA_PERSIST_DIR)
        if not raw_dir.is_absolute():
            # backend 根目录 = app/rag -> app -> backend(/app in Docker)
            backend_root = Path(__file__).resolve().parents[2]
            persist_dir = backend_root / "vectorstore" / "chroma"
        else:
            persist_dir = raw_dir
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )

        # Get or create collection (with dimension compatibility check)
        self.collection = self.chroma_client.get_or_create_collection(
            name="financial_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        self._check_embedding_dimension_compat()

        self._embedding_client = None
        self._local_documents = self._load_local_documents()

    def _check_embedding_dimension_compat(self):
        """Check if existing ChromaDB embeddings match current model dimensions.

        If there's a dimension mismatch (e.g. old index built with 768-dim bge-base
        but current model is 1024-dim bge-large), delete and recreate the collection.
        """
        if self.collection.count() == 0:
            return

        try:
            peek = self.collection.peek(limit=1)
            if not peek or not peek.get("embeddings") or not peek["embeddings"]:
                return

            stored_dim = len(peek["embeddings"][0])
            expected_dim = 1024 if "large" in settings.EMBEDDING_MODEL else 768

            if stored_dim != expected_dim:
                _logger.warning(
                    f"[RAG] Embedding dimension mismatch: stored={stored_dim}, "
                    f"expected={expected_dim} ({settings.EMBEDDING_MODEL}). "
                    f"Deleting old collection and recreating..."
                )
                self.chroma_client.delete_collection("financial_knowledge")
                self.collection = self.chroma_client.get_or_create_collection(
                    name="financial_knowledge",
                    metadata={"hnsw:space": "cosine"}
                )
                _logger.info("[RAG] Collection recreated. Re-index needed.")
            else:
                _logger.info(f"[RAG] Embedding dimensions OK: {stored_dim}d")
        except Exception as e:
            _logger.warning(f"[RAG] Dimension check failed (non-fatal): {e}")

    def _get_embedding_client(self) -> OpenAI:
        """Lazy-init OpenAI client for SiliconFlow Embedding API."""
        if self._embedding_client is None:
            self._embedding_client = OpenAI(
                api_key=settings.EMBEDDING_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL,
            )
        return self._embedding_client

    def _load_local_documents(self) -> List[dict]:
        """Load from data/knowledge, raw_data/knowledge, raw_data/finance_report, dealed_data (md/json/html)."""
        base = Path(__file__).resolve().parents[2] / "data"
        documents = []
        seen_sources: set[str] = set()

        def add_doc(content: str, source: str, key: str) -> None:
            if not content or len(content.strip()) < 20:
                return
            if key in seen_sources:
                return
            seen_sources.add(key)
            documents.append({
                "source": source,
                "content": content,
                "tokens": self._tokenize_text(content),
            })

        # 1. knowledge, raw_data: 仅 md
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

        # 2. dealed_data: md, json, html
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
        """Extract text from MinerU JSON (pdf_info[].para_blocks[].lines[].spans[].content)."""
        try:
            import json
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
        """Extract text from HTML body."""
        try:
            import re
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
        """Generate query embedding via SiliconFlow API."""
        embeddings = self._call_embedding_api([query])
        return embeddings[0]

    async def search(self, query: str) -> KnowledgeResult:
        """
        Search: keyword match first, then vector search ranked by cosine distance.
        """
        local_result = self._search_local_documents(query)
        if local_result.documents:
            return local_result

        # Vector search via ChromaDB (cosine distance, lower = more similar)
        query_embedding = self._embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=settings.RAG_TOP_K
        )

        if not results['documents'] or not results['documents'][0]:
            return KnowledgeResult(documents=[], total_found=0)

        # Rank by cosine similarity (1 - distance) and filter by threshold
        ranked = []
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i] if results['distances'] else 0
            similarity = 1.0 - distance  # cosine space: distance in [0, 2]
            if similarity >= settings.RAG_SCORE_THRESHOLD:
                ranked.append(Document(
                    content=doc,
                    source=results['metadatas'][0][i].get('source', 'unknown'),
                    score=similarity,
                ))

        ranked.sort(key=lambda d: d.score, reverse=True)
        top_results = ranked[:settings.RAG_TOP_N]

        return KnowledgeResult(
            documents=top_results,
            total_found=len(results['documents'][0])
        )

    async def search_grounded(self, query: str, score_threshold: float = 0.3) -> KnowledgeResult:
        """Direct vector search without token-match shortcircuit.

        Unlike search(), this always queries ChromaDB.
        Only documents with similarity >= score_threshold are returned.
        """
        if self.collection.count() == 0:
            return KnowledgeResult(documents=[], total_found=0)

        query_embedding = self._embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=settings.RAG_TOP_K,
        )

        if not results["documents"] or not results["documents"][0]:
            return KnowledgeResult(documents=[], total_found=0)

        ranked = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            similarity = 1.0 - distance
            if similarity >= score_threshold:
                ranked.append(Document(
                    content=doc,
                    source=results["metadatas"][0][i].get("source", "unknown"),
                    score=similarity,
                ))

        ranked.sort(key=lambda d: d.score, reverse=True)

        return KnowledgeResult(
            documents=ranked[:settings.RAG_TOP_N],
            total_found=len(results["documents"][0]),
        )

    def add_documents(self, documents: List[str], metadatas: List[dict], ids: List[str]):
        """Add documents to the knowledge base using SiliconFlow API embeddings."""
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_embeddings = self._call_embedding_api(batch)
            all_embeddings.extend(batch_embeddings)
            _logger.info(f"[Embedding API] Embedded {min(i + batch_size, len(documents))}/{len(documents)} docs")

        self.collection.add(
            documents=documents,
            embeddings=all_embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def get_collection_count(self) -> int:
        """Get total document count"""
        return self.collection.count()
