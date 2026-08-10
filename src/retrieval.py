"""
retrieval.py — Hybrid retrieval cho recommendation RAG.

Pipeline:
    query
      ├─ Dense (Chroma vector)  → ứng viên theo NGỮ NGHĨA
      └─ BM25 (lexical)         → ứng viên theo TỪ KHÓA khớp chính xác (mã CPU/GPU, tên dòng)
            └─ RRF fuse (top_n) ─→ Cross-encoder rerank ─→ top_k

Tại sao:
- Bi-encoder (embedding) mã hoá query & doc RIÊNG rồi so cosine → nhanh, "hiểu" ngữ nghĩa,
  nhưng hay trượt khớp từ khóa chính xác và xếp hạng chưa tinh.
- BM25 bù phần khớp từ khóa. RRF ghép 2 danh sách chỉ bằng THỨ HẠNG (không cần chuẩn hoá score).
- Cross-encoder đọc CẢ (query, doc) cùng lúc → chấm độ liên quan chính xác hơn hẳn → rerank top-N.

Reranker chạy LOCAL (miễn phí). Model tải 1 lần (~2GB) ở lần dùng đầu.
"""
import bm25s
import numpy as np
from langchain_core.documents import Document

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANKER = None


def _get_reranker():
    """Lazy-load cross-encoder (chỉ tải model khi thực sự dùng)."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(RERANKER_MODEL)
    return _RERANKER


class HybridRetriever:
    def __init__(self, vectorstore):
        self.vs = vectorstore
        # Lấy toàn bộ corpus từ Chroma 1 lần để dựng BM25 trên CÙNG tập tài liệu
        data = vectorstore._collection.get(include=["documents", "metadatas"])
        self.docs = data["documents"]        # page_content
        self.metas = data["metadatas"]       # metadata (name, img, tags...)
        self.ids = data["ids"]               # = metadata['id'] (đặt lúc build)
        self.id2idx = {i: k for k, i in enumerate(self.ids)}
        # BM25 dùng bm25s: dựng INDEX THƯA (sparse, kiểu inverted-index) — chỉ đụng doc chứa
        # token qua phép nhân ma trận thưa, KHÔNG brute-force quét toàn bộ như rank_bm25.
        corpus_tokens = bm25s.tokenize(self.docs, stopwords=None, show_progress=False)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens, show_progress=False)

    def _dense_ranked(self, query, n):
        """Danh sách index xếp theo độ tương đồng vector."""
        idxs = []
        for d in self.vs.similarity_search(query, k=n):
            key = d.metadata.get("id")
            if key in self.id2idx:
                idxs.append(self.id2idx[key])
        return idxs

    def _bm25_ranked(self, query, n):
        """Danh sách index xếp theo điểm BM25 (bm25s, sparse index). Bỏ doc điểm 0 (không khớp)."""
        q_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
        results, scores = self.bm25.retrieve(q_tokens, k=min(n, len(self.docs)), show_progress=False)
        return [int(i) for i, s in zip(results[0], scores[0]) if s > 0]

    @staticmethod
    def _rrf(ranked_lists, k=60):
        """Reciprocal Rank Fusion: điểm = Σ 1/(k + rank). Chỉ dùng thứ hạng, không cần chuẩn hoá."""
        scores = {}
        for lst in ranked_lists:
            for rank, idx in enumerate(lst):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def search(self, query, k=5, top_n=20, rerank=True, return_scores=False):
        """Hybrid + RRF (+ rerank) → list[Document] top-k.
        return_scores=True → trả (docs, scores) với score rerank ∈ [0,1] (sigmoid).
        """
        dense = self._dense_ranked(query, top_n)
        bm25 = self._bm25_ranked(query, top_n)
        fused = self._rrf([dense, bm25])[:top_n]
        if not fused:
            return ([], []) if return_scores else []

        ordered_scores = None
        if rerank:
            ce = _get_reranker()
            raw = np.asarray(ce.predict([(query, self.docs[i]) for i in fused]), dtype=float)
            probs = 1.0 / (1.0 + np.exp(-raw))  # sigmoid → [0,1]
            order = np.argsort(probs)[::-1]
            fused = [fused[o] for o in order]
            ordered_scores = [float(probs[o]) for o in order]

        docs = [Document(page_content=self.docs[i], metadata=self.metas[i]) for i in fused[:k]]
        if return_scores:
            scores = (ordered_scores or [None] * len(fused))[:k]
            return docs, scores
        return docs
