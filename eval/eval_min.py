"""
eval_min.py — Đánh giá retrieval, so sánh:
    (A) dense-only    (vector similarity thuần — "trước")
    (B) hybrid+rerank (BM25+dense+RRF+cross-encoder — "sau")

Relevance nhị phân theo metadata (tự kiểm chứng, khách quan):
- Câu NHU CẦU   → relevant nếu doc có đúng use_case_tag.
- Câu RÀNG BUỘC → relevant nếu doc thỏa giá/RAM/hãng.

Chỉ số: Precision@5, Precision@10, NDCG@5, NDCG@10, Hit@5.
- Precision@k: tỉ lệ top-k liên quan (đo "đúng bao nhiêu").
- NDCG@k: thưởng khi xếp món liên quan LÊN CAO (đo "xếp hạng tốt không").
- Hit@5: top-5 có ≥1 món liên quan.
(KHÔNG dùng Recall cho câu nhu cầu: tập đúng theo tag rất lớn → recall@k luôn tí xíu, vô nghĩa.)

Chạy:  python eval/eval_min.py
"""
import json
import math
import os
import sys

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from retrieval import HybridRetriever


def load_goldset(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def satisfies(meta, check):
    if "price_max" in check and not (meta.get("price_num", 0) <= check["price_max"]):
        return False
    if "ram_min" in check and not ((meta.get("ram_gb") or 0) >= check["ram_min"]):
        return False
    if "brand" in check and check["brand"] not in (meta.get("brand", "") or "").lower():
        return False
    return True


def relevance(item, meta):
    if item["type"] == "need":
        return 1.0 if item["tag"] in (meta.get("use_case_tags", "") or "") else 0.0
    return 1.0 if satisfies(meta, item["check"]) else 0.0


def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _ndcg(rels, k):
    idcg = _dcg([1.0] * k)  # lý tưởng: k món liên quan xếp đầu (pool liên quan >> k)
    return _dcg(rels[:k]) / idcg if idcg else 0.0


def eval_mode(name, retrieve_fn, gold):
    per_type = {}
    for item in gold:
        metas = retrieve_fn(item["query"], 10)[:10]
        rels = [relevance(item, m) for m in metas]
        m = {
            "P@5": sum(rels[:5]) / 5,
            "P@10": sum(rels[:10]) / 10,
            "NDCG@5": _ndcg(rels, 5),
            "NDCG@10": _ndcg(rels, 10),
            "Hit@5": 1.0 if any(rels[:5]) else 0.0,
        }
        per_type.setdefault(item["type"], []).append(m)
    out = {"name": name}
    for typ, lst in per_type.items():
        out[typ] = {key: sum(x[key] for x in lst) / len(lst) for key in lst[0]}
    return out


METRICS = ["P@5", "P@10", "NDCG@5", "NDCG@10", "Hit@5"]


def print_table(title, rows, typ):
    print(f"\n### {title}")
    print(f"{'Mode':20s}" + "".join(f"{m:>9s}" for m in METRICS))
    print("-" * (20 + 9 * len(METRICS)))
    for r in rows:
        vals = r[typ]
        print(f"{r['name']:20s}" + "".join(f"{vals[m]:9.3f}" for m in METRICS))


def main():
    gold = load_goldset(os.path.join(os.path.dirname(__file__), "goldset.jsonl"))
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(persist_directory="./vectorstore", embedding_function=emb, collection_name="langchain")
    retriever = HybridRetriever(vs)

    def dense_fn(q, k):
        return [d.metadata for d in vs.similarity_search(q, k=k)]

    def hybrid_fn(q, k):
        return [d.metadata for d in retriever.search(q, k=k, rerank=True)]

    n_need = sum(g["type"] == "need" for g in gold)
    n_con = sum(g["type"] == "constraint" for g in gold)
    print(f"Goldset: {len(gold)} câu ({n_need} nhu cầu, {n_con} ràng buộc)")

    rows = [eval_mode("A) dense-only", dense_fn, gold),
            eval_mode("B) hybrid+rerank", hybrid_fn, gold)]

    print_table("Câu NHU CẦU (relevance = đúng use_case_tag)", rows, "need")
    print_table("Câu RÀNG BUỘC (relevance = thỏa giá/RAM/hãng)", rows, "constraint")


if __name__ == "__main__":
    main()
