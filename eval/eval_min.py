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
import argparse
import json
import math
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from multi_query import expand_query
from query_decomposition import decompose_query
from retrieval import HybridRetriever

LOG_PATH = os.path.join(os.path.dirname(__file__), "experiments.md")


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


def _has_tag(item, meta):
    return item["tag"] in (meta.get("use_case_tags", "") or "")


def relevance(item, meta):
    if item["type"] == "need":
        return 1.0 if _has_tag(item, meta) else 0.0
    if item["type"] == "combined":  # câu KHÓ: đúng tag VÀ thỏa ràng buộc
        return 1.0 if (_has_tag(item, meta) and satisfies(meta, item["check"])) else 0.0
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


TYPES = [("need", "Câu NHU CẦU (relevance = đúng use_case_tag)"),
         ("constraint", "Câu RÀNG BUỘC (relevance = thỏa giá/RAM/hãng)"),
         ("combined", "Câu KHÓ (relevance = đúng tag VÀ thỏa ràng buộc)")]


def print_table(title, rows, typ):
    print(f"\n### {title}")
    print(f"{'Mode':22s}" + "".join(f"{m:>9s}" for m in METRICS))
    print("-" * (22 + 9 * len(METRICS)))
    for r in rows:
        vals = r[typ]
        print(f"{r['name']:22s}" + "".join(f"{vals[m]:9.3f}" for m in METRICS))


def append_log(rows, gold, note):
    """Ghi kết quả (markdown) kèm timestamp + ghi chú vào eval/experiments.md để lưu vết thí nghiệm."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n## {ts}" + (f" — {note}" if note else ""), f"Goldset: {len(gold)} câu\n"]
    for typ, title in TYPES:
        lines.append(f"**{title}**\n")
        lines.append("| Mode | " + " | ".join(METRICS) + " |")
        lines.append("|" + "---|" * (len(METRICS) + 1))
        for r in rows:
            vals = r[typ]
            lines.append(f"| {r['name']} | " + " | ".join(f"{vals[m]:.3f}" for m in METRICS) + " |")
        lines.append("")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n📝 Đã lưu kết quả vào {os.path.relpath(LOG_PATH, _ROOT)}")


def eval_recall(retriever, gold, all_metas, Ns=(20, 50)):
    """RECALL ĐÚNG — đo ở TẦNG RETRIEVAL (ứng viên hybrid rộng, CHƯA rerank).
    Recall@N = (số máy đúng lọt vào top-N ứng viên) / POOL (tổng máy đúng trong toàn KB).
    Tách khỏi display: đây đo 'lưới vớt có bỏ sót máy đúng không', không phải top-5 show ra.
    """
    per_type = {}
    N = max(Ns)
    for item in gold:
        pool = sum(relevance(item, m) for m in all_metas)  # tổng máy ĐÚNG trong kho
        cand = retriever.search(item["query"], k=N, top_n=N, rerank=False)  # ứng viên rộng
        rels = [relevance(item, d.metadata) for d in cand]
        row = {"pool": pool}
        for n in Ns:
            row[f"Recall@{n}"] = (sum(rels[:n]) / pool) if pool else 0.0
        per_type.setdefault(item["type"], []).append(row)
    return {typ: {k: sum(x[k] for x in lst) / len(lst) for k in lst[0]}
            for typ, lst in per_type.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="", help="Ghi chú cho lần thí nghiệm (lưu vào log)")
    ap.add_argument("--no-log", action="store_true", help="Không ghi vào experiments.md")
    ap.add_argument("--recall", action="store_true",
                    help="CHỉ đo Recall tầng retrieval (nhanh, không rerank/LLM)")
    ap.add_argument("--full", action="store_true",
                    help="Thêm 2 mode thí nghiệm: multi-query & decomposition")
    args = ap.parse_args()

    gold = load_goldset(os.path.join(os.path.dirname(__file__), "goldset.jsonl"))
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(persist_directory="./vectorstore", embedding_function=emb, collection_name="langchain")
    retriever = HybridRetriever(vs)

    if args.recall:
        res = eval_recall(retriever, gold, retriever.metas)
        print("\n### RECALL — tầng RETRIEVAL (ứng viên hybrid, chưa rerank); mẫu số = pool (toàn máy đúng)")
        for typ, title in TYPES:
            r = res.get(typ, {})
            print(f"\n{title}")
            print(f"  pool TB = {r.get('pool', 0):.0f} máy đúng | "
                  f"Recall@20 = {r.get('Recall@20', 0):.3f} | Recall@50 = {r.get('Recall@50', 0):.3f}")
        print("\n→ pool nhỏ (ràng buộc hẹp) => recall CAO & có nghĩa; pool lớn (nhu cầu rộng) => recall bị trần.")
        return

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)  # sinh biến thể/khía cạnh câu hỏi

    def dense_fn(q, k):
        return [d.metadata for d in vs.similarity_search(q, k=k)]

    def hybrid_fn(q, k):
        return [d.metadata for d in retriever.search(q, k=k, rerank=True)]

    def multiquery_fn(q, k):
        queries = expand_query(model, q, n=3)  # [gốc, +3 biến thể]
        return [d.metadata for d in retriever.search_multi(queries, k=k, rerank=True, rerank_query=q)]

    def decompose_fn(q, k):
        subs = decompose_query(model, q, max_parts=4)  # [gốc, +sub theo khía cạnh]
        return [d.metadata for d in retriever.search_multi(subs, k=k, rerank=True, rerank_query=q)]

    n_need = sum(g["type"] == "need" for g in gold)
    n_con = sum(g["type"] == "constraint" for g in gold)
    print(f"Goldset: {len(gold)} câu ({n_need} nhu cầu, {n_con} ràng buộc)")

    rows = [eval_mode("A) dense-only", dense_fn, gold),
            eval_mode("B) hybrid+rerank", hybrid_fn, gold)]
    if args.full:
        rows += [eval_mode("C) +multi-query", multiquery_fn, gold),
                 eval_mode("D) +decomposition", decompose_fn, gold)]

    for typ, title in TYPES:
        print_table(title, rows, typ)

    if not args.no_log:
        append_log(rows, gold, args.note)


if __name__ == "__main__":
    main()
