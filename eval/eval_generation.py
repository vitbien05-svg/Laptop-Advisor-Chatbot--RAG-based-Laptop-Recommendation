"""
eval_generation.py — Đánh giá TẦNG GENERATION của RAG (khác eval_min.py chỉ đo retrieval).

Luồng mỗi câu:  retrieve top-k (hybrid+rerank)  →  LLM tư vấn  →  đo 3 thuộc tính:
  1. Groundedness (TẤT ĐỊNH): máy LLM gợi ý có NẰM trong ngữ cảnh retrieve không → bắt "bịa máy".
     Cách đo: bóc token mã-máy đặc trưng (gần như duy nhất) trong câu trả lời, đối chiếu:
     token thuộc máy ĐÃ retrieve = grounded; token thuộc máy KHÁC (ngoài ngữ cảnh) = hallucination.
  2. Constraint-compliance: máy được gợi ý có thỏa ràng buộc user (giá/RAM/hãng) không.
  3. Need-coverage: câu trả lời có nhắc đúng KHÍA CẠNH của nhu cầu (gaming→GPU, mỏng nhẹ→cân nặng...).

Recommendation KHÔNG có 1 đáp án đúng → đo THUỘC TÍNH kiểm chứng được, không exact-match.
Chạy:  python eval/eval_generation.py
"""
import os
import re
import sys
from collections import Counter, defaultdict

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
load_dotenv(os.path.join(_ROOT, ".env"))

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from retrieval import HybridRetriever
from eval_min import load_goldset, satisfies

# Từ khóa để đo need-coverage: câu trả lời NÊN nhắc tới khía cạnh này cho từng nhu cầu
TAG_KEYWORDS = {
    "gaming": ["gpu", "rtx", "gtx", "card rời", "chơi game", "đồ họa rời"],
    "gaming_cao_cap": ["rtx", "vram", "card rời", "chơi game"],
    "lap_trinh_ky_thuat": ["ram", "cpu", "nhân", "luồng", "ssd"],
    "do_hoa_sang_tao": ["màu", "srgb", "oled", "ips", "đồ họa"],
    "mong_nhe_di_dong": ["nhẹ", "kg", "cân nặng", "di động", "pin"],
    "van_phong_hoc_tap": ["văn phòng", "học tập", "office", "sinh viên"],
    "hieu_nang_cao": ["ram", "hiệu năng", "cpu", "tác vụ nặng"],
    "giai_tri_da_phuong_tien": ["màn", "giải trí", "phim", "âm thanh"],
}

SYSTEM = (
    "Bạn là trợ lý tư vấn laptop cho sinh viên. CHỈ được gợi ý các máy CÓ trong DANH SÁCH ỨNG VIÊN "
    "cho sẵn — TUYỆT ĐỐI không nêu máy ngoài danh sách. Chọn 2-3 máy hợp nhất, nêu TÊN máy và GIẢI "
    "THÍCH ngắn gọn vì sao hợp nhu cầu (CPU/GPU/RAM/màn/cân nặng/giá) kèm đánh đổi. Trả lời tiếng Việt."
)


# Token SPEC (RAM/ổ cứng/tần số/công suất) — KHÔNG phải mã máy, phải loại kẻo báo bịa oan
_SPEC = re.compile(r"^\d+(GB|TB|MB|HZ|W|WH|NM|K|MM)$", re.I)


def model_tokens(name):
    """Token mã-máy: vừa chữ vừa số, ≥4 ký tự (vd 'A1505VA', 'FX608JHR', 'BQ1160W').
    Loại token spec (16GB, 512GB, 144HZ...) để không nhầm là 'máy bịa'."""
    toks = re.findall(r"[A-Za-z0-9]{4,}", name or "")
    return {t.upper() for t in toks
            if re.search(r"[A-Za-z]", t) and re.search(r"\d", t) and not _SPEC.match(t.upper())}


def build_prompt(query, docs):
    ctx = "\n\n".join(
        f"[{i+1}] {d.metadata.get('name','')} | {d.metadata.get('price_num','?')} triệu | "
        f"RAM {d.metadata.get('ram_gb','?')}GB | GPU {d.metadata.get('gpu_name','') or 'tích hợp'}\n"
        f"{d.page_content[:500]}"
        for i, d in enumerate(docs)
    )
    return f"NHU CẦU: {query}\n\nDANH SÁCH ỨNG VIÊN:\n{ctx}\n\nHãy tư vấn."


def main():
    gold = load_goldset(os.path.join(os.path.dirname(__file__), "goldset.jsonl"))
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(persist_directory="./vectorstore", embedding_function=emb, collection_name="langchain")
    r = HybridRetriever(vs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Vocab máy: token đặc trưng (gần như duy nhất, ≤2 máy) → tập chỉ số máy, để bắt "bịa máy ngoài ngữ cảnh"
    tok_count = Counter()
    tok_to_idxs = defaultdict(set)
    for i, m in enumerate(r.metas):
        for t in model_tokens(m.get("name", "")):
            tok_count[t] += 1
            tok_to_idxs[t].add(i)
    distinctive = {t for t, c in tok_count.items() if c <= 2}

    g_scores, comply_scores, cover_scores, halluc_total = [], [], [], 0
    print(f"Đánh giá generation trên {len(gold)} câu...\n")

    for item in gold:
        q = item["query"]
        docs = r.search(q, k=5, rerank=True)
        retr_idxs = {r.id2idx[d.metadata["id"]] for d in docs if d.metadata.get("id") in r.id2idx}
        ans = (llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=build_prompt(q, docs))]).content or "")

        # 1. Groundedness
        ans_toks = {t for t in model_tokens(ans) if t in distinctive}
        grounded = [t for t in ans_toks if tok_to_idxs[t] & retr_idxs]
        halluc = [t for t in ans_toks if not (tok_to_idxs[t] & retr_idxs)]
        g = 1.0 if not ans_toks else len(grounded) / len(ans_toks)
        g_scores.append(g)
        halluc_total += len(halluc)

        # máy được gợi ý (đã retrieve, được nhắc trong câu trả lời)
        rec_idxs = {i for t in grounded for i in (tok_to_idxs[t] & retr_idxs)}

        # 2. Constraint-compliance
        if item["type"] in ("constraint", "combined") and rec_idxs:
            comply = sum(satisfies(r.metas[i], item["check"]) for i in rec_idxs) / len(rec_idxs)
            comply_scores.append(comply)

        # 3. Need-coverage
        if item["type"] in ("need", "combined"):
            kws = TAG_KEYWORDS.get(item["tag"], [])
            cover_scores.append(1.0 if any(k in ans.lower() for k in kws) else 0.0)

        flag = f"  ⚠ BỊA: {halluc}" if halluc else ""
        print(f"[{item['type']:9s}] g={g:.2f} rec={len(rec_idxs)} :: {q[:48]}{flag}")

    def avg(x):
        return sum(x) / len(x) if x else float("nan")

    print("\n=== TỔNG HỢP GENERATION ===")
    print(f"Groundedness (không bịa máy)     : {avg(g_scores):.3f}   (1.0 = mọi máy nêu ra đều có trong ngữ cảnh)")
    print(f"Constraint-compliance            : {avg(comply_scores):.3f}   (máy gợi ý thỏa giá/RAM/hãng)")
    print(f"Need-coverage (nhắc đúng khía cạnh): {avg(cover_scores):.3f}")
    print(f"Tổng số token 'máy bịa' (ngoài ngữ cảnh): {halluc_total}")


if __name__ == "__main__":
    main()
