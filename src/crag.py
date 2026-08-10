"""
crag.py — Corrective RAG (rút gọn, hợp bài toán recommendation).

Ý tưởng (từ paper CRAG 2024, thích nghi cho rec):
- KHÔNG tin retrieval mù quáng. Sau khi lấy kết quả → GRADER chấm: tốt / mơ hồ / kém.
- Kém  → SỬA: viết lại query (rewrite) rồi tìm lại.
- Vẫn kém → KHÔNG bịa, mà HỎI LẠI user (clarify) để lấy thêm ngân sách/nhu cầu.

Chi phí: +1 call chấm điểm mỗi lần (rẻ, gpt-4o-mini); chỉ khi kém mới tốn thêm rewrite/clarify.
"""
from langchain_core.messages import HumanMessage

# Ngưỡng gate theo ĐIỂM RERANK (sigmoid ∈ [0,1]), calibrate từ probe thực tế:
#   ≥ HIGH → tin tưởng (dùng luôn) | ≤ LOW → kém (sửa/hỏi lại) | giữa → tạm dùng
# Reranker cho ~0.50 khi KHÔNG khớp gì (điểm trung tính) → LOW đặt sát trên 0.50 để bắt nhiễu.
# HẠN CHẾ ĐÃ BIẾT: vùng 0.51-0.55 nhập nhằng — câu mơ hồ và câu-yếu-nhưng-hợp-lệ lẫn nhau;
# đặt LOW thấp để KHÔNG từ chối oan câu hợp lệ, đổi lại đôi khi bỏ sót câu mơ hồ.
SCORE_HIGH = 0.62
SCORE_LOW = 0.51


def rewrite(model, query) -> str:
    resp = model.invoke([HumanMessage(content=(
        "Viết lại câu tìm laptop sau cho RÕ và GIÀU TỪ KHÓA kỹ thuật hơn (giữ nguyên ý, "
        f"chỉ 1 câu, không giải thích): {query}"
    ))])
    return (resp.content or query).strip()


def clarify_question(model, query) -> str:
    resp = model.invoke([HumanMessage(content=(
        f"Chưa tìm được laptop phù hợp cho nhu cầu: '{query}'. Hãy viết 1 câu hỏi NGẮN, thân "
        "thiện hỏi thêm thông tin quan trọng nhất (ngân sách hoặc nhu cầu chính) để tư vấn tốt hơn."
    ))])
    return (resp.content or "Bạn cho mình biết thêm ngân sách và nhu cầu chính nhé?").strip()


def corrective_retrieve(model, retriever, query, k=5, verbose=False):
    """
    Gate theo điểm rerank (không tốn LLM call để chấm):
      - top ≥ HIGH → tin tưởng, dùng luôn.
      - top ≤ LOW  → kém → rewrite thử lại; vẫn kém → HỎI LẠI user.
      - giữa       → tạm dùng kết quả.
    Trả về (docs, clarify). clarify=None nếu ổn.
    """
    docs, scores = retriever.search(query, k=k, return_scores=True)
    top = scores[0] if scores else 0.0
    if verbose:
        print(f"[CRAG] top score lần 1: {top:.3f}")
    if top >= SCORE_HIGH:
        return docs, None

    # điểm thấp → rewrite rồi thử lại (chỉ tốn LLM khi cần)
    q2 = rewrite(model, query)
    docs2, scores2 = retriever.search(q2, k=k, return_scores=True)
    top2 = scores2[0] if scores2 else 0.0
    if verbose:
        print(f"[CRAG] rewrite: {q2!r} -> top score lần 2: {top2:.3f}")

    best_docs = docs2 if top2 >= top else docs
    if max(top, top2) <= SCORE_LOW:
        return best_docs, clarify_question(model, query)  # vẫn kém → hỏi lại
    return best_docs, None
