"""
query_decomposition.py — Query Decomposition cho recommendation RAG.

KHÁC multi-query:
- Multi-query: diễn đạt LẠI cùng một nhu cầu bằng nhiều cách (paraphrase) → tăng độ phủ từ khóa.
- Decomposition: TÁCH một nhu cầu phức hợp thành các KHÍA CẠNH độc lập, mỗi khía cạnh 1 sub-query
  (vd "laptop mỏng nhẹ pin trâu lập trình" → "mỏng nhẹ di động" / "pin trâu" / "CPU mạnh lập trình").

Sau khi tách, retrieve từng sub-query rồi RRF gộp: tài liệu thỏa NHIỀU khía cạnh sẽ xuất hiện ở
nhiều danh sách → điểm RRF cộng dồn → nổi lên top (xấp xỉ "giao" mềm các tiêu chí).

Hợp với câu nhu cầu ĐA TIÊU CHÍ; câu đơn giản thì tự trả về chính nó (không tách vô ích).
Chi phí: +1 LLM call (gpt-4o-mini). Ràng buộc CỨNG (giá/RAM) vẫn nên để MongoDB lọc, không tách ở đây.
"""
from langchain_core.messages import HumanMessage


def decompose_query(model, query, max_parts=4):
    """Tách `query` thành các sub-query theo khía cạnh. Trả [câu gốc, sub_1, ...] (loại trùng).
    Câu gốc đứng đầu để làm mốc rerank; nếu nhu cầu đơn giản, LLM trả về đúng 1 dòng."""
    prompt = (
        f'Nhu cầu mua laptop: "{query}".\n'
        f"Tách thành tối đa {max_parts} TIÊU CHÍ/khía cạnh ĐỘC LẬP để tìm kiếm riêng "
        "(ví dụ: mục đích dùng, hiệu năng CPU/GPU, tính di động/cân nặng, màn hình, pin). "
        "Mỗi tiêu chí là 1 cụm tìm kiếm NGẮN trên 1 dòng. Nếu nhu cầu đã đơn giản (1 tiêu chí) "
        "thì chỉ trả về đúng 1 dòng. KHÔNG đánh số, KHÔNG giải thích."
    )
    try:
        resp = model.invoke([HumanMessage(content=prompt)])
        raw = resp.content or ""
    except Exception:
        raw = ""  # lỗi LLM → fallback câu gốc

    parts = [ln.strip(" -•\t*").strip() for ln in raw.splitlines() if ln.strip()]

    seen, out = set(), []
    for q in [query, *parts]:
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out[: max_parts + 1]
