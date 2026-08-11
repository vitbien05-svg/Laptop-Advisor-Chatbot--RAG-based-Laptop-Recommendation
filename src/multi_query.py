"""
multi_query.py — Query expansion (Multi-Query Retrieval) cho recommendation RAG.

Ý tưởng: người dùng thường hỏi bằng "ngôn ngữ nhu cầu" (vd "máy để làm đồ án"), trong khi
tài liệu lại mô tả bằng "ngôn ngữ kỹ thuật" (CPU/GPU/RAM). Một câu hỏi duy nhất dễ TRƯỢT khi
thiếu keyword có trong KB. Multi-query sinh vài BIẾN THỂ diễn đạt lại cùng nhu cầu (đồng nghĩa,
thuật ngữ kỹ thuật, mục đích sử dụng) → retrieve cho từng biến thể → RRF hợp nhất tất cả.

→ Đây chính là bước "query processing" để vá lỗi "retrieval sai khi câu hỏi thiếu keyword".
Chi phí: +1 LLM call (rẻ, gpt-4o-mini) để sinh biến thể; phần retrieve vẫn cục bộ (ms).
"""
from langchain_core.messages import HumanMessage


def expand_query(model, query, n=3):
    """Sinh tối đa `n` biến thể + trả về [câu gốc, biến_thể_1, ...] (đã loại trùng, giữ thứ tự).
    Câu gốc LUÔN đứng đầu để dùng làm mốc rerank."""
    prompt = (
        f'Người dùng hỏi tư vấn mua laptop: "{query}".\n'
        f"Hãy viết {n} cách DIỄN ĐẠT LẠI khác nhau cho CÙNG nhu cầu đó, đa dạng từ khóa "
        "(đồng nghĩa, thuật ngữ kỹ thuật như CPU/GPU/RAM/SSD, mục đích sử dụng cụ thể). "
        "Mỗi biến thể trên 1 dòng riêng. KHÔNG đánh số, KHÔNG gạch đầu dòng, KHÔNG giải thích."
    )
    try:
        resp = model.invoke([HumanMessage(content=prompt)])
        raw = resp.content or ""
    except Exception:
        raw = ""  # LLM lỗi → fallback về chỉ câu gốc (không làm hỏng luồng)

    variants = [ln.strip(" -•\t*").strip() for ln in raw.splitlines() if ln.strip()]

    seen, out = set(), []
    for q in [query, *variants]:
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out[: n + 1]
