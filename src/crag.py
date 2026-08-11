"""
crag.py — Corrective RAG với LLM GRADER (đúng tinh thần paper CRAG 2024).

Ý tưởng: KHÔNG tin retrieval mù. Sau khi lấy kết quả → một LLM GRADER đọc (nhu cầu + các máy
lấy được) và phán 1 trong 3:
    - correct   : máy khớp tốt nhu cầu             → dùng luôn.
    - ambiguous : nhu cầu QUÁ MƠ HỒ/thiếu thông tin → HỎI LẠI user (không đoán bừa).
    - incorrect : lấy SAI, không liên quan          → REWRITE query, tìm lại, chấm lại.

Khác bản cũ (chỉ dùng điểm rerank): grader là LLM → ĐỌC được nội dung, phân biệt được "mơ hồ"
(cần hỏi lại) với "sai" (cần tìm lại) — thứ mà một con số rerank không tách được.

Chi phí: +1 LLM call (grader) mỗi lần; thêm rewrite/clarify khi cần. Dùng gpt-4o-mini nên rẻ.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

GRADER_SYSTEM = (
    "Bạn là bộ chấm chất lượng kết quả tra cứu laptop. Đọc NHU CẦU và DANH SÁCH máy lấy được, "
    "phán một trong ba nhãn:\n"
    "- 'ambiguous': khi KHÔNG rõ TÁC VỤ CỤ THỂ để chọn cấu hình — vd chỉ nói 'đi thực tập', 'đi "
    "làm', 'đi học', 'dùng chung chung' mà KHÔNG cho biết ngành/tác vụ (chơi game? đồ họa? lập "
    "trình? AI?...) VÀ cũng không có ngân sách/hãng. Nếu có TÁC VỤ cụ thể (game/đồ họa/lập trình/AI/"
    "văn phòng/giải trí) HOẶC ngân sách/hãng/RAM thì KHÔNG phải ambiguous.\n"
    "- 'correct': nhu cầu đủ rõ VÀ các máy khớp tốt.\n"
    "- 'incorrect': nhu cầu đủ rõ nhưng máy lấy về KHÔNG liên quan.\n"
    "Chỉ chấm dựa trên nhu cầu và danh sách được cho."
)


class Grade(BaseModel):
    verdict: str = Field(description="Một trong: 'correct' | 'ambiguous' | 'incorrect'")
    reason: str = Field(description="Lý do ngắn gọn (1 câu).")


def grade_retrieval(model, query, docs) -> Grade:
    """LLM grader: chấm (nhu cầu, docs) → correct/ambiguous/incorrect."""
    ctx = "\n".join(
        f"- {d.metadata.get('name', '?')}: {d.page_content[:180]}" for d in docs[:5]
    ) or "(không lấy được máy nào)"
    prompt = f"NHU CẦU:\n{query}\n\nDANH SÁCH MÁY LẤY ĐƯỢC:\n{ctx}"
    try:
        return model.with_structured_output(Grade).invoke(
            [SystemMessage(content=GRADER_SYSTEM), HumanMessage(content=prompt)]
        )
    except Exception:
        return Grade(verdict="correct", reason="grader lỗi → tạm tin retrieval")


def rewrite(model, query) -> str:
    resp = model.invoke([HumanMessage(content=(
        "Viết lại câu tìm laptop sau cho RÕ và GIÀU TỪ KHÓA kỹ thuật hơn (giữ nguyên ý, "
        f"chỉ 1 câu, không giải thích): {query}"
    ))])
    return (resp.content or query).strip()


def clarify_question(model, query) -> str:
    resp = model.invoke([HumanMessage(content=(
        f"Nhu cầu mua laptop chưa đủ rõ để tư vấn: '{query}'. Hãy viết 1 câu hỏi NGẮN, thân thiện "
        "hỏi thêm 1-2 thông tin QUAN TRỌNG NHẤT (mục đích dùng / ngân sách) để tư vấn tốt hơn."
    ))])
    return (resp.content or "Bạn cho mình biết thêm mục đích dùng và ngân sách nhé?").strip()


def corrective_retrieve(model, retriever, query, k=5, verbose=False):
    """
    LLM-grader CRAG:
      correct   → dùng luôn.
      ambiguous → HỎI LẠI user (clarify).
      incorrect → rewrite → tìm lại → chấm lại; vẫn không ổn → hỏi lại.
    Trả về (docs, clarify). clarify=None nếu ổn.
    """
    docs = retriever.search(query, k=k)
    grade = grade_retrieval(model, query, docs)
    if verbose:
        print(f"[CRAG] lần 1: {grade.verdict} — {grade.reason}")

    if grade.verdict == "correct":
        return docs, None
    if grade.verdict == "ambiguous":
        return docs, clarify_question(model, query)

    # incorrect → rewrite → thử lại → chấm lại
    q2 = rewrite(model, query)
    docs2 = retriever.search(q2, k=k)
    grade2 = grade_retrieval(model, query, docs2)
    if verbose:
        print(f"[CRAG] rewrite {q2!r} -> lần 2: {grade2.verdict} — {grade2.reason}")

    if grade2.verdict == "correct":
        return docs2, None
    # vẫn chưa ổn → trả kết quả tốt hơn nhưng KÈM hỏi lại (không đoán bừa)
    return docs2, clarify_question(model, query)
