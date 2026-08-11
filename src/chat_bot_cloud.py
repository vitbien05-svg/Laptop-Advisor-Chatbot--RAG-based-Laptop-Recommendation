"""
chat_bot_cloud.py — RAG tư vấn laptop (luồng CỐ ĐỊNH, không còn agent tự-quyết-tool).

Luồng:
    query
      → ① Query understanding: LLM bóc ràng buộc CỨNG (giá/RAM/hãng-loại-trừ)
      → ② LUÔN retrieve + CRAG (LLM grader): correct → dùng | ambiguous → HỎI LẠI | incorrect → rewrite
      → ③ Lọc ràng buộc cứng (tất định, gồm LOẠI hãng — vd 'né Acer')
      → ④ LLM tư vấn có giải thích 'tại sao' + đánh đổi
Vì CRAG nằm ở bước ② luôn chạy, KHÔNG bị agent bypass như bản cũ.
"""
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
import re

# ══════════════════════════════════════════════════════════════
# KHỞI TẠO MODEL & VECTORSTORE
# ══════════════════════════════════════════════════════════════
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./vectorstore", embedding_function=embeddings,
                     collection_name="langchain")

from retrieval import HybridRetriever
from crag import corrective_retrieve
_retriever = HybridRetriever(vectorstore)

# ── Gom sản phẩm truy xuất trong 1 lượt (để UI show ảnh) ──
_retrieved_products: list = []


def _add_product(p: dict):
    if p.get("url") and all(p["url"] != x.get("url") for x in _retrieved_products):
        _retrieved_products.append(p)


def _doc_to_product(d):
    m = d.metadata or {}
    return {"name": m.get("name", ""), "price_m": m.get("price_num"), "cpu": m.get("cpu", ""),
            "ram": m.get("ram_gb"), "gpu": m.get("gpu_name", ""), "storage": "",
            "weight": "", "url": m.get("url", ""), "img": m.get("img", "")}


def _model_tokens(name: str):
    toks = re.findall(r"[A-Za-z0-9]{4,}", name or "")
    return {t.upper() for t in toks if re.search(r"[A-Za-z]", t) and re.search(r"\d", t)}


def _select_products(answer: str, max_n: int = 4):
    ans = (answer or "").upper()
    matched = [p for p in _retrieved_products
               if any(tok in ans for tok in _model_tokens(p.get("name", "")))]
    chosen = matched if matched else _retrieved_products[:max_n]
    return chosen[:max_n]


# ══════════════════════════════════════════════════════════════
# ① QUERY UNDERSTANDING — LLM bóc ràng buộc cứng
# ══════════════════════════════════════════════════════════════
class Constraints(BaseModel):
    price_max: float | None = Field(None, description="Giá tối đa (TRIỆU đồng). '25 củ'/'dưới 25'→25")
    price_min: float | None = Field(None, description="Giá tối thiểu (triệu)")
    ram_min: int | None = Field(None, description="RAM tối thiểu (GB). 'RAM to/lớn'→16")
    brand_include: str | None = Field(None, description="Chỉ lấy hãng này (lowercase), nếu user chỉ định")
    brand_exclude: list[str] = Field(default_factory=list, description="Các hãng LOẠI TRỪ (lowercase), vd 'né Acer'→['acer']")


def extract_constraints(model, query) -> Constraints:
    try:
        return model.with_structured_output(Constraints).invoke([
            SystemMessage(content="Bóc ràng buộc CỨNG từ câu hỏi mua laptop. Giá đơn vị TRIỆU "
                          "('25 củ đổ lại'→price_max=25, 'dưới 20'→price_max=20). 'né/không/tránh "
                          "hãng X'→brand_exclude. 'RAM to/lớn'→ram_min=16. Không có thì để trống."),
            HumanMessage(content=query)])
    except Exception:
        return Constraints()


def apply_constraints(docs, c: Constraints):
    out = []
    for d in docs:
        m = d.metadata or {}
        price = m.get("price_num") or 0
        ram = m.get("ram_gb") or 0
        brand = (m.get("brand") or "").lower()
        if c.price_max and price > c.price_max:
            continue
        if c.price_min and price < c.price_min:
            continue
        if c.ram_min and ram < c.ram_min:
            continue
        if c.brand_include and c.brand_include.lower() not in brand:
            continue
        if any((b or "").lower() in brand for b in (c.brand_exclude or []) if b):
            continue
        out.append(d)
    return out


# ══════════════════════════════════════════════════════════════
# ④ TƯ VẤN — LLM đọc docs làm ngữ cảnh, giải thích 'tại sao'
# ══════════════════════════════════════════════════════════════
ADVISOR_SYSTEM = (
    "Bạn là trợ lý tư vấn laptop cho sinh viên Việt Nam. CHỈ gợi ý máy CÓ trong DANH SÁCH ỨNG VIÊN "
    "cho sẵn (không bịa máy ngoài danh sách). Chọn 2-3 máy hợp nhất, nêu TÊN + giá + cấu hình và "
    "GIẢI THÍCH ngắn vì sao hợp nhu cầu (CPU/GPU/RAM/màn/cân nặng) kèm đánh đổi. Thân thiện, tiếng Việt."
)


def _advise(model, query, docs, chat_history):
    ctx = "\n\n".join(
        f"[{i+1}] {d.metadata.get('name','')} | {d.metadata.get('price_num','?')} triệu | "
        f"RAM {d.metadata.get('ram_gb','?')}GB | GPU {d.metadata.get('gpu_name','') or 'tích hợp'} | "
        f"{d.metadata.get('url','')}\n{d.page_content[:400]}"
        for i, d in enumerate(docs))
    messages = [SystemMessage(content=ADVISOR_SYSTEM), *chat_history,
                HumanMessage(content=f"NHU CẦU: {query}\n\nDANH SÁCH ỨNG VIÊN:\n{ctx}\n\nHãy tư vấn.")]
    return (model.invoke(messages).content or "").strip()


# ══════════════════════════════════════════════════════════════
# RUN — luồng RAG cố định
# ══════════════════════════════════════════════════════════════
def run_agent(user_message: str, chat_history: list):
    """query → (constraints) → LUÔN retrieve+CRAG → lọc ràng buộc → tư vấn. Trả (answer, products)."""
    _retrieved_products.clear()

    cons = extract_constraints(model, user_message)
    has_signal = any([cons.price_max, cons.price_min, cons.ram_min, cons.brand_include, cons.brand_exclude])
    docs, clarify = corrective_retrieve(model, _retriever, user_message, k=12)

    # ② CRAG bảo mơ hồ → HỎI LẠI, nhưng CHỈ khi query thực sự trống (không bóc được ràng buộc nào).
    #    Có ràng buộc (giá/RAM/hãng) = query đã đủ cụ thể → không hỏi lại oan (tránh grader nhiễu).
    if clarify and not has_signal:
        return clarify, []

    # ③ lọc ràng buộc cứng (gồm loại hãng)
    docs = apply_constraints(docs, cons)
    if not docs:
        return ("Mình chưa tìm được máy khớp đủ ràng buộc (giá/RAM/hãng) trong nhóm phù hợp nhất. "
                "Bạn nới ngân sách hoặc bớt bớt điều kiện giúp mình nhé?"), []
    docs = docs[:6]
    for d in docs:
        _add_product(_doc_to_product(d))

    # ④ tư vấn
    answer = _advise(model, user_message, docs, chat_history)
    return answer, _select_products(answer)


# ══════════════════════════════════════════════════════════════
# TEST CLI
# ══════════════════════════════════════════════════════════════
def handle_conversation():
    chat_history = []
    print("Chatbot tư vấn laptop — gõ 'q' để thoát")
    while True:
        user_input = input("\nBạn: ").strip()
        if user_input.lower() == "q":
            break
        if not user_input:
            continue
        response, products = run_agent(user_input, chat_history)
        print(f"\nBot: {response}")
        if products:
            print("\n[Sản phẩm liên quan]")
            for p in products:
                print(f"  - {p['name']} | {p.get('price_m','?')} triệu")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        chat_history = chat_history[-10:]


if __name__ == "__main__":
    handle_conversation()
