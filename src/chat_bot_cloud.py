
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import os
import re

# ══════════════════════════════════════════════════════════════
# KHỞI TẠO MODEL & VECTORSTORE
# ══════════════════════════════════════════════════════════════

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6,
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embeddings,
    collection_name="langchain",
)

# Hybrid retriever: BM25 + dense + RRF + cross-encoder rerank (reranker lazy-load lần đầu)
from retrieval import HybridRetriever
from crag import corrective_retrieve
_retriever = HybridRetriever(vectorstore)

# ── Gom sản phẩm được truy xuất trong 1 lượt hỏi (để UI show ảnh) ──
_retrieved_products: list = []


def _add_product(p: dict):
    """Thêm sản phẩm (dedup theo url)."""
    if p.get("url") and all(p["url"] != x.get("url") for x in _retrieved_products):
        _retrieved_products.append(p)


def _model_tokens(name: str):
    """Trích mã model/SKU đặc trưng (vừa chữ vừa số, ≥4 ký tự) để khớp với câu trả lời."""
    toks = re.findall(r"[A-Za-z0-9]{4,}", name or "")
    return {t.upper() for t in toks if re.search(r"[A-Za-z]", t) and re.search(r"\d", t)}


def _select_products(answer: str, max_n: int = 4):
    """Chọn sản phẩm để show ảnh: ưu tiên máy được NHẮC trong câu trả lời; nếu không có → top vài máy."""
    ans = (answer or "").upper()
    matched = [p for p in _retrieved_products
               if any(tok in ans for tok in _model_tokens(p.get("name", "")))]
    chosen = matched if matched else _retrieved_products[:max_n]
    return chosen[:max_n]

# ══════════════════════════════════════════════════════════════
# TOOL 1: QUERY MONGODB — lọc có cấu trúc
# ══════════════════════════════════════════════════════════════

@tool
def query_mongodb(
    price_min: float = None,
    price_max: float = None,
    brand: str = None,
    ram_min: int = None,
    gpu_type: str = None,
    limit: int = 8,
    weight_min: float = None,
    weight_max: float = None,
) -> str:
    """
    Truy vấn MongoDB để lấy danh sách laptop theo các tiêu chí cụ thể.
    Dùng khi người dùng có yêu cầu về GIÁ, THƯƠNG HIỆU, RAM, hoặc LOẠI GPU rõ ràng.

    Args:
        price_min: Giá tối thiểu (đơn vị: TRIỆU đồng, ví dụ 10 = 10 triệu)
        price_max: Giá tối đa (đơn vị: TRIỆU đồng, ví dụ 20 = 20 triệu)
        brand: Thương hiệu laptop (ví dụ: "asus", "dell", "hp", "lenovo", "acer", "msi")
        ram_min: RAM tối thiểu (GB, ví dụ 16)
        gpu_type: Loại GPU - dùng "card rời" hoặc "card tích hợp"
        limit: Số lượng kết quả tối đa (mặc định 8)
        weight_min: Trọng lượng tối thiểu (Đơn vị: kg, ví dụ 1.5 = 1.5kg)
        weight_max: Trọng lượng tối đa (Đơn vị: kg, ví dụ 2.0 = 2.0kg) 
    """
    try:
        client = MongoClient("mongodb://localhost:27017/")
        collection = client["LaptopDataDB"]["laptop_cleaned"]

        query = {}

        # Lọc giá (đổi từ triệu → VNĐ)
        price_filter = {}
        if price_min is not None:
            price_filter["$gte"] = price_min * 1_000_000
        if price_max is not None:
            price_filter["$lte"] = price_max * 1_000_000
        if price_filter:
            query["price"] = price_filter

        #Query theo cân nặng:
        weight_filter = {}
        if weight_min is not None:
            weight_filter["$gte"] = weight_min
        if weight_max is not None:
            weight_filter["$lte"] = weight_max
        if weight_filter:
            query["weight_kg"] = weight_filter 
        # Lọc thương hiệu (case-insensitive)
        if brand:
            query["brand"] = {"$regex": brand.strip(), "$options": "i"}

        # Lọc RAM
        if ram_min is not None:
            query["ram_info"] = {"$gte": ram_min}

        # Lọc GPU
        if gpu_type:
            if "rời" in gpu_type.lower():
                query["gpu_type"] = {
                    "$regex": "card rời|dedicated|nvidia|amd radeon rx|rtx|rx",
                    "$options": "i",
                }
            elif "tích hợp" in gpu_type.lower():
                query["gpu_type"] = {
                    "$regex": "tích hợp|integrated|intel|uhd|iris",
                    "$options": "i",
                }

        results = list(collection.find(query, {"_id": 0}).limit(limit))
        client.close()

        if not results:
            return "Không tìm thấy laptop nào phù hợp với tiêu chí đã lọc."

        output_lines = [f"Tìm thấy {len(results)} laptop phù hợp:\n"]
        for i, lap in enumerate(results, 1):
            price_vnd = lap.get("price", 0)
            price_m = price_vnd / 1_000_000 if price_vnd else 0
            name = lap.get("name_product", "N/A")
            brand_val = lap.get("brand", "N/A")
            cpu = lap.get("cpu_name", "N/A")
            ram = lap.get("ram_info", "N/A")
            gpu = lap.get("gpu_name", lap.get("gpu_type", "N/A"))
            storage = lap.get("storage_capacity", "N/A")
            url = lap.get("url_product", "")
            weight_kg = lap.get("weight_kg", "N/A")
            _add_product({
                "name": name, "price_m": round(price_m, 1) if price_m else None,
                "cpu": cpu, "ram": ram, "gpu": gpu, "storage": storage,
                "weight": weight_kg, "url": url, "img": lap.get("img_product", ""),
            })
            output_lines.append(
                f"[{i}] {name} ({brand_val})\n"
                f"Giá: {price_m:.1f} triệu VNĐ\n"
                f"CPU: {cpu} | RAM: {ram}GB | GPU: {gpu} | SSD: {storage}\n"
                f"Cân Nặng: {weight_kg}\n"
                f"Link: {url}\n"
            )
        return "\n".join(output_lines)

    except Exception as e:
        return f"Lỗi khi truy vấn MongoDB: {str(e)}"


# ══════════════════════════════════════════════════════════════
# TOOL 2: SEMANTIC SEARCH — tìm kiếm theo ngữ nghĩa
# ══════════════════════════════════════════════════════════════

@tool
def search_vector(query: str, k: int = 5) -> str:
    """
    Tìm kiếm laptop theo ngữ nghĩa dùng ChromaDB vector search.
    câu hỏi không có filter cụ thể về giá/thương hiệu.

    Args:
        query: Từ khóa tìm kiếm (ví dụ: "laptop mỏng nhẹ cho lập trình")
        k: Số lượng kết quả (mặc định 5)
    """
    try:
        # CRAG: hybrid retrieval → chấm điểm → sửa (rewrite) → hỏi lại nếu vẫn kém
        docs, clarify = corrective_retrieve(model, _retriever, query, k=k)
        if not docs:
            return "Không tìm thấy laptop phù hợp qua semantic search."
        lines = [f"Tìm thấy {len(docs)} laptop liên quan:\n"]
        for i, doc in enumerate(docs, 1):
            m = doc.metadata or {}
            _add_product({
                "name": m.get("name", ""), "price_m": m.get("price_num"),
                "cpu": m.get("cpu", ""), "ram": m.get("ram_gb"), "gpu": m.get("gpu_name", ""),
                "storage": "", "weight": "", "url": m.get("url", ""), "img": m.get("img", ""),
            })
            lines.append(f"[{i}] {doc.page_content[:400]}\n")
        if clarify:
            lines.append(
                f"\n[LƯU Ý cho trợ lý: retrieval chưa chắc khớp nhu cầu. Hãy HỎI LẠI người dùng: {clarify}]"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Lỗi semantic search: {str(e)}"


# ══════════════════════════════════════════════════════════════
# AGENT EXECUTOR — tool-calling loop
# ══════════════════════════════════════════════════════════════

TOOLS = [query_mongodb, search_vector]
TOOLS_MAP = {t.name: t for t in TOOLS}

model_with_tools = model.bind_tools(TOOLS)

SYSTEM_PROMPT = """Bạn là trợ lý tư vấn mua laptop cho sinh viên Việt Nam. Bạn có 2 công cụ:

1. **query_mongodb**: Dùng khi user hỏi có GIÁ LAPTOP, THƯƠNG HIỆU, RAM, hoặc loại GPU.
   - Ví dụ: "laptop dưới 20 triệu", "laptop Dell RAM 16GB", "laptop gaming card rời"

2. **search_vector**: Dùng khi mà người dùng hỏi những vấn đề khác ví dụ như màn hình, camera máy...
   - Ví dụ: "laptop có màn 120hz", "laptop có camera 2k"

Quy tắc:
- Uư tiên tìm kiếm bằng tool query_mongodb trước( những vấn đề kĩ thuật như giá, ram, card đồ họa), Những đặc điểm có tính chất mô tả (màu sắc, độ bền, chất lượng camera, nhu cầu sử dụng), hãy dùng search_vector."
- Trả lời bằng tiếng Việt, ngắn gọn, thân thiện, không cần dùng emoji
- Luôn đề cập giá và cấu hình khi tư vấn
- Nếu vẫn không tìm thấy, hãy hỏi họ lại bạn có con máy khác không
- Kế thừa ngân sách và nhu cầu từ lịch sử hội thoại"""


def run_agent(user_message: str, chat_history: list) -> str:
    """
    Chạy agent với tool-calling loop.

    Args:
        user_message: Tin nhắn hiện tại của user
        chat_history: Danh sách LangChain message objects (HumanMessage / AIMessage)

    Returns:
        Chuỗi phản hồi cuối cùng của AI
    """
    _retrieved_products.clear()
    messages = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + chat_history
        + [HumanMessage(content=user_message)]
    )

    answer = None
    for _ in range(5):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            answer = response.content
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name in TOOLS_MAP:
                result = TOOLS_MAP[tool_name].invoke(tool_args)
            else:
                result = f"Tool '{tool_name}' không tồn tại."

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

    if answer is None:
        answer = "Xin lỗi, tôi không thể tìm được kết quả phù hợp. Bạn có thể mô tả lại nhu cầu không?"

    products = _select_products(answer)
    return answer, products


# ══════════════════════════════════════════════════════════════
# TEST CLI (chạy trực tiếp file này để test nhanh)
# ══════════════════════════════════════════════════════════════

def handle_conversation():
    chat_history = []
    print("Chatbot")
    print("Gõ 'q' để thoát")
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
                print(f"  - {p['name']} | {p.get('price_m','?')} triệu | {p.get('img','')[:60]}")

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]


if __name__ == "__main__":
    handle_conversation()
