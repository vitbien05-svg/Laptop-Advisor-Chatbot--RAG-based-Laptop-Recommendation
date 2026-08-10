"""
app.py — Streamlit Chatbot UI
Laptop Advisor cho sinh viên Việt Nam
"""

import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from pymongo import MongoClient
import pandas as pd
import math

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG — phải là lệnh Streamlit đầu tiên
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Tư vấn Laptop",
    page_icon="💻",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }
#MainMenu, header, footer { visibility: hidden; }
.stApp { background: #0f1117; }
.main .block-container { padding-top: 1rem; padding-bottom: 6rem; max-width: 760px; }
.chat-header { text-align: center; padding: 2rem 0 1.5rem; }
.chat-header h1 { font-size: 1.6rem; font-weight: 600; color: #f0f0f0; margin: 0; letter-spacing: -0.5px; }
.chat-header p { color: #6b7280; font-size: 0.875rem; margin: 0.3rem 0 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #1e2029 !important; border-radius: 16px !important;
    border: 1px solid #2a2d3a !important; padding: 0.75rem 1rem !important; margin-bottom: 0.5rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: transparent !important; border-radius: 16px !important;
    padding: 0.75rem 1rem !important; margin-bottom: 0.5rem !important;
}
[data-testid="stChatMessageAvatarUser"] { background: #3b5bdb !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: #1e2029 !important; border: 1px solid #2a2d3a !important; }
[data-testid="stChatInput"] {
    background: #1e2029 !important; border: 1px solid #2a2d3a !important;
    border-radius: 14px !important; color: #f0f0f0 !important;
}
[data-testid="stChatInput"]:focus-within { border-color: #3b5bdb !important; box-shadow: 0 0 0 3px rgba(59,91,219,0.15) !important; }
.typing-indicator { display: flex; gap: 5px; align-items: center; padding: 4px 0; }
.typing-indicator span { width: 7px; height: 7px; border-radius: 50%; background: #3b5bdb; animation: bounce 1.2s infinite; }
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,60%,100%{transform:translateY(0);opacity:0.5} 30%{transform:translateY(-6px);opacity:1} }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 4px; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# LOAD MODEL & RETRIEVER — chỉ chạy 1 lần, cache lại
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ Đang khởi động AI...")
def load_resources():
    """
    Load ChromaDB từ disk (không embed lại).
    Nếu chưa có vectorstore thì build từ MongoDB.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    import os

    if os.path.exists("./vectorstore"):
        # Load vectorstore đã build sẵn — nhanh
        vectorstore = Chroma(
            persist_directory="./vectorstore",
            embedding_function=embeddings,
            collection_name="langchain",  # tên collection mặc định của Chroma
        )
    else:
        # Build lần đầu từ MongoDB (mất vài phút)
        from langchain_core.documents import Document

        def _safe(val, default=""):
            if val is None:
                return default
            if isinstance(val, float) and math.isnan(val):
                return default
            return val

        def create_laptop_document(row):
            parts = []
            name, brand = row.get("name_product"), row.get("brand")
            if pd.notnull(name) and pd.notnull(brand):
                parts.append(f"Laptop {name} thương hiệu {brand}.")
            price = row.get("price")
            if pd.notnull(price) and price > 0:
                parts.append(f"Giá bán: {price/1_000_000:.2f} triệu VNĐ.")
            perf = []
            if pd.notnull(row.get("cpu_name")):
                perf.append(f"Chip xử lí {row.get('cpu_name')}")
            if pd.notnull(row.get("ram_info")):
                perf.append(f"RAM {row.get('ram_info')}GB {row.get('ram_type','')}")
            if pd.notnull(row.get("gpu_type")):
                perf.append(f"card {row.get('gpu_type')}")
            if pd.notnull(row.get("gpu_ram")):
                perf.append(f"{row.get('gpu_ram')}GB VRAM")
            if pd.notnull(row.get("gpu_name")):
                perf.append(row.get("gpu_name"))
            if pd.notnull(row.get("storage_capacity")):
                perf.append(row.get("storage_capacity"))
            if perf:
                parts.append("Cấu hình: " + ", ".join(perf) + ".")
            disp = []
            if pd.notnull(row.get("display_size")):
                disp.append(f"{row.get('display_size')}")
            if pd.notnull(row.get("resolution")):
                disp.append(row.get("resolution"))
            if pd.notnull(row.get("panel_type")):
                disp.append(row.get("panel_type"))
            if pd.notnull(row.get("refresh_rate")):
                disp.append(f"{int(row['refresh_rate'])}Hz")
            if disp:
                parts.append("Màn hình: " + ", ".join(disp) + ".")
            design = []
            if pd.notnull(row.get("weight_kg")):
                design.append(f"{row.get('weight_kg')}kg")
            if pd.notnull(row.get("release_year")):
                design.append(f"ra mắt {int(row['release_year'])}")
            if design:
                parts.append("Thiết kế: " + ", ".join(design) + ".")
            metadata = {
                "id": str(row.get("id")),
                "brand": str(brand).lower(),
                "price_num": float(price / 1_000_000) if pd.notnull(price) else 0,
                "ram_gb": (
                    int(row.get("ram_info", 0))
                    if pd.notnull(row.get("ram_info"))
                    else 0
                ),
                "url": str(row.get("url_product", "")),
                "img": str(row.get("img_product", "")),
                "gpu_name": str(row.get("gpu_name", "")),
            }
            return Document(page_content=" ".join(parts), metadata=metadata)

        client = MongoClient("mongodb://localhost:27017/")
        data = pd.DataFrame(
            list(client["LaptopDataDB"]["laptop_cleaned"].find({}, {"_id": 0}))
        )
        client.close()
        docs = [create_laptop_document(row) for _, row in data.iterrows()]
        vectorstore = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory="./vectorstore",
            ids=[d.metadata["id"] for d in docs],
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # LLM + chains
    model = OllamaLLM(model="qwen2.5:3b")

    template = """
Bạn là một người tư vấn mua laptop cho sinh viên tại Việt Nam.
Hãy cho ra câu trả lời ngắn gọn. Khi người dùng hỏi thông tin chi tiết về máy thì mới đưa ra 
Quy tắc:
1. Chỉ tư vấn dựa trên DANH SÁCH LAPTOP bên dưới
2. Hãy tư vấn nhiệt tình về sản phẩm (hãy đưa ra giá bán của máy)
3. Nếu không tìm thấy → hỏi thêm ( chỉ hỏi thêm 3 câu là hết cỡ)
4. Luôn cho user biết giá của sản phẩm + cấu hình máy
--- DANH SÁCH LAPTOP ---
{rag_context}
-- Lịch sử câu trả lời --
{chat_history}
Câu hỏi: {question}
Trả lời:"""


    return retriever, chain_answer


def format_docs(docs):
    if not docs:
        return "Không tìm thấy laptop phù hợp."
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1))


# ══════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════
retriever, chain_answer = load_resources()

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = ""

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="chat-header">
    <h1>💻 Tư vấn Laptop</h1>
    <p>Trợ lý AI giúp sinh viên chọn laptop phù hợp · 435 mẫu cập nhật 2025</p>
</div>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
# GỢI Ý BAN ĐẦU
# ══════════════════════════════════════════════════════════════
SUGGESTIONS = [
    "💰 Laptop dưới 15 triệu cho sinh viên",
    "🎮 Laptop gaming card rời RTX",
    "🪶 Laptop mỏng nhẹ dưới 1.5kg",
    "🎨 Laptop đồ họa màn hình đẹp",
]

if not st.session_state.messages:
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state._suggestion = s

# ══════════════════════════════════════════════════════════════
# HIỂN THỊ LỊCH SỬ
# ══════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ══════════════════════════════════════════════════════════════
# XỬ LÝ INPUT
# ══════════════════════════════════════════════════════════════
user_input = st.chat_input("Bạn cần tư vấn laptop như thế nào?")

if not user_input and hasattr(st.session_state, "_suggestion"):
    user_input = st.session_state._suggestion
    del st.session_state._suggestion

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            """
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>""",
            unsafe_allow_html=True,
        )

        docs = retriever.invoke(user_input)
        rag_context = format_docs(docs)
        response = chain_answer.invoke(
            {
                "rag_context": rag_context,
                "chat_history": st.session_state.history,
                "question":    user_input,
            }
        )

        placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history += f"\nUser: {user_input}\nAI: {response}"

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Laptop Advisor")
    st.markdown(
        "Trợ lý tư vấn laptop cho sinh viên Việt Nam, make by Việt sama Studio."
    )
    st.divider()
    if st.button("🗑️ Delete history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = ""
        st.rerun()
    st.markdown("---")
    st.caption("Data:From  Thế Giới Di Động · 435 sample")
