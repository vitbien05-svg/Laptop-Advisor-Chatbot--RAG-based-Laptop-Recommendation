"""
user_interface_cloud.py — Streamlit UI dùng AI Agent (OpenAI + Tool Use)
"""

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from chat_bot_cloud import run_agent


def render_products(products):
    """Hiển thị card sản phẩm có ẢNH + giá + cấu hình + link."""
    if not products:
        return
    st.markdown("**🛒 Sản phẩm gợi ý**")
    cols = st.columns(min(len(products), 2))
    for i, p in enumerate(products):
        with cols[i % len(cols)]:
            if p.get("img"):
                st.image(p["img"], use_container_width=True)
            st.markdown(f"**{p.get('name', '')}**")
            if p.get("price_m"):
                st.markdown(f"💰 **{p['price_m']} triệu**")
            specs = " · ".join(
                str(x) for x in [
                    p.get("cpu"),
                    f"{p.get('ram')}GB RAM" if p.get("ram") else None,
                    p.get("gpu"),
                ] if x
            )
            if specs:
                st.caption(specs)
            if p.get("url"):
                st.markdown(f"[🔗 Xem chi tiết]({p['url']})")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Tư vấn Laptop AI",
    page_icon="💻",
    layout="centered",
)


# Nạp reranker (~2GB) MỘT LẦN lúc khởi động, có spinner rõ ràng (thay vì treo trắng ở câu đầu)
@st.cache_resource(show_spinner="🔧 Đang tải AI engine lần đầu (~30s, chỉ 1 lần)...")
def _warmup_engine():
    from retrieval import _get_reranker
    _get_reranker()
    return True


_warmup_engine()

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }
#MainMenu, header, footer { visibility: hidden; }
.stApp { background: #0f1117; }
.main .block-container { padding-top: 1rem; padding-bottom: 6rem; max-width: 760px; }
.chat-header { text-align: center; padding: 2rem 0 1.5rem; }
.chat-header h1 { font-size: 1.6rem; font-weight: 600; color: #f0f0f0; margin: 0; letter-spacing: -0.5px; }
.chat-header p { color: #6b7280; font-size: 0.875rem; margin: 0.3rem 0 0; }
.agent-badge {
    display: inline-block; background: linear-gradient(135deg, #3b5bdb, #9775fa);
    color: white; font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
    border-radius: 20px; margin-left: 8px; vertical-align: middle; letter-spacing: 0.5px;
}
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
[data-testid="stChatInput"]:focus-within { border-color: #9775fa !important; box-shadow: 0 0 0 3px rgba(151,117,250,0.15) !important; }
.typing-indicator { display: flex; gap: 5px; align-items: center; padding: 4px 0; }
.typing-indicator span { width: 7px; height: 7px; border-radius: 50%; background: #9775fa; animation: bounce 1.2s infinite; }
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,60%,100%{transform:translateY(0);opacity:0.5} 30%{transform:translateY(-6px);opacity:1} }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="chat-header">
    <h1>💻 Tư vấn Laptop <span class="agent-badge">AI AGENT</span></h1>
    <p>Trợ lý thông minh · Tự động tìm kiếm MongoDB + Semantic Search · 435 mẫu 2025</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lc_history" not in st.session_state:
    st.session_state.lc_history = []  # LangChain message objects

# ══════════════════════════════════════════════════════════════
# GỢI Ý BAN ĐẦU
# ══════════════════════════════════════════════════════════════
SUGGESTIONS = [
    "💰 Laptop dưới 15 triệu cho sinh viên",
    "🎮 Laptop gaming card rời RTX dưới 25 triệu",
    "🪶 Laptop mỏng nhẹ thương hiệu LG hoặc Dell",
    "🎨 Laptop đồ họa màn hình đẹp RAM 16GB",
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
        if msg.get("products"):
            render_products(msg["products"])

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
        placeholder.markdown("""
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>""", unsafe_allow_html=True)

        # Gọi AI Agent
        response, products = run_agent(user_input, st.session_state.lc_history)
        placeholder.markdown(response)
        render_products(products)

    # Lưu lịch sử
    st.session_state.messages.append({"role": "assistant", "content": response, "products": products})
    st.session_state.lc_history.append(HumanMessage(content=user_input))
    st.session_state.lc_history.append(AIMessage(content=response))

    # Giữ tối đa 10 lượt gần nhất
    if len(st.session_state.lc_history) > 20:
        st.session_state.lc_history = st.session_state.lc_history[-20:]

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 💻 Laptop Advisor")
    st.markdown("Trợ lý AI tư vấn laptop cho sinh viên Việt Nam.")
    st.divider()

    st.markdown("**⚙️ AI Engine**")
    st.caption("🤖 Model: GPT-4o-mini")
    st.caption("🔍 Tool 1: MongoDB Query (structured filter)")
    st.caption("🧠 Tool 2: ChromaDB Semantic Search")
    st.caption("📦 Embedding: text-embedding-3-small")

    st.divider()
    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.rerun()

    st.markdown("---")
    st.caption("Data: Thế Giới Di Động · 435 sản phẩm")
