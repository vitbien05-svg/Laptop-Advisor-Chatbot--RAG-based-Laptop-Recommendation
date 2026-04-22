from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_database import retriever

template = """
Bạn là một người tư vấn mua laptop cho sinh viên tại Việt Nam.
Hãy cho ra câu trả lời ngắn gọn. Khi người dùng hỏi thông tin chi tiết về máy thì mới đưa ra 
Quy tắc:
1. Chỉ tư vấn dựa trên DANH SÁCH LAPTOP bên dưới
2. Hãy tư vấn nhiệt tình về sản phẩm (hãy đưa ra giá bán của máy)
3. Nếu không tìm thấy → hỏi thêm ( chỉ hỏi thêm 3 câu là hết cỡ)
4. Luôn cho user biết giá của sản phẩm + cấu hình máy
5. Phải kế thừa yêu cầu trước đó (ngân sách, nhu cầu)
6. Không được mâu thuẫn với yêu cầu trước
--- DANH SÁCH LAPTOP ---
{rag_context}
-- Lịch sử câu trả lời --
-Trước khi đưa ra câu trả lời. Hãy thật tập trung xem xét lịch xử câu trả lời trước khi đưa ra:
{chat_history}

Câu hỏi: {question}
Trả lời:
"""


recap_template = """
Bạn là AI giúp hiểu yêu cầu người dùng. Mục đích là tóm tắt lại văn bản về tư vấn máy tính 

Dựa vào lịch sử hội thoại, hãy viết lại yêu cầu hiện tại thành 1 đoạn văn ngắn đầy đủ.

Yêu cầu:
-Hãy giữ lại tất cả thông tin quan trọng về máy tính, nhu cầu người dùng, mục đích
- Không thêm thông tin mới
- Viết đầy đủ thông tin máy tính, nhu cầu người dùng 
- giữ lại yêu cầu, mong muốn của người dùng về sản phẩm 

History:
{chat_history}
User input:
{question}
Full query:

"""
prompt_answer = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="qwen2.5:3b")
chain_answer = prompt_answer | model

prompt_recap = ChatPromptTemplate.from_template(recap_template)
chain_recap = prompt_recap | model


def format_docs(docs):
    if not docs:
        return "Không tìm thấy laptop phù hợp."
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1))


def handle_conversation():
    history = ""
    print("Welcome to the AI chatbot, Type 'q' to quit")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "q":
            break
        full_query = chain_recap.invoke(
            {"chat_history": history, "question": user_input}
        )
        docs = retriever.invoke(full_query)
        rag_context = format_docs(docs)
        result = chain_answer.invoke(
            {
                "rag_context": rag_context,
                "chat_history": history,
                "question": full_query,
            }
        )
        print("Bot: ", result)
        history += f"\nUser: {user_input}\nAI: {result}"


if __name__ == "__main__":
    handle_conversation()
