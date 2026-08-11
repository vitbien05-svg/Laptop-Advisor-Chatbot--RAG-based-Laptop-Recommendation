"""
eval_ragas.py — Đánh giá GENERATION bằng framework RAGAS (không tự chế).

2 metric (bỏ context precision/recall vì retrieval đã đo riêng ở eval_min.py → tránh chồng chéo):
  - Faithfulness    : câu trả lời có bám ngữ cảnh không (RAGAS tách mệnh đề → kiểm NLI). Bắt bịa.
  - Answer Relevancy: câu trả lời có đúng trọng tâm câu hỏi không (RAGAS sinh câu hỏi → so cosine).

Luồng: mỗi câu goldset → chạy RAG thật (retrieve+CRAG+lọc+tư vấn) lấy (question, answer, contexts)
→ RAGAS chấm. Câu bị CRAG hỏi lại (mơ hồ) → bỏ qua (không có generation để chấm).

Chạy:  python eval/eval_ragas.py
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.chdir(_ROOT)
from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

import chat_bot_cloud as cb
from crag import corrective_retrieve

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def load_goldset(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    gold = load_goldset(os.path.join(os.path.dirname(__file__), "goldset.jsonl"))

    samples = []
    print(f"Sinh câu trả lời RAG cho {len(gold)} câu...")
    for item in gold:
        q = item["query"]
        cons = cb.extract_constraints(cb.model, q)
        has_signal = any([cons.price_max, cons.price_min, cons.ram_min,
                          cons.brand_include, cons.brand_exclude])
        docs, clarify = corrective_retrieve(cb.model, cb._retriever, q, k=12)
        if clarify and not has_signal:
            print(f"  (bỏ qua — CRAG hỏi lại) {q[:45]}")
            continue
        docs = cb.apply_constraints(docs, cons)[:6]
        if not docs:
            continue
        answer = cb._advise(cb.model, q, docs, [])
        samples.append(SingleTurnSample(
            user_input=q, response=answer,
            retrieved_contexts=[d.page_content for d in docs],
        ))
    print(f"Chấm RAGAS trên {len(samples)} câu (có generation)...")

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    ds = EvaluationDataset(samples=samples)
    ctx_precision = LLMContextPrecisionWithoutReference()  # context precision KHÔNG cần gold answer
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, ctx_precision],
                      llm=llm, embeddings=emb)
    print("\n=== KẾT QUẢ RAGAS ===")
    print(result)


if __name__ == "__main__":
    main()
