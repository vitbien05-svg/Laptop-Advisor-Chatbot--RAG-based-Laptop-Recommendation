# 💻 Laptop Advisor Chatbot — Agentic Recommendation RAG

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-🦜-green" alt="LangChain">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/MongoDB-4EA94B?logo=mongodb&logoColor=white" alt="MongoDB">
  <img src="https://img.shields.io/badge/ChromaDB-Vector_DB-purple" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

An **agentic RAG** chatbot that recommends laptops to Vietnamese students by need and budget.
It's a **recommendation** problem (many valid answers), not factual QA — which drives the retrieval design.

> 📚 Deep-dive (design decisions, evaluation numbers, why-each-choice): **[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)**

---

## 🧩 Techniques

- **Agentic routing** — an LLM agent (GPT-4o-mini) picks between `query_mongodb` (hard filters: price/RAM/brand/GPU) and `search_vector` (semantic need matching).
- **Hybrid retrieval + RRF** — BM25 (`bm25s`, sparse index — exact CPU/GPU codes) fused with dense embeddings via Reciprocal Rank Fusion.
- **Cross-encoder reranking** — `bge-reranker-v2-m3` (local, free) re-scores candidates for precision.
- **CRAG corrective loop** — gates on rerank score; if weak, rewrites the query or **asks a clarifying question** instead of answering blindly.
- **Data enrichment** — rule-based use-case tags (no hallucination) + LLM-distilled need summaries, so documents speak the user's "need language".

**Stack:** Playwright · MongoDB · Pandas · OpenAI `text-embedding-3-small` · ChromaDB · LangChain · Streamlit.

---

## 🏗️ Pipeline

```
Crawl (Playwright) → MongoDB (raw) → Clean (pandas+regex) → MongoDB (laptop_cleaned)
   → Enrich (rule tags + LLM summary) → Embed (OpenAI) → ChromaDB (vectors)

User query
   → Agent (GPT-4o-mini, tool-calling)
        ├─ query_mongodb → structured filter (price / RAM / brand / GPU)
        └─ search_vector → Hybrid (BM25 + dense) → RRF → Cross-encoder rerank → CRAG
   → Answer + product cards (with images)
```

---

## 🗂️ Structure

```
src/         # RAG pipeline: crawl → enrich → index → retrieve → agent → UI
eval/        # retrieval evaluation (Precision@k, NDCG@k)
data/        # raw crawl snapshot + taxonomy validation sample
notebooks/   # cleaning notebook + early prototypes
docs/        # deep-dive documentation
assets/      # architecture diagram + demo screenshot
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
playwright install chromium

cp .env.example .env          # then put your real OPENAI_API_KEY in .env

# Run the app (from the project root, so ./vectorstore resolves)
streamlit run src/user_interface_cloud.py     # → http://localhost:8501
```

Rebuild data from scratch (optional): `crawling_data.py` → `crawl_description.py` →
`clean_product.ipynb` → `taxonomy.py --write` → `enrich.py` → `vector_database.py`.
Evaluate retrieval: `python eval/eval_min.py`.

> A fully-local variant (Ollama) exists via `src/chat_bot.py` / `src/user_interface.py` —
> no API key, but weaker quality.

---

**Source:** Thế Giới Di Động · ~435 laptop models · for educational/research use.
