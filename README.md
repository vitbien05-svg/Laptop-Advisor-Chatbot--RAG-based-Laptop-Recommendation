# 💻 Laptop Advisor Chatbot — Recommendation RAG for Vietnamese Students

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-🦜-green" alt="LangChain">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/MongoDB-4EA94B?logo=mongodb&logoColor=white" alt="MongoDB">
  <img src="https://img.shields.io/badge/ChromaDB-Vector_DB-purple" alt="ChromaDB">
  <img src="https://img.shields.io/badge/RAGAS-eval-orange" alt="RAGAS">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

A **RAG** chatbot that advises Vietnamese students on which laptop to buy, by need and budget.
It's a **recommendation** problem (many valid answers), not factual QA — which drives the retrieval
and evaluation design. The generation path is a **fixed RAG flow** (always retrieve → grade → advise),
not a tool-routing agent, so retrieval and the corrective step can never be bypassed.

---

## 🧩 Techniques

- **Query understanding** — an LLM parses the message into **hard constraints** (price / RAM / brand,
  including *exclusions* like "not Acer") used as a deterministic filter.
- **Fixed RAG flow** — every query **always retrieves**, gets graded by CRAG, then is answered. No agent
  decides whether to retrieve, so the corrective loop is never skipped.
- **Hybrid retrieval + RRF** — BM25 (`bm25s`, sparse inverted index — exact CPU/GPU codes) fused with
  dense embeddings (HNSW/ANN) via Reciprocal Rank Fusion.
- **Cross-encoder reranking** — `bge-reranker-v2-m3` (local, free) re-scores the ~20 candidates → top-k.
- **CRAG with an LLM grader** — grades retrieval as **correct / ambiguous / incorrect**; *ambiguous* →
  **asks a clarifying question**, *incorrect* → **rewrites the query** and retries, instead of answering blindly.
- **Data enrichment** — rule-based **use-case taxonomy** (deterministic, no hallucination) + LLM-distilled
  **need summaries** (marketing hype stripped, grounded in specs), so documents speak the user's "need language".

**Stack:** Playwright · MongoDB · Pandas · OpenAI `text-embedding-3-small` · ChromaDB · LangChain · RAGAS · Streamlit.

---

## 🏗️ Pipeline

```
Crawl (Playwright + BS4) → MongoDB → Clean (pandas+regex)
   → Enrich (taxonomy rules + LLM summary) → Embed (OpenAI) → ChromaDB

User query
   → ① Query understanding (LLM → price / RAM / brand incl. exclude)
   → ② Always retrieve: Hybrid (BM25 + dense) → RRF → Cross-encoder rerank
   → ③ CRAG (LLM grader): correct → use | ambiguous → ask | incorrect → rewrite & retry
   → ④ Apply hard constraints (deterministic filter, incl. brand exclusion)
   → ⑤ LLM advises: recommends machines + explains WHY + trade-offs (product cards)
```

---

## 📊 Evaluation (two layers)

**Retrieval** — 24 self-written gold queries (need / constraint / combined), auto-scored from metadata.

| Query type | Mode | P@5 | NDCG@5 |
|---|---|---|---|
| Need | dense-only | 0.47 | 0.47 |
| Need | **hybrid + rerank** | **0.93** | **0.91** |
| Combined (tag + constraint) | hybrid + rerank | 0.44 | 0.43 |

**Generation** — RAGAS (`Faithfulness`, `Answer Relevancy`) + a deterministic groundedness check
(recommended machines must exist in the retrieved context). Context precision/recall are intentionally
**omitted** — retrieval is already measured above, and recall needs a gold answer a recommendation task lacks.

| Metric | Score |
|---|---|
| Faithfulness (RAGAS) | 0.88 |
| Answer Relevancy (RAGAS) | 0.55 |
| Groundedness (deterministic, no hallucinated machine) | 1.00 |

Run: `python eval/eval_min.py` (retrieval) · `python eval/eval_ragas.py` (generation) ·
`python eval/test_hard_cases.py` (CRAG behaviour on tricky queries).

---

## 🗂️ Structure

```
src/         # crawl → clean → taxonomy → enrich → embed → retrieve → CRAG → advise → UI
eval/        # eval_min (retrieval), eval_generation + eval_ragas (generation), test_hard_cases (CRAG)
data/        # raw crawl snapshot + taxonomy validation sample
notebooks/   # cleaning notebook + early prototypes
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

> A fully-local variant (Ollama) exists via `src/chat_bot.py` / `src/user_interface.py` —
> no API key, but weaker quality.

---

**Source:** Thế Giới Di Động · ~435 laptop models · for educational/research use.
