# Laptop Advisor Chatbot — Tài liệu Project (Recommendation RAG)

> Tài liệu này giải thích **toàn bộ** project: mục tiêu, kiến trúc, cách xử lý dữ liệu, cách
> triển khai từng thành phần, **vì sao** chọn các công nghệ/kỹ thuật đó, cách đánh giá và kết
> quả thu được. Mục đích: để hiểu sâu và trình bày được khi phỏng vấn.

---

## 1. Mục tiêu & bài toán

Chatbot **tư vấn mua laptop** cho sinh viên Việt Nam theo nhu cầu + ngân sách. Đây là bài toán
**recommendation** (gợi ý nhiều đáp án phù hợp), **không phải** QA factual (1 đáp án đúng). Điều
này chi phối mọi quyết định thiết kế phía sau (cách đánh giá, chọn kỹ thuật RAG...).

**Kiến trúc tổng thể (end-to-end):**
```
Crawl (Playwright)  →  MongoDB (raw)  →  Clean (pandas+regex)  →  MongoDB (laptop_cleaned)
      → Enrich (rule tags + LLM summary)  →  Embed → ChromaDB (vector)
Câu hỏi → Agent(LLM) ─┬─ query_mongodb (lọc cứng: giá/RAM/hãng)
                      └─ search_vector → Hybrid(BM25+dense)+RRF → Rerank → CRAG
                                       → Câu trả lời + card sản phẩm (ảnh)
```

---

## 2. Thu thập dữ liệu (Crawling)

- **Nguồn:** Thế Giới Di Động (TGDD) — 7 hãng, ~435 laptop.
- **Công cụ:** `Playwright` (headless browser) + `BeautifulSoup`.
- **Vì sao Playwright chứ không phải `requests`?** Trang TGDD render bằng **JavaScript**
  (nội dung nạp động, có nút "Xem thêm", lazy-load). `requests` chỉ lấy HTML tĩnh ban đầu →
  thiếu dữ liệu. Playwright chạy trình duyệt thật → thấy đúng nội dung sau khi JS chạy.
- **Chiến lược:** auto-scroll + click "Xem thêm" để lấy hết sản phẩm; vào từng trang chi tiết
  lấy bảng specs; **delay ngẫu nhiên 1–4s** tránh bị chặn (chống bot).
- **Bổ sung mô tả** (`crawl_description.py`): crawl thêm khối "Thông tin sản phẩm" (đoạn văn
  mô tả tự nhiên) ở selector `div.text-detail`. Viết **async, chạy song song 5 trang** → nhanh
  ~5x so với tuần tự. **Resume-able** (bỏ qua sản phẩm đã có), chặn tải ảnh/CSS/font cho nhanh.

---

## 3. Lưu trữ: **Vì sao MongoDB thay vì SQL?**

Đây là câu hay bị hỏi. Lý do chọn **MongoDB (NoSQL document)**:

1. **Dữ liệu bán cấu trúc, thưa (sparse), schema hay đổi.** Mỗi laptop có tập thông số KHÁC
   nhau: máy này có `color_gamut`, máy kia không; có máy công bố tần số quét, có máy không.
   → SQL sẽ đầy cột `NULL` hoặc phải `ALTER TABLE` liên tục. MongoDB cho phép mỗi document có
   field riêng, thêm/bớt tự do.
2. **Pipeline tiến hoá, thêm field liên tục.** Trong quá trình làm, ta **thêm dần** các field
   mới vào cùng collection: `description_raw`, `use_case_tags`, `use_case_summary`,
   `price_tier`... Với MongoDB chỉ cần `$set`, KHÔNG cần migration/đổi schema.
3. **JSON-native, khớp Python.** Document = dict Python = LangChain `Document`. Đọc/ghi thẳng,
   không cần ORM hay ánh xạ quan hệ.
4. **Không có quan hệ phức tạp.** Đây là **1 loại thực thể** (laptop), không cần JOIN nhiều bảng,
   không cần transaction phức tạp → thế mạnh của SQL không dùng tới.
5. **Prototype nhanh**: crawl → lưu → sửa → lưu lại, không tốn công thiết kế schema chặt.

**Trung thực (để phản biện):** nếu bài toán có **nhiều quan hệ** (đơn hàng–khách–sản phẩm),
cần **transaction ACID**, hoặc **truy vấn phân tích phức tạp (JOIN/aggregate nhiều bảng)** thì
**SQL hợp hơn**. Ở quy mô 435 sản phẩm + 1 thực thể, cả hai đều chạy tốt; MongoDB được chọn vì
**linh hoạt schema + tốc độ lặp**. Đây là quyết định theo *đặc thù dữ liệu*, không phải "NoSQL
luôn tốt hơn".

**Hai collection:** `laptops` (raw crawl) và `laptop_cleaned` (đã làm sạch — nguồn để build vector).

---

## 4. Tiền xử lý dữ liệu (`clean_product.ipynb`)

Từ raw → sạch, dùng `pandas` + `regex`:
- **Bỏ** sản phẩm thiếu tên/giá.
- **Trích số bằng regex:** giá `"14.390.000₫"` → `14390000`; RAM `"16 GB"` → `16`; tần số quét
  (lấy `max` nếu có nhiều số, `"hãng không công bố"` → `None`).
- **CPU:** trích từ tên máy bằng regex 2 lượt (pattern chính + pattern vá cho Snapdragon/Ryzen
  AI/Intel N).
- **GPU:** tách `gpu_type` (tích hợp/rời), `gpu_ram` (VRAM), `gpu_name` (làm sạch ngoặc/VRAM).
- **Kích thước:** tách `physical_dimensions` → dài/rộng/dày/`weight_kg`.
- **Chuẩn hoá `material`** → plastic/metal/carbon/other.

**Điểm yếu tự nhận (nói khi phỏng vấn):** regex giòn (dễ sai khi format lạ), thiếu
validation/dedup tự động, notebook không idempotent. **Cải thiện:** pydantic schema validation,
upsert theo `id`, pipeline hoá (Airflow/Prefect).

---

## 5. Làm giàu dữ liệu (Enrichment) — chìa khoá cải thiện retrieval

**Vấn đề gốc:** document ban đầu chỉ có **specs khô** ("i5, 16GB, 15.6 inch"). Nhưng người dùng
hỏi bằng **ngôn ngữ nhu cầu** ("máy cho sinh viên hay lướt web nhiều tab"). Câu hỏi và document
**không chung không gian ngữ nghĩa** → semantic search sai. **Insight: dữ liệu tốt > retriever xịn.**

Giải pháp — enrichment **2 tầng**:

### Tầng A — Nhãn use-case bằng RULE (`taxonomy.py`)
- Gán nhãn (`gaming`, `van_phong_hoc_tap`, `mong_nhe_di_dong`, `do_hoa_sang_tao`,
  `lap_trinh_ky_thuat`, `hieu_nang_cao`, `giai_tri_da_phuong_tien`, `gaming_cao_cap`) bằng
  **luật xác định trên specs** → **không hallucination, 100% kiểm chứng được**.
- **Ground truth = specs (facts) + taxonomy được định nghĩa & kiểm chứng.** Rule KHÔNG phải
  chân lý tuyệt đối — nó là *lớp gán nhãn kiểm chứng được*; ngưỡng là lựa chọn thiết kế nên
  được **document + validate** (chấm tay sample, kiểm tra phân phối nhãn 16–50%, không nhãn nào
  >60%/<3%).
- **Ngưỡng calibrate theo phân phối THỰC TẾ của catalog**, không theo cảm tính. VD `gaming` lấy
  **VRAM ≥ 4 (GPU rời RTX/GTX/RX, loại MX)** — floor inclusive để không loại oan máy 2022 như
  RTX 3050; phân biệt "ngưỡng PHÂN LOẠI (máy này LÀ gì)" ≠ "ngưỡng KHUYẾN NGHỊ MUA MỚI".

### Tầng B — Chưng cất mô tả bằng LLM (`enrich.py`)
- Dùng **gpt-4o-mini** (temperature=0, structured output) tóm tắt đoạn mô tả marketing thành
  **use-case summary khách quan** (bỏ cường điệu "đáng kinh ngạc") + `highlights` bám specs.
- **Rào chắn chống bịa:** taxonomy đóng, **cross-check với specs** (LLM gợi ý tag mâu thuẫn với
  specs → loại); **additive** (không phá specs gốc → blast radius nhỏ).
- **Chi phí thật:** ~5.500đ cho toàn bộ 435 máy (one-time).
- **Kết quả kiểm chứng:** 22% máy LLM-tag lệch rule-tag → **giữ rule làm chuẩn** (bám specs).
  VD máy TUF Gaming: LLM đề xuất "đồ hoạ" (vì marketing) nhưng rule từ chối đúng (panel chỉ
  62.5% sRGB, không đủ chuẩn màu) → minh chứng "rule (specs) > LLM (marketing)".

**Chưng cất (offline, build KB) ≠ CRAG (online, lúc chạy)** — hai thứ khác nhau, bổ trợ nhau.

---

## 6. Vector hoá (`vector_database.py`)

- Mỗi laptop → 1 `Document`: `page_content` = specs (mô tả tiếng Việt) **+ nhãn nhu cầu (cụm
  từ có dấu)** + **use-case summary**. `metadata` = id, name, brand, giá, RAM, gpu, **img**,
  use_case_tags, price_tier...
- **Embedding:** `text-embedding-3-small` (OpenAI). **Vì sao?** Chất lượng tiếng Việt tốt, chi
  phí ~free (~150đ cho 435 doc), nhất quán. (Phương án local miễn phí: Ollama `mxbai-embed-large`
  — đánh đổi chất lượng + phải rebuild. **Không được trộn** embedding model giữa index và query.)
- **Vector store:** ChromaDB (persist local). ID = `id` sản phẩm → rebuild là upsert, không nhân bản.

---

## 7. Nâng cấp Retrieval (phần "chất" nhất)

### 7.1 Hybrid: BM25 + Dense + RRF (`retrieval.py`)
- **Dense (embedding):** hiểu ngữ nghĩa, nhưng hay trượt **khớp từ khóa chính xác** (mã CPU/GPU).
- **BM25 (lexical):** bắt đúng "RTX 4060", "i5-1335U", tên dòng. Dùng **`bm25s`** — dựng
  **index thưa (sparse, kiểu inverted-index)**, chỉ đụng doc chứa token qua nhân ma trận thưa
  (~0.1 ms/query), KHÔNG brute-force như `rank_bm25`.
- **RRF (Reciprocal Rank Fusion):** ghép 2 danh sách chỉ bằng **thứ hạng**: điểm = Σ `1/(k+rank)`
  (k=60). **Vì sao RRF?** BM25 score và cosine **khác thang đo hoàn toàn** → không cộng trực tiếp
  được; RRF chỉ dùng thứ hạng nên ghép được mọi retriever, không cần chuẩn hoá.
- **Minh chứng sống:** câu *"máy nhẹ mang đi học"* — dense-only trả máy gaming (nặng) lên đầu;
  hybrid+RRF trả đúng máy mỏng nhẹ. BM25 bù đúng chỗ dense yếu.

### 7.2 Rerank: Cross-encoder (`bge-reranker-v2-m3`, chạy LOCAL/free)
- **Bi-encoder (embedding)** mã hoá query & doc **RIÊNG** → nhanh nhưng xếp hạng chưa tinh.
- **Cross-encoder** đưa **(query + doc) CHUNG** vào model → chấm độ liên quan **chính xác hơn hẳn**,
  nhưng chậm → chỉ dùng **rerank ~20 ứng viên** (không quét cả 435).
- **2 tầng:** bi-encoder quét rộng nhanh → cross-encoder chấm kỹ số ít. Vừa nhanh vừa chính xác.

### 7.3 CRAG (Corrective RAG) (`crag.py`) — vá "retrieval sai khi thiếu keyword"
- Ý tưởng (paper CRAG 2024, thích nghi cho rec): **đừng tin retrieval mù quáng**. Chấm chất
  lượng → kém thì **sửa** (rewrite query) → vẫn kém thì **HỎI LẠI user** thay vì trả lời bừa.
- **Bài học quan trọng (talking point mạnh):** ban đầu dùng **LLM làm grader** → **nhiễu**,
  chỉnh prompt là *seesaw* (lúc quá dễ, lúc quá gắt, từ chối oan câu tốt). → **Đổi sang gate
  bằng ĐIỂM của cross-encoder reranker** (sigmoid ∈ [0,1]): rẻ hơn (không tốn call chấm, dùng
  điểm đã có), ổn định hơn. Ngưỡng calibrate từ probe thực tế (câu tốt ~0.72, câu vô nghĩa ~0.50).
- **Hạn chế đã biết (trung thực):** vùng điểm ~0.51–0.55 nhập nhằng (câu mơ hồ và câu-yếu-nhưng
  -hợp-lệ lẫn nhau). Đặt ngưỡng thấp để không từ chối oan câu hợp lệ, đổi lại đôi khi bỏ sót câu mơ hồ.
- **Demo sống:** *"gợi ý máy nào đó đi"* (mơ hồ) → CRAG hỏi lại ngân sách/nhu cầu, KHÔNG dump bừa.

### 7.4 Vì sao KHÔNG dùng DRAG (paper được đưa)?
Đọc paper **DRAG (Debate-Augmented RAG, 2025)** và **từ chối có lý do**:
- DRAG cho **QA factual 1 đáp án** (đo EM/F1 với gold answer) — bài của ta là **recommendation**
  (nhiều đáp án đúng, không có gold EM).
- DRAG tốn **~10 LLM call/query** (multi-agent debate) → **phá latency & chi phí**, đúng 2 thứ
  interviewer quan tâm.
- Chọn **CRAG** vì **hợp bài toán + rẻ + đo được**. (Tiêu chí chọn kỹ thuật: hợp + chi phí + đo
  được, KHÔNG phải "paper mới nhất".)

---

## 8. Agent & định tuyến công cụ (`chat_bot_cloud.py`)

- Dùng **gpt-4o-mini** + tool-calling loop. **2 công cụ:**
  - `query_mongodb`: lọc **cứng** (giá/RAM/hãng/GPU) — **vì sao?** Embedding **kém với số học &
    exact-match** ("dưới 15 triệu", "RAM 16GB"); lọc chính xác nên để MongoDB làm.
  - `search_vector`: nhu cầu **mô tả** → Hybrid + Rerank + CRAG.
- Agent tự route: ràng buộc cứng → MongoDB; nhu cầu mềm → semantic. **Đây là câu trả lời "của
  mình" cho "vì sao pure vector search sai"** — đã gặp và tự tách 2 nhánh.
- Trả về `(text, products)`: gom sản phẩm đã truy xuất (tên, giá, cấu hình, **ảnh**, link),
  chọn máy **khớp câu trả lời** qua mã model.

---

## 9. Giao diện (`user_interface_cloud.py`, Streamlit)

- Chat UI (dark theme), **card sản phẩm có ẢNH** + giá + cấu hình + link "Xem chi tiết".
- Kế thừa lịch sử hội thoại (nhớ ngân sách/nhu cầu trước).
- **Reranker preload lúc khởi động** (spinner ~30s, chỉ 1 lần) để câu đầu không treo trắng trang.

---

## 10. Đánh giá (Evaluation) (`eval/eval_min.py` + `goldset.jsonl`)

**Phương pháp:** gold set 16 câu (8 nhu cầu, 8 ràng buộc). Relevance **nhị phân, tự kiểm chứng
từ metadata** (không gán nhãn tay chủ quan):
- Câu nhu cầu → relevant nếu doc có đúng `use_case_tag`.
- Câu ràng buộc → relevant nếu thỏa giá/RAM/hãng.

**Chỉ số & vì sao:**
- **Precision@5, @10:** top-k đúng bao nhiêu.
- **NDCG@5, @10:** thưởng khi xếp món liên quan **lên cao** (đo chất lượng *xếp hạng*, không chỉ
  đếm). NDCG@k = DCG@k / IDCG@k, DCG = Σ `rel_i / log2(i+1)`.
- **Hit@5:** top-5 có ≥1 món đúng.
- **KHÔNG dùng Recall cho câu nhu cầu:** tập đúng theo tag rất lớn (VD `gaming` >130 máy) →
  recall@5 luôn tí xíu, vô nghĩa (đây là "cái bẫy" thường gặp).

**So sánh (A) dense-only vs (B) hybrid+rerank:**

### Câu NHU CẦU (relevance = đúng use_case_tag)
| Mode | P@5 | P@10 | NDCG@5 | NDCG@10 | Hit@5 |
|---|---|---|---|---|---|
| A) dense-only | 0.725 | 0.675 | 0.746 | 0.704 | 1.000 |
| **B) hybrid+rerank** | **0.975** | **0.938** | **0.984** | **0.955** | 1.000 |

### Câu RÀNG BUỘC (relevance = thỏa giá/RAM/hãng)
| Mode | P@5 | P@10 | NDCG@5 | NDCG@10 | Hit@5 |
|---|---|---|---|---|---|
| A) dense-only | 0.700 | 0.675 | 0.707 | 0.687 | 0.750 |
| **B) hybrid+rerank** | **0.750** | **0.750** | **0.750** | **0.750** | 0.750 |

**Diễn giải:**
- **Câu nhu cầu cải thiện mạnh:** NDCG@10 **0.70 → 0.96**, P@5 **0.73 → 0.98** — reranker đưa
  đúng máy lên đầu.
- **Hit@5 = 1.0 cả hai** ở câu nhu cầu: bão hoà (tag rộng) → không phân biệt được (đã biết trước).
- **Câu ràng buộc gần như KHÔNG cải thiện** (hybrid ≈ dense): retrieval thuần (cả BM25 lẫn dense)
  **không lọc chính xác được số/giá** → đây chính là lý do route ràng buộc cứng sang
  `query_mongodb` (đạt ~100% khi qua tool đó). Đây là bằng chứng cho quyết định tách 2 nhánh.

**Đánh giá tốc độ (machine-independent):** đo **số LLM call/câu** (độc lập máy — giống cách paper
báo cáo) + bóc tách theo stage; reranker chạy **local (0đ, không call)**. Giây tuyệt đối phụ
thuộc máy nên không lấy làm chính.

---

## 11. Hạn chế đã biết & hướng phát triển

- CRAG grader vùng ~0.5 nhập nhằng (bản chất bài toán khó).
- Regex cleaning giòn; nên thêm pydantic validation + upsert idempotent.
- 1 máy không có mô tả (trang không có khối text).
- Ollama concurrency yếu → bản deploy nhiều user nên dùng API + FastAPI async + queue.
- Có thể mở rộng: crawl thêm shop (FPT/CellphoneS) → thêm bài toán **entity resolution/dedup**.

---

## 11b. Mở rộng lên ~100.000+ documents (câu hỏi scale)

Hiện tại 435 docs → mọi thứ chạy in-memory, tức thì. Ở **100k docs**, phải xử lý từng thành phần.
Nguyên tắc: **chỉ ra đúng bottleneck** (không phải "đổi hết"), vì nhiều phần đã scale sẵn.

### Bảng: cái gì đã scale, cái gì vỡ
| Thành phần | Hiện tại (435) | Ở 100k | Cách xử lý |
|---|---|---|---|
| **Dense (Chroma HNSW)** | ANN ~O(log N) | ✅ Vẫn ổn (HNSW chạy tốt tới hàng triệu) | Tune `ef_search`/`M`; RAM ~100k×1536×4B ≈ **0.6GB** — chấp nhận được |
| **BM25 (`bm25s`)** | sparse index, ~0.1ms/query | ✅ Ổn in-memory tới ~1M docs | Chục triệu/phân tán/cần service chung → Elasticsearch/OpenSearch hoặc Qdrant/Weaviate |
| **Rerank (cross-encoder)** | rerank ~20 ứng viên | ✅ Ổn — chi phí **độc lập N** (chỉ chấm top-k đã lấy) | Không đổi. Đây là điểm mạnh thiết kế |
| **RRF fusion** | ghép top-N | ✅ Ổn — chỉ ghép top-N mỗi retriever | Không đổi |
| **MongoDB (query_mongodb)** | collection scan | ⚠️ Chậm nếu không index | **Thêm index** trên `price`, `ram_info`, `brand`, `gpu_type`, `weight_kg` (compound index) |
| **Embedding (build)** | 435 docs, ~free | ⚠️ Tốn thời gian | **Batch API** (2048 input/request) + async; chi phí ~100k×800 token×$0.02/1M ≈ **$1.6** |
| **Enrich (LLM distill)** | 435 call, ~5.5k đ | ⚠️ Tốn nhất | **OpenAI Batch API** (rẻ 50%, async), hoặc parallel; ~100k call ≈ **$50** (gpt-4o-mini) |
| **Serving (Streamlit in-memory)** | 1 process | ❌ Không chịu nhiều user | **FastAPI async** + vector DB as-a-service + Redis (cache/session) + autoscale |

### Chi tiết các quyết định ở 100k

**1. BM25 (`bm25s`) đã scale tới ~1M docs in-memory.**
`bm25s` dựng **index thưa (sparse)** → ~0.1ms/query, RAM vừa phải → **ở 100k chạy tốt**. Chỉ khi
lên **hàng chục triệu docs**, cần **phân tán**, hoặc cần **một service tìm kiếm dùng chung nhiều
app** thì mới đẩy sang **Elasticsearch/OpenSearch** (inverted index Lucene) hoặc vector DB hybrid
(Qdrant/Weaviate/Vespa) lo cả sparse (BM25/SPLADE) lẫn dense trong 1 hệ.

**2. Dense đã scale — chỉ cần tune, và cân nhắc nén ở quy mô rất lớn.**
HNSW ổn ở 100k (RAM ~0.6GB). Ở **10M+** thì cân nhắc **IVF-PQ** (quantization nén vector), **DiskANN**
(index trên đĩa), hoặc DB phân tán (**Milvus/Qdrant** cluster, shard theo brand/khoảng giá).

**3. Rerank & RRF không phụ thuộc N — nói rõ điểm này.**
Reranker chỉ chấm **top-20~100 ứng viên** đã retrieve → chi phí **cố định theo k**, không tăng theo
kích thước kho. Kiến trúc 2 tầng (ANN quét rộng → cross-encoder chấm hẹp) chính là cách giữ latency
ổn định dù kho to.

**4. MongoDB: thêm index cho filter.**
`query_mongodb` lọc theo giá/RAM/hãng → ở 100k cần **index** (nếu không là COLLSCAN toàn collection).
`db.laptop_cleaned.createIndex({price:1, ram_info:1, brand:1})` v.v. MongoDB bản thân chứa 100k docs
là chuyện nhỏ.

**5. Pipeline dữ liệu: batch hoá + có checkpoint.**
- Crawl 100k: **crawl phân tán/song song** (queue Celery/Redis), rate-limit, **resume/checkpoint** (đã
  làm resume-able), rotate proxy/user-agent.
- Clean: pandas ổn tới ~1M dòng; lớn hơn dùng **Dask/Spark** hoặc xử lý theo chunk.
- Enrich (LLM): dùng **Batch API** (async, rẻ 50%) — đây là chi phí lớn nhất ở scale.

**6. Serving nhiều user:** bỏ in-memory Streamlit → **FastAPI async** + **vector DB dịch vụ**
(Qdrant/Milvus managed) + **Redis** (semantic cache giảm tải LLM, session store) + **autoscale container**
sau load balancer. Reranker để **microservice riêng** (GPU) để không chặn API.

**Tóm tắt câu trả lời phỏng vấn:** *"Ở 100k, dense (Chroma HNSW), BM25 (bm25s — sparse index
~0.1ms) và rerank/RRF của em đều vẫn ổn: rerank chỉ chấm top-k nên độc lập N, bm25s là index thưa
chứ không brute-force. Việc cần làm chủ yếu là thêm index MongoDB cho filter, batch embedding/enrich
(Batch API), và tách serving sang FastAPI async + vector DB dịch vụ + Redis cache. Chỉ khi lên hàng
chục triệu docs / cần phân tán / cần service chung thì mới đẩy BM25 sang Elasticsearch, nén vector
bằng IVF-PQ/DiskANN, hoặc dùng vector DB phân tán (Milvus/Qdrant)."*

---

## 12. Tech stack

| Thành phần | Công nghệ | Vì sao |
|---|---|---|
| Crawl | Playwright + BeautifulSoup | Trang render JS động |
| Lưu trữ | MongoDB | Schema linh hoạt, dữ liệu thưa, lặp nhanh |
| Clean | pandas + regex | Trích/chuẩn hoá số từ text |
| Enrich | Rule (taxonomy) + gpt-4o-mini | Tag không hallucination + summary ngữ nghĩa |
| Embedding | text-embedding-3-small | Tiếng Việt tốt, rẻ |
| Vector DB | ChromaDB | Nhẹ, persist local |
| Retrieval | BM25 (`bm25s`, sparse) + dense + RRF + cross-encoder (bge-reranker-v2-m3) | Hybrid + rerank |
| Corrective | CRAG (gate điểm rerank) | Phát hiện retrieval kém → hỏi lại |
| LLM | gpt-4o-mini | Rẻ, đủ mạnh, scale qua API |
| Agent | LangChain tool-calling | Route structured vs semantic |
| UI | Streamlit | Nhanh dựng, card ảnh |

---

## 13. Các file chính

> Toàn bộ mã nguồn pipeline nằm trong `src/`. Notebook làm sạch nằm trong `notebooks/`.
> Chạy mọi lệnh **từ thư mục gốc project** để `./vectorstore` phân giải đúng.

| File | Vai trò |
|---|---|
| `src/crawling_data.py` | Crawl specs (Playwright) |
| `src/crawl_description.py` | Crawl mô tả (async, resume-able) |
| `notebooks/clean_product.ipynb` | Làm sạch (pandas+regex) |
| `src/taxonomy.py` | Gán nhãn use-case bằng rule |
| `src/enrich.py` | Chưng cất mô tả bằng LLM |
| `src/vector_database.py` | Build Chroma (specs+tags+summary) |
| `src/retrieval.py` | Hybrid + RRF + rerank |
| `src/crag.py` | Corrective retrieval |
| `src/chat_bot_cloud.py` | Agent + 2 tool + gom sản phẩm |
| `src/user_interface_cloud.py` | Streamlit UI + card ảnh |
| `eval/eval_min.py`, `eval/goldset.jsonl` | Đánh giá before/after |
