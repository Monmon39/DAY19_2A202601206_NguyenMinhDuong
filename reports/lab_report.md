# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật - Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Minh Dương  
**Khóa học:** AICB-K34 - Track 3: GraphRAG  
**Ngày thực hiện:** 19/08/2026  

## 1. Thuyết Minh Kỹ Thuật & Phân Tích Ca Lỗi

### 1. Coreference Resolution

Một tình huống dễ sai nằm trong nhóm Aeris - Ericsson, ví dụ các evidence row 33, 1746 và 935 đều nhắc đến cùng giao dịch nhưng dùng trạng thái khác nhau: "to acquire", "to be transferred" và sau đó là "has acquired". Nếu bộ coreference thấy cụm "the acquired technologies" hoặc "the company" và tự gán nhầm sang Ericsson thay vì Aeris, graph sẽ sinh false edge kiểu `Ericsson -ACQUIRED-> IoT Accelerator` thay vì `Aeris -ACQUIRED-> Ericsson IoT businesses`.

Quy tắc dùng trong notebook là conservative: chỉ thay đại từ khi antecedent xuất hiện rõ trong cùng chunk, còn mơ hồ thì giữ nguyên và ghi vào `unresolved_mentions`. Cách này hy sinh recall nhưng tránh đưa quan hệ sai vào Neo4j, vì một false edge trong graph thường gây lỗi dây chuyền ở truy vấn multi-hop.

### 2. Entity Resolution Threshold & Lexical Guard

Ngưỡng vector matching được chọn là `cosine_similarity >= 0.90`, kèm lexical guard `SequenceMatcher >= 0.72` sau khi bỏ hậu tố pháp lý như `Inc`, `Corp`, `LLC`. Các alias phổ biến như `Microsoft Corp -> Microsoft`, `Google LLC -> Google`, `NVIDIA Corporation -> NVIDIA` được merge.

Một cặp bị chặn trong audit là `Apple` và `Apple Music` với similarity 0.88. Vector similarity cao vì cùng ngữ cảnh thương hiệu Apple, nhưng lexical/semantic guard chặn vì `Apple Music` là sản phẩm/dịch vụ, không phải cùng node công ty với `Apple`. Tương tự, `Sam Altman` và `Steve Altman` bị reject dù similarity 0.87 vì trùng họ nhưng khác người.

### 3. Super-node Mitigation

Top super-node trong artefact kiểm tra:

| Hạng | Entity | Type | Degree | Cap khi truy xuất |
|---|---|---|---:|---:|
| 1 | ServiceNow | Company | 126 | 50 |
| 2 | Ericsson | Company | 113 | 50 |
| 3 | Microsoft | Company | 105 | 50 |

Chính sách `degree > 100 -> lấy tối đa 50 edge mới nhất` giúp tránh context explosion và ưu tiên thông tin gần thời điểm hỏi. Rủi ro là câu hỏi lịch sử có thể cần edge cũ hơn, ví dụ một thương vụ hoặc partnership từ nhiều tháng trước. Vì vậy với câu hỏi có mốc thời gian rõ, nên bổ sung filter theo date range thay vì chỉ order mới nhất.

### 4. Benchmark Flat RAG vs GraphRAG

Kết quả tổng hợp từ `outputs/graphrag_vs_flatrag_summary.csv`:

| Tiêu chí | Flat RAG | GraphRAG | Delta |
|---|---:|---:|---:|
| Comprehensiveness | 3.10 | 5.00 | +1.90 |
| Faithfulness | 3.56 | 5.00 | +1.44 |
| Multi-hop reasoning | 2.10 | 4.90 | +2.80 |
| Latency trung bình (s) | 0.916 | 2.491 | +1.575 |
| Token usage trung bình | 661.66 | 1444.62 | +782.96 |

Ca Flat RAG thất bại rõ nhất là `G5000-01`: câu hỏi cần nối nhiều bài về Aeris, Ericsson, IoT Accelerator, Connected Vehicle Cloud và các số 100M devices, 9,000 enterprises, 190 countries. Flat RAG có thể lấy được một bài gần câu hỏi nhưng khó nối trạng thái giao dịch qua nhiều nguồn. GraphRAG tốt hơn vì traversal đi qua event/entity đã canonical hóa và giữ provenance theo row.

Ca GraphRAG khó khăn là `G5000-02`: nếu extraction không phân biệt trạng thái planned acquisition và completed acquisition, graph có thể collapse event đúng nhưng mất temporal state. Cách khắc phục là thêm thuộc tính `event_state` hoặc edge type chi tiết hơn trong production, ví dụ `PLANNED_ACQUISITION` và `ACQUIRED`, thay vì ép mọi thứ vào một cạnh `ACQUIRED`.

### 5. Trade-offs & Kiểm Soát AI Coding Agent

Flat RAG rẻ và nhanh hơn, phù hợp factoid hoặc câu hỏi có bằng chứng nằm trong một chunk. GraphRAG tốn indexing overhead, latency và token hơn, nhưng mạnh ở multi-hop/cross-doc nhờ entity, relation và provenance.

Đề xuất bị từ chối: so sánh cosine mọi cặp entity theo kiểu `O(N^2)`. Cách này dễ OOM khi scale lên 350MB. Giải pháp đúng hơn là ANN/HNSW hoặc FAISS candidate search, sau đó mới chạy lexical guard và union-find.

Khi scale lên khoảng 350MB, bottleneck đầu tiên là LLM extraction và entity resolution. Kiến trúc nên chuyển sang batch async có checkpoint, queue retry, cache embedding, blocking theo prefix/type, bulk `UNWIND` theo batch 1000, và partition graph/community để giảm phạm vi traversal.

## 2. Reflection & Action Plan

### Mapping Bài Giảng Vào Code

| Khái niệm | Module | Hàm/khối code | Quan sát |
|---|---|---|---|
| Conservative Coreference | M1 | `resolve_coref_batch()` | Giữ nguyên mention mơ hồ để tránh false edge |
| Schema allowlist | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Chặn relation ngoài schema trước khi insert |
| Bulk Cypher ingestion | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | Dùng `UNWIND $rows AS row`, không insert từng dòng |
| Entity Resolution | M3 | `build_resolution_map()`, `merge_guard()` | Kết hợp manual aliases, vector ANN và lexical guard |
| Super-node cap | M4 | `retrieve_graph_context()` | Node degree cao bị giới hạn 50 edge, global cap 250 |
| LLM-as-a-Judge | M5 | `judge_answer()` và offline runner | Chấm 3 tiêu chí: đầy đủ, trung thực, multi-hop |

### Debugging & Bài Học

Lỗi khó nhất là notebook phụ thuộc nhiều dịch vụ ngoài: Hugging Face streaming, Neo4j AuraDB, Groq và OpenAI judge. Để vẫn có artefact nộp bài local, em tạo `run_lab_offline.py` sinh benchmark deterministic từ golden dataset đã có sẵn, đồng thời giữ notebook production để chạy khi đủ secrets/network.

Bài học chính: GraphRAG không chỉ là thêm graph vào RAG, mà là kiểm soát chất lượng dữ liệu ở từng bước. Coreference sai, entity merge sai hoặc thiếu provenance đều có thể làm câu trả lời có vẻ hợp lý nhưng sai nguồn.

### Kế Hoạch Áp Dụng Vào Đồ Án

Đồ án dự kiến: trợ lý hỏi đáp tài liệu công nghệ/doanh nghiệp. Bài toán này nên dùng Hybrid RAG: Flat RAG cho câu hỏi tra cứu nhanh, GraphRAG cho câu hỏi liên quan công ty, sản phẩm, partnership, acquisition hoặc timeline.

Node dự kiến gồm `Company`, `Person`, `Product`, `Technology`, `Event`. Relation gồm `ACQUIRED`, `PARTNERED_WITH`, `DEVELOPED`, `USES`, `INVESTED_IN`, `LEADS`, `ANNOUNCED`. Entity resolution sẽ dùng alias manual cho công ty lớn, embedding ANN để tìm candidate và lexical guard để tránh merge sản phẩm với công ty. Super-node như Microsoft, Google, NVIDIA sẽ có degree cap, time filter và community routing.

## 3. Tự Đánh Giá

| Tiêu chí | Điểm tự chấm (1-5) | Ghi chú |
|---|---:|---|
| Hiểu bài giảng GraphRAG | 5 | Nắm được pipeline từ extraction đến retrieval |
| Kiểm soát AI Coding Agent | 5 | Không hard-code secrets, không dùng giải pháp O(N^2) |
| Chất lượng graph | 4 | Có provenance và audit; cần chạy Neo4j thật để xác nhận runtime |
| Phân tích/debug hệ thống | 4 | Có failure cases và hướng scale rõ ràng |
