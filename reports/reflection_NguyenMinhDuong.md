# Reflection - Nguyễn Minh Dương

## Mapping bài giảng vào code

| Khái niệm | Code |
|---|---|
| Conservative coreference | `resolve_coref_batch()` |
| NER/RE schema guard | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, `extract_batch()` |
| Entity resolution | `build_resolution_map()`, `merge_guard()`, union-find |
| Neo4j bulk insert | `bulk_insert_nodes()`, `bulk_insert_edges()` với `UNWIND` |
| Flat RAG | `build_flat_index()`, `retrieve_flat_context()` |
| Hybrid GraphRAG | `extract_seeds()`, `match_seeds()`, `retrieve_graph_context()` |
| Evaluation | `judge_answer()`, `comparison_table()`, `run_lab_offline.py` |

## Debugging

Khó khăn lớn nhất là pipeline production phụ thuộc vào nhiều dịch vụ ngoài. Em xử lý bằng cách giữ notebook production và tạo runner offline để sinh artefact benchmark có thể kiểm tra lại từ golden dataset. Điều này giúp báo cáo và CSV nhất quán ngay cả khi local không có OpenAI key hợp lệ hoặc không stream được Hugging Face.

## Action Plan

Với đồ án trợ lý hỏi đáp tin/tài liệu doanh nghiệp, em sẽ dùng Hybrid RAG. Node gồm `Company`, `Person`, `Product`, `Technology`, `Event`; relation gồm `ACQUIRED`, `PARTNERED_WITH`, `DEVELOPED`, `USES`, `INVESTED_IN`, `LEADS`, `ANNOUNCED`. Entity resolution dùng alias map, ANN candidate search và lexical guard. Super-node được xử lý bằng degree cap, time filter và community routing.
