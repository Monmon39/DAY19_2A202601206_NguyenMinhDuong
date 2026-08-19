# Technical Defense - Lab 19

1. **Coreference:** Với nhóm Aeris - Ericsson, cụm "the acquired technologies" có thể bị gán nhầm nếu chunk chứa nhiều công ty. False coreference sẽ tạo false edge trong graph, nên pipeline chỉ resolve khi antecedent rõ trong cùng chunk.
2. **Threshold:** Chọn `cosine >= 0.90` cho vector entity matching và lexical guard `>= 0.72`. Mức này giảm false merge nhưng vẫn bắt được alias công ty có hậu tố pháp lý.
3. **Rejected high-sim pair:** `Apple` vs `Apple Music` similarity 0.88 bị reject vì một bên là công ty, một bên là service/product.
4. **Top super-nodes:** ServiceNow degree 126, Ericsson degree 113, Microsoft degree 105 trong artefact `outputs/graph_health_checks.csv`.
5. **Temporal cap:** Lấy 50 edge mới nhất giúp giảm token và ưu tiên tin mới, nhưng có thể bỏ sót sự kiện lịch sử nếu câu hỏi hỏi về quá khứ.
6. **Flat RAG thắng ở đâu:** Factoid và lookup trực tiếp vì latency thấp, ít token, không cần graph traversal.
7. **GraphRAG thắng ở đâu:** Multi-hop/cross-doc vì nối được entity, relation, event state và provenance qua nhiều bài.
8. **Latency/token trade-off:** Overall latency Flat 0.916s, GraphRAG 2.491s; token Flat 661.66, GraphRAG 1444.62.
9. **Agent proposal rejected:** Không dùng pairwise cosine `O(N^2)` cho toàn bộ entity set; thay bằng ANN candidate search, lexical guard và union-find.
10. **Scale 350MB:** Bottleneck là LLM extraction và entity resolution. Cần async batch, checkpoint, retry queue, cache embedding, ANN index và community partitioning.
