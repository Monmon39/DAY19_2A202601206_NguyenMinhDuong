# Failure Analysis - Lab 19

## Case 1: Flat RAG fails, GraphRAG succeeds

**Question:** `G5000-01`, Aeris - Ericsson IoT transaction.

Flat RAG dễ lấy một hoặc hai chunk liên quan nhưng thiếu chuỗi reasoning đầy đủ: Ericsson IoT Accelerator và Connected Vehicle Cloud chuyển sang Aeris, sau đó footprint được mô tả là hơn 100 triệu IoT devices, 9,000 enterprises và 190 countries. GraphRAG tốt hơn vì canonicalize Aeris/Ericsson/assets thành một event có nhiều evidence rows và traversal gom được quan hệ lẫn provenance.

## Case 2: GraphRAG fails or becomes risky

**Question:** `G5000-02`, planned transfer hay completed acquisition.

GraphRAG có thể sai nếu extraction ép cả "to acquire" và "has acquired" vào relation `ACQUIRED` mà không lưu `event_state` và `published_date`. Khi đó graph đúng entity nhưng sai timeline. Cách sửa là mở rộng schema event: `PLANNED_ACQUISITION`, `ACQUIRED`, hoặc thêm thuộc tính `status`, `valid_from`, `published_date` trên edge/event node.

## Root Causes

Flat RAG thất bại vì context bị phân mảnh theo chunk và similarity không biểu diễn tốt quan hệ nhiều bước. GraphRAG thất bại khi upstream extraction thiếu precision, entity resolution merge sai, hoặc super-node cap cắt mất edge cần thiết.

## Mitigations

Giữ provenance bắt buộc trên mọi edge, audit entity-resolution, dùng conservative coreference, thêm temporal state cho event, và cho retrieval nhận biết date range thay vì chỉ lấy edge mới nhất.
