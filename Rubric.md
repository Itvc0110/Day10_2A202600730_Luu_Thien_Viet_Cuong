# Rubric

## Cách tính điểm

Tổng điểm cơ bản: `0-90`

Điểm bonus: `90-100`

## Mục 1: Code structure và project organization - 10 điểm

- 0-3: Code rời rạc, khó đọc, không chia module rõ
- 4-7: Có chia module nhưng chưa nhất quán
- 8-10: Cấu trúc rõ ràng, dễ follow, tên file/hàm hợp lý

## Mục 2: Raw data ingestion - 15 điểm

- 0-5: Chưa load được data hoặc load rất thiếu
- 6-10: Load được data nhưng parse chưa ổn định
- 11-15: Load, parse, lưu raw response và raw records đầy đủ

## Mục 3: Cleaning và data modeling - 15 điểm

- 0-5: Cleaning sơ sài, mất nhiều thông tin
- 6-10: Có cleaned dataset nhưng chưa chắc
- 11-15: Clean rõ ràng, `text_for_embedding` hợp lý, schema tốt

## Mục 4: Embedding và vector store - 10 điểm

- 0-3: Chưa tạo được embedding/index
- 4-7: Tạo được embedding nhưng query chưa ổn
- 8-10: ChromaDB + MiniLM hoạt động đúng, retrieval hợp lý

## Mục 5: Agent và multi-provider LLM - 10 điểm

- 0-3: Agent chưa chạy
- 4-7: Agent chạy nhưng provider support còn ít hoặc lỗi
- 8-10: Agent chạy tốt, provider abstraction rõ ràng

## Mục 6: Evaluation và scoring - 10 điểm

- 0-3: Chưa có evaluation
- 4-7: Có metrics nhưng chưa đầy đủ
- 8-10: Có baseline metrics rõ ràng, answers artifacts đầy đủ

## Mục 7: Data observability - 10 điểm

- 0-3: Chưa có quality/freshness
- 4-7: Có check cơ bản
- 8-10: Có quality checks, freshness, báo cáo rõ ràng

## Mục 8: Corruption và comparison - 10 điểm

- 0-3: Chưa simulate corruption
- 4-7: Có corruption nhưng chưa đo được impact rõ
- 8-10: Có corrupted, repaired, comparison metrics hợp lý

## Bonus 90-100

Bonus chỉ tính khi bài làm đã đạt ít nhất 90 điểm cơ bản.

Có thể cộng bonus nếu:

- Báo cáo markdown rất rõ ràng, có phân tích thay đổi metrics
- Data corruption scenario hợp lý và có ý nghĩa
- Có thêm visualization hoặc bảng so sánh đẹp
- Có test hoặc validation bổ sung
- Có cài đặt CLI/use case để dễ reproduce

## Trừ điểm

- Không chạy được end-to-end
- Commit thiếu file quan trọng
- Hard-code path hoặc key nhạy cảm
- Báo cáo không match artifact thực tế
