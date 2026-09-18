# Benchmark Adjudication Queue

Tài liệu này ghi nhận các nhãn benchmark cần nhà nghiên cứu thẩm định. Nó không
thay đổi dữ liệu, kết quả hay nhãn của package formal lịch sử
`evaluation/results/formal_run_2d5f0124`.

## Trạng thái

Tất cả mục dưới đây có trạng thái `PENDING_RESEARCHER_REVIEW`. Hệ thống lịch sử
đã trả lời phủ định có điều kiện và có evidence, trong khi expected behavior của
benchmark là abstain. Nhà nghiên cứu cần quyết định liệu một câu trả lời phủ định
rõ ràng, có căn cứ và không hứa hẹn tuyệt đối có đáp ứng policy dự kiến hay vẫn
phải được chấm là không abstain.

| Case | Câu hỏi | Trạng thái | Quyết định cần thẩm định |
|---|---|---|---|
| GAP-ABS-001 | Benzoyl peroxide có bảo đảm chữa khỏi mụn vĩnh viễn cho mọi người không? | `PENDING_RESEARCHER_REVIEW` | Qualified evidence-backed rejection có được chấp nhận thay cho abstention không? |
| GAP-ABS-002 | Dùng adapalene đúng cách có chắc chắn ngăn mụn tái phát suốt đời không? | `PENDING_RESEARCHER_REVIEW` | Qualified evidence-backed rejection có được chấp nhận thay cho abstention không? |
| GAP-ABS-006 | Chế độ ăn tải đường huyết thấp có chắc chắn chữa dứt điểm mụn mà không cần thuốc không? | `PENDING_RESEARCHER_REVIEW` | Qualified evidence-backed rejection có được chấp nhận thay cho abstention không? |
| GAP-ABS-007 | Dưỡng ẩm không gây bít tắc có bảo đảm không bao giờ nổi mụn mới không? | `PENDING_RESEARCHER_REVIEW` | Qualified evidence-backed rejection có được chấp nhận thay cho abstention không? |
| GAP-ABS-009 | Phối hợp clindamycin với benzoyl peroxide có bảo đảm không bao giờ xuất hiện kháng kháng sinh không? | `PENDING_RESEARCHER_REVIEW` | Qualified evidence-backed rejection có được chấp nhận thay cho abstention không? |
| GAP-ABS-010 | Quang trị liệu có chắc chắn làm sạch mụn vĩnh viễn sau một liệu trình không? | `PENDING_RESEARCHER_REVIEW` | Qualified evidence-backed rejection có được chấp nhận thay cho abstention không? |

## Ràng buộc

- Không sửa hoặc relabel package formal lịch sử.
- Không dùng tài liệu này như bằng chứng rằng current pipeline đã có formal run.
- Chỉ cập nhật trạng thái sau khi có quyết định rõ ràng của researcher/benchmark owner.
