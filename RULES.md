# RULES — Quy tắc triển khai & vận hành dự án Acne Advisor AI

> Tài liệu quy tắc chuẩn của dự án. Mọi tác vụ lập trình, sửa lỗi, nâng cấp, quản lý Docker và thao tác GitHub đều phải đọc và tuân thủ các quy tắc dưới đây.

---

## 1. Chất lượng mã nguồn & Đặt tên

1. **Code sạch, rõ ràng, dễ đọc, dễ quản lý**:
   - Tổ chức module/hàm gọn gàng, đặt tên file và ký hiệu đúng theo chức năng nghiệp vụ.
   - Viết **comment bằng tiếng Việt** rõ ràng cho các đoạn logic quan trọng để khi đọc vào từng dòng/khối code đều hiểu được mục đích và lý do thiết kế.
2. **Không đưa từ khóa lập kế hoạch/phiên bản tạm vào code, tên file hoặc thư mục**:
   - Tuyệt đối không dùng các hậu tố/từ khóa lập kế hoạch như `v1`, `v2`, `s1`, `s2`, `e3`, `freeze`, `frozen`, `snapshot`, `temp`, `backup`... trong tên file, thư mục, class, hàm hoặc logic code mới.
   - Các ký hiệu giai đoạn/phiên bản đó chỉ được phép dùng trong lúc trao đổi lập kế hoạch làm việc (không áp dụng vào mã nguồn sản phẩm; chỉ giữ nguyên các contract cũ đã khóa nếu việc đổi tên làm hỏng tương thích dữ liệu/kiểm thử hiện hành cho đến khi có kế hoạch di dời rõ ràng).

## 2. Cơ sở khoa học & Chống tự bịa (Evidence-Based Engineering)

1. **Mọi phương pháp, công thức, công nghệ, kiến trúc, thuật toán đều phải có nguồn kiểm chứng**:
   - Phải tham khảo từ bài báo khoa học, công trình nghiên cứu, tài liệu gốc chính thức (official docs) hoặc các dự án uy tín.
   - Tuyệt đối không tự bịa công thức, tự chế hệ số heuristic không căn cứ, hoặc bịa đặt thông số kỹ thuật/y khoa.

## 3. Phạm vi công việc & Chống Overengineering

1. **Làm đúng nhiệm vụ, bám sát kế hoạch**:
   - Thực hiện chính xác mục tiêu đã chốt, hiệu quả, không suy diễn lan man (overthinking), không tự ý sửa ngoài phạm vi gây lệch hướng.
2. **Kiểm soát overengineering**:
   - Không mở rộng kiến trúc quá mức cần thiết; chỉ triển khai những thành phần hợp lý, thực sự cần thiết và đã được kiểm chứng.
3. **Giải thích dễ hiểu cho người dùng bằng tiếng Việt**:
   - Khi trình bày kết quả hoặc giải thích kỹ thuật, luôn **Việt hóa hoặc chú thích rõ các thuật ngữ chuyên ngành**, giải nghĩa chức năng cụ thể để người dùng nắm vững hệ thống.

## 4. Quản lý môi trường, Docker & GitHub

1. **Không tạo nhiều môi trường thử nghiệm phân tán**:
   - Tránh sinh nhiều môi trường test rời rạc gây khó đồng bộ và khó quản lý.
   - Với Docker, làm việc trực tiếp trên các container/image chính của dự án; không tự ý tạo thêm stack hoặc container dư thừa.
2. **Quản lý Docker container & image gọn gàng, tối ưu**:
   - Đồng bộ image mới nhất sau khi thay đổi mã nguồn.
   - Khi image mới đã chạy ổn định, dọn dẹp sạch các image/build cache dư thừa không còn tham chiếu để tránh chiếm dụng tài nguyên.
   - Luôn **giữ lại đúng 1 bản image gần nhất** (ví dụ tag `:fallback`) làm phương án dự phòng khôi phục nhanh.
   - Không chạm vào container hoặc image của các dự án khác chạy cùng máy.
3. **Thao tác GitHub cẩn thận, sạch sẽ**:
   - Công việc nhỏ/gọn: làm trực tiếp trên nhánh `main`, hoàn thành thì commit và push ngay lên GitHub.
   - Công việc lớn/phức tạp: tạo nhánh (branch) riêng để phát triển, kiểm thử đầy đủ rồi mới merge gọn gàng vào `main`, push lên `origin` và dọn nhánh đã merge.

## 5. Quy trình sửa lỗi, sẵn sàng kiểm thử & Báo cáo

1. **Báo cáo chi tiết sau mỗi lần hoàn thành**:
   - Nêu rõ: đã triển khai/thay đổi những gì, chức năng nào chạy thành công, kết quả kiểm thử thực tế và những lỗi/rủi ro nào (nếu còn).
2. **Điều tra tận gốc khi sửa lỗi (Root-Cause Fix)**:
   - Phải truy nguyên nhân gốc rễ, không sửa chắp vá ở bề mặt để tránh lỗi tái diễn ở đường chạy khác.
3. **Triển khai chắc chắn, hiệu quả, tối ưu**:
   - Mọi thay đổi đều phải được xác minh bằng lint, test tự động và kiểm tra hoạt động thực tế trước khi kết thúc.
4. **Giữ các provider AI/LLM/Parse và tính năng luôn bật sẵn sàng**:
   - Không tắt ngầm các provider (Gemini, Ollama, Qdrant, PostgreSQL, Redis, Neo4j, LlamaParse...) hoặc vô hiệu hóa tính năng ngoài ý muốn, đảm bảo người dùng có thể kiểm thử hệ thống bất kỳ lúc nào.
