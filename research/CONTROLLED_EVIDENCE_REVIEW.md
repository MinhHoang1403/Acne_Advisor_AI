# Review nhanh 8 controlled evidence mappings

Trạng thái: `RESEARCHER_REVIEW_REQUIRED`. Đây là đối chiếu kỹ thuật giữa câu hỏi, text
chunk và provenance; không phải phê duyệt tính đúng y khoa. Mọi excerpt dưới đây được
chép nguyên văn từ payload active build `d4a1819fe7fb77fe1f40`.

Quy ước: các chunk trong cùng một ô “alternatives” có thể thay nhau theo mapping hiện
tại; “joint” cho biết có cần ghép nhiều chunk để đáp ứng một aspect hay không.

## 1. `bp_antibiotic`

**Question:** Benzoyl peroxide có phải kháng sinh không?

| Aspect | Evidence, excerpt và provenance | Role / cấu trúc | Người nghiên cứu cần xác nhận |
|---|---|---|---|
| `bp_classification` — phân biệt benzoyl peroxide với kháng sinh bôi | `31daf417-f490-51ac-a2fb-ff1248d8eb7e`: “Benzoyl peroxide is a topical antiseptic…” — DermNet, *Acne In Children*, `web_dermnetnz_8d930888c730deaa`, <https://dermnetnz.org/topics/acne-in-children>. `fd9eb6fa-9f14-5147-ab66-cf67eb45fddd`: “Benzoyl peroxide: This active ingredient reduces acne-causing bacteria on your skin… another acne medication like a retinoid or an antibiotic.” — AAD, *Acne: Treat*, `web_aad_8b5ac304bb90a8f1`, <https://www.aad.org/public/diseases/acne/derm-treat/treat>. | `31daf417…`: `DIRECT_SUPPORT`; `fd9eb6fa…`: `PARTIAL_SUPPORT`, `QUALIFIER`. Alternatives; joint: không theo mapping hiện tại. | `31daf417…` có đủ trực tiếp để kết luận phân loại không; `fd9eb6fa…` có đủ nếu đứng một mình hay chỉ là qualifier bổ sung. |

- [ ] APPROVE
- [ ] APPROVE_WITH_CHANGES
- [ ] REJECT
- [ ] NEEDS_DOMAIN_EXPERT

## 2. `adapalene_class`

**Question:** Adapalene thuộc nhóm thuốc gì?

| Aspect | Evidence, excerpt và provenance | Role / cấu trúc | Người nghiên cứu cần xác nhận |
|---|---|---|---|
| `adapalene_classification` — adapalene nằm trong nhóm retinoid bôi | `460222d6-c4ba-56f9-a26c-bf255b6afb39`: section `Điều trị mụn trứng cá / Các thuốc điều trị cụ thể / Retinoid tại chỗ`; text: “Lựa chọn một trong các hoạt chất sau: - Adapalen (0,1%, 0,3% kem hoặc lotion 0,1%)…” — Bộ Y tế VN, `vn_moh_dermatology_4416_2023_acne`, <https://moh.gov.vn/>. | `DIRECT_SUPPORT`; không alternative; joint: không. Evidence dựa trên heading cộng với mục liệt kê. | Heading có được chấp nhận là một phần của bằng chứng trực tiếp hay cần chunk có câu khẳng định đầy đủ trong body. |

- [ ] APPROVE
- [ ] APPROVE_WITH_CHANGES
- [ ] REJECT
- [ ] NEEDS_DOMAIN_EXPERT

## 3. `clindamycin_monotherapy`

**Question:** Có nên dùng clindamycin đơn độc không?

| Aspect | Evidence, excerpt và provenance | Role / cấu trúc | Người nghiên cứu cần xác nhận |
|---|---|---|---|
| `topical_antibiotic_monotherapy` — kháng sinh bôi không được khuyến nghị đơn trị liệu | `2719272a-82f7-5052-a06f-34efa505ac31`: “Remark: Topical antibiotic monotherapy is not recommended.” — AAD guideline 2024, section `Management of Acne Vulgaris / Table III`, `aad_acne_guideline_2024`, <https://www.jaad.org/article/S0190-9622(23)03389-3/fulltext>. `06a0e33d-b9f6-53c0-b5ed-a163ec1e0d5b`: “they may be used … as monotherapy (except topical antibiotics)…” — cùng guideline, section `Acne Treatment Recommendations / Topical Therapies`. | Cả hai: `DIRECT_SUPPORT`, `QUALIFIER`. Alternatives; joint: không. | Khẳng định về toàn bộ topical antibiotics có đủ để trả lời riêng clindamycin không; hai chunk có thực sự là alternatives tương đương. |

- [ ] APPROVE
- [ ] APPROVE_WITH_CHANGES
- [ ] REJECT
- [ ] NEEDS_DOMAIN_EXPERT

## 4. `adapalene_bp_comparison`

**Question:** Adapalene và benzoyl peroxide khác nhau thế nào?

| Aspect | Evidence, excerpt và provenance | Role / cấu trúc | Người nghiên cứu cần xác nhận |
|---|---|---|---|
| `adapalene_classification` — đặc điểm phân loại của adapalene | `460222d6-c4ba-56f9-a26c-bf255b6afb39`: heading `Retinoid tại chỗ`; “Lựa chọn một trong các hoạt chất sau: - Adapalen…” — Bộ Y tế VN, `vn_moh_dermatology_4416_2023_acne`, <https://moh.gov.vn/>. | `DIRECT_SUPPORT`; bắt buộc theo mapping đề xuất; joint với một chunk BP để trả lời hai phía so sánh. | Heading + body có đủ trực tiếp và nội dung “phân loại” có đủ cho yêu cầu “khác nhau thế nào” không. |
| `bp_classification` — đặc điểm phân loại của benzoyl peroxide | `31daf417…`: “Benzoyl peroxide is a topical antiseptic…” — DermNet, `web_dermnetnz_8d930888c730deaa`. Alternative `fd9eb6fa…`: “This active ingredient reduces acne-causing bacteria…” — AAD, `web_aad_8b5ac304bb90a8f1`. | `31daf417…`: `DIRECT_SUPPORT`; `fd9eb6fa…`: `PARTIAL_SUPPORT`. Bắt buộc theo mapping đề xuất; alternatives; joint với evidence adapalene. | Partial AAD có đủ thay direct DermNet không; hai aspect phân loại đã đủ tối thiểu cho câu hỏi so sánh chưa. |
| `fixed_combination_exists` — nguồn nêu phối hợp hai hoạt chất | `12259038-373a-5ead-a76c-6e536aef00b9`: “Sản phẩm kết hợp các thuốc bôi như: - Adapalene + Benzoyl Peroxide…” — Bộ Y tế VN, section `Sản Phẩm Điều Trị Kết Hợp`, `vn_moh_dermatology_4416_2023_acne`, <https://moh.gov.vn/>. | `DIRECT_SUPPORT`; **hữu ích bổ sung**, chưa mặc định bắt buộc. | **REVIEW_REQUIRED:** câu hỏi hỏi sự khác nhau, không trực tiếp hỏi có chế phẩm phối hợp hay không. Xác nhận nên là aspect bắt buộc, optional, hay loại khỏi gold. |

- [ ] APPROVE hai aspect phân loại là core; combination là optional
- [ ] APPROVE_WITH_CHANGES
- [ ] REJECT
- [ ] NEEDS_DOMAIN_EXPERT

## 5. `retinoid_pregnancy`

**Question:** Retinoid có dùng khi mang thai không?

| Aspect | Evidence, excerpt và provenance | Role / cấu trúc | Người nghiên cứu cần xác nhận |
|---|---|---|---|
| `topical_retinoid_pregnancy` — lời khuyên tránh/ngừng cùng qualifier theo hoạt chất | `193a0908-65e2-5013-83c3-0c7c35432152`: “Adapalene … Most experts recommend stopping this treatment during pregnancy.”; cùng chunk cũng tách isotretinoin/tazarotene thành “never to take” — AAD, `web_aad_88a4258288696392`, <https://www.aad.org/public/diseases/acne/derm-treat/pregnancy>. `5e32098d-cf83-5445-8d71-55750cf96ab5`: “Do not use topical preparations containing: Topical retinoids (tretinoin, isotretinoin and adapalene)” — DermNet, `web_dermnetnz_bd03071a0f669923`, <https://dermnetnz.org/topics/acne-in-pregnancy>. | Cả hai: `DIRECT_SUPPORT`, `QUALIFIER`; alternatives; joint: không. | Câu hỏi dùng “retinoid” rộng: xác nhận scope topical/oral, mức chắc chắn và liệu hai chunk có thể thay nhau mà không làm mất qualifier. |

- [ ] APPROVE
- [ ] APPROVE_WITH_CHANGES
- [ ] REJECT
- [ ] NEEDS_DOMAIN_EXPERT

## 6. `dermatologist_referral`

**Question:** Khi nào cần gặp bác sĩ da liễu?

| Aspect | Evidence, excerpt và provenance | Role / cấu trúc | Người nghiên cứu cần xác nhận |
|---|---|---|---|
| `referral_triggers` — các tình huống cụ thể nên gặp bác sĩ da liễu | `0564b723-4643-5aa0-ae71-c322c9871dd7`: “You have deep, painful acne like the acne cysts and nodules…”; “As your acne clears, you can see scars or dark spots.”; “If you answered ‘yes’ to six or more of the questions, you should see a dermatologist.” — AAD, *Acne: When Derm*, `web_aad_502603ec115bf029`, <https://www.aad.org/public/diseases/acne/diy/when-derm>. | `DIRECT_SUPPORT`, `QUALIFIER`; không alternative; joint: không. | Xác nhận các trigger nào phải giữ và ngưỡng “six or more” có phải qualifier bắt buộc hay chỉ thuộc công cụ tự đánh giá của nguồn. |

- [ ] APPROVE
- [ ] APPROVE_WITH_CHANGES
- [ ] REJECT
- [ ] NEEDS_DOMAIN_EXPERT

## 7. `historical_02`

**Question:** Mụn đầu trắng khác mụn đầu đen ở điểm chính nào?

| Aspect | Evidence, excerpt và provenance | Role / cấu trúc | Người nghiên cứu cần xác nhận |
|---|---|---|---|
| `open_closed_comedones` — phân biệt nhân đóng và nhân mở | `1a5c4052-4796-5a98-b56a-331ed3d98d51`: “Mụn nhân đóng (closed comedone) hay mụn đầu trắng… hình vòm, nhẵn…”; “Mụn nhân mở (open comedone) hay mụn đầu đen… trung tâm có lỗ mở chứa chất sừng xám, nâu hoặc đen.” — Bộ Y tế VN, section `TRỨNG CÁ / 2. CHẨN ĐOÁN / 2.1. Triệu chứng lâm sàng`, `vn_moh_dermatology_4416_2023_acne`, <https://moh.gov.vn/>. | `DIRECT_SUPPORT`; một chunk hỗ trợ cả hai phía; không alternative; joint: không. | Xác nhận đây là khác biệt “điểm chính” đủ trực tiếp và không cần thêm giải thích về oxy hóa/màu đen. |

- [ ] APPROVE
- [ ] APPROVE_WITH_CHANGES
- [ ] REJECT
- [ ] NEEDS_DOMAIN_EXPERT

## 8. `paraphrase_inflammatory_lesions`

**Question:** Các tổn thương viêm thường gặp của mụn trứng cá gồm những dạng nào?

| Aspect | Evidence, excerpt và provenance | Role / cấu trúc | Người nghiên cứu cần xác nhận |
|---|---|---|---|
| `inflammatory_lesion_types` — sẩn viêm, mụn mủ/sẩn mủ, cục hoặc nang | `38da2786-b72d-546e-a351-d6dff6dc0880`: “tổn thương viêm như các sẩn viêm, sẩn mủ, cục, nang.” — Bộ Y tế VN, section `TRỨNG CÁ / 1. ĐẠI CƯƠNG / 1.1. Khái niệm`. Alternative `1a5c4052…`: “Mụn mủ (pustule)… Sẩn mủ (papulopustular)…” — section `2.1. Triệu chứng lâm sàng`. Alternative `35c3c35a-4f3a-55d8-b6c3-d7ee51efea0b`: “Cục (Nodule)… viêm sâu…”; “Nang (Cyst)… tổn thương viêm…” — section `Các Loại Tổn Thương`. Tất cả thuộc `vn_moh_dermatology_4416_2023_acne`, <https://moh.gov.vn/>. | `38da2786…`, `1a5c4052…`: `DIRECT_SUPPORT`; `35c3c35a…`: `PARTIAL_SUPPORT`. Alternatives theo mapping hiện tại; joint: không. | `35c3c35a…` chỉ nêu cục/nang nên có đủ đứng một mình không; danh sách đầy đủ tối thiểu cần những loại nào. |

- [ ] APPROVE
- [ ] APPROVE_WITH_CHANGES
- [ ] REJECT
- [ ] NEEDS_DOMAIN_EXPERT

## Quyết định còn mở

1. Phê duyệt/sửa 8 mappings, đặc biệt các alternative `PARTIAL_SUPPORT`.
2. Quyết định `fixed_combination_exists` là core, optional hay loại khỏi case comparison.
3. Quyết định qualifier bắt buộc cho BP, topical antibiotic, pregnancy và referral.

Không có checkbox nào được tự đánh dấu trong nghiên cứu này.
