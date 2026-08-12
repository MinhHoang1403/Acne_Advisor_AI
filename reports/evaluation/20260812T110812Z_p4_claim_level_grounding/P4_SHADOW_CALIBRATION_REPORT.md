# P4 Shadow Calibration Report

Fixture: frozen provider-free `p4_claim_grounding_fixture_v1`, 32 pairs, 14 critical. Vietnamese and mixed Vietnamese/English cases are included. This calibration validates deterministic plumbing; it is not a claim of live-model clinical calibration.

## Claim results

| Case | Claim ID | Claim | Critical? | Evidence IDs | Gold | Predicted | Correct? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| supported_tazorac_alias | claim:fbc59acc40bb9eb9 | Tazorac chứa tazarotene, một retinoid bôi điều trị mụn. | No | e-s01 | SUPPORTED | SUPPORTED | Yes |
| supported_adapalene_class | claim:4c861561ca1fe51b | Adapalene là một retinoid bôi dùng để điều trị mụn. | No | e-s02 | SUPPORTED | SUPPORTED | Yes |
| supported_bp_not_antibiotic | claim:305fee3178bcf89c | Benzoyl peroxide không phải là kháng sinh. | Yes | e-s03 | SUPPORTED | SUPPORTED | Yes |
| supported_clindamycin_antibiotic | claim:b1f2470a8362e042 | Clindamycin là kháng sinh bôi tại chỗ dùng trong điều trị mụn. | Yes | e-s04 | SUPPORTED | SUPPORTED | Yes |
| supported_blackhead_clogging | claim:1a1415642bdab605 | Mụn đầu đen hình thành khi lỗ chân lông bị bít tắc bởi bã nhờn và tế bào chết. | No | e-s05 | SUPPORTED | SUPPORTED | Yes |
| supported_blackhead_oxidation | claim:b273f5078ff2afe8 | Màu đen của mụn đầu đen liên quan đến quá trình oxy hóa khi nhân mụn tiếp xúc với không khí. | No | e-s06 | SUPPORTED | SUPPORTED | Yes |
| supported_bp_antimicrobial | claim:161171f27b1618f7 | Benzoyl peroxide có tác dụng kháng khuẩn với C. acnes và hỗ trợ giảm bít tắc. | No | e-s07 | SUPPORTED | SUPPORTED | Yes |
| supported_topical_irritation | claim:0a61c0fb2986abe1 | Retinoid bôi có thể gây khô, đỏ và bong tróc da. | No | e-s08 | SUPPORTED | SUPPORTED | Yes |
| supported_epiduo_composition | claim:f7ed293788710026 | Epiduo chứa adapalene và benzoyl peroxide. | No | e-s09 | SUPPORTED | SUPPORTED | Yes |
| supported_mixed_language | claim:55d3f12b599c149f | Topical retinoids help reduce comedonal acne and follicular plugging. | No | e-s10 | SUPPORTED | SUPPORTED | Yes |
| supported_pregnancy_contraindication | claim:521c560e5e76c955 | Tazarotene không được dùng trong thai kỳ. | Yes | e-s11 | SUPPORTED | SUPPORTED | Yes |
| supported_emergency_action | claim:8061ba0cdbde0267 | Khó thở kèm sưng lưỡi sau dùng thuốc cần được cấp cứu ngay. | Yes | e-s12 | SUPPORTED | SUPPORTED | Yes |
| partial_retinoid_overclaim | claim:db5d5423a88ec4e9 | Retinoid bôi luôn luôn chữa khỏi hoàn toàn mụn đầu đen trong một tuần. | No | e-p01 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | Yes |
| partial_bp_speed_overclaim | claim:6fb138e2260cf924 | Benzoyl peroxide luôn luôn loại bỏ hoàn toàn C. acnes trong một tuần. | No | e-p02 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | Yes |
| partial_adapalene_overclaim | claim:824c188bfdb55c65 | Adapalene luôn luôn làm sạch hoàn toàn mọi nhân mụn trong một tuần. | No | e-p03 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | Yes |
| partial_tazarotene_guarantee | claim:0e5bad2c17b27b6c | Tazarotene chắc chắn chữa khỏi hoàn toàn mụn trong một tuần. | No | e-p04 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | Yes |
| partial_antibiotic_overclaim | claim:1faa029ab66ef239 | Clindamycin luôn luôn chữa khỏi hoàn toàn mọi mụn viêm trong một tuần. | Yes | e-p05 | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | Yes |
| unsupported_adapalene_bleaching | claim:8e334ea92f26befd | Adapalene làm bạc màu tóc, quần áo và khăn trải giường sau mỗi lần bôi. | No | e-u01 | UNSUPPORTED | UNSUPPORTED | Yes |
| unsupported_bp_oral | claim:474abbdd1bc2710d | Benzoyl peroxide là thuốc uống toàn thân được hấp thu để điều trị mọi dạng mụn nặng. | No | e-u02 | UNSUPPORTED | UNSUPPORTED | Yes |
| unsupported_clindamycin_mechanism | claim:bd2d006ac9fb8c51 | Clindamycin trực tiếp hòa tan nút sừng và mở mọi lỗ chân lông bị bít tắc. | No | e-u03 | UNSUPPORTED | UNSUPPORTED | Yes |
| unsupported_tazarotene_sunscreen | claim:4ff89df54761cfed | Tazarotene thay thế hoàn toàn kem chống nắng và bảo vệ da khỏi mọi tia cực tím. | No | e-u04 | UNSUPPORTED | UNSUPPORTED | Yes |
| unsupported_antibiotic_monotherapy_safety | claim:8ac44b5e40c79c27 | Clindamycin an toàn tuyệt đối khi dùng đơn độc kéo dài cho mọi người bị mụn. | Yes | e-u05 | UNSUPPORTED | UNSUPPORTED | Yes |
| contradicted_bp_antibiotic | claim:38978682ba27e311 | Benzoyl peroxide là kháng sinh. | Yes | e-c01 | CONTRADICTED | CONTRADICTED | Yes |
| contradicted_adapalene_antibiotic | claim:87e14aac6d66fa08 | Adapalene là kháng sinh. | Yes | e-c02 | CONTRADICTED | CONTRADICTED | Yes |
| contradicted_clindamycin_retinoid | claim:ed507fa4acece528 | Clindamycin là retinoid. | Yes | e-c03 | CONTRADICTED | CONTRADICTED | Yes |
| contradicted_pregnancy | claim:f0a7c6ecb22ba542 | Tazarotene an toàn khi mang thai. | Yes | e-c04 | CONTRADICTED | CONTRADICTED | Yes |
| contradicted_emergency | claim:d0103ed0bad6ff56 | Khó thở và sưng lưỡi sau dùng thuốc không cần cấp cứu. | Yes | e-c05 | CONTRADICTED | CONTRADICTED | Yes |
| no_evidence_adapalene_scar | claim:f3d1d4dd7df2dc9b | Adapalene xóa hoàn toàn mọi sẹo lõm do mụn. | No | none | NO_EVIDENCE | NO_EVIDENCE | Yes |
| no_evidence_invalid_provenance | claim:b1912bb4d1197872 | Benzoyl peroxide xóa hoàn toàn mọi vết thâm sau mụn. | No | none | NO_EVIDENCE | NO_EVIDENCE | Yes |
| no_evidence_pregnancy | claim:f0a7c6ecb22ba542 | Tazarotene an toàn khi mang thai. | Yes | none | NO_EVIDENCE | NO_EVIDENCE | Yes |
| no_evidence_antibiotic | claim:4f1c74102430bbd7 | Clindamycin dùng đơn độc lâu dài không làm tăng nguy cơ kháng kháng sinh. | Yes | none | NO_EVIDENCE | NO_EVIDENCE | Yes |
| no_evidence_source_claim | claim:a5aaf2b815ffef31 | Tài liệu hiện có xác nhận isotretinoin an toàn tuyệt đối cho mọi người bị mụn. | Yes | none | NO_EVIDENCE | NO_EVIDENCE | Yes |

## Confusion matrix

| Gold \ Predicted | SUPPORTED | PARTIAL | UNSUPPORTED | CONTRADICTED | NO_EVIDENCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| SUPPORTED | 12 | 0 | 0 | 0 | 0 |
| PARTIALLY_SUPPORTED | 0 | 5 | 0 | 0 | 0 |
| UNSUPPORTED | 0 | 0 | 5 | 0 | 0 |
| CONTRADICTED | 0 | 0 | 0 | 5 | 0 |
| NO_EVIDENCE | 0 | 0 | 0 | 0 | 5 |

## Evidence mapping

| Claim group | Candidate evidence | Chosen evidence | Provenance valid? | Mapping correct? |
| --- | --- | --- | --- | --- |
| Supported (12) | e-s01..e-s12 | e-s01..e-s12 | Yes | 12/12 |
| Partial (5) | e-p01..e-p05 | e-p01..e-p05 | Yes | 5/5 |
| Unsupported (5) | e-u01..e-u05 | e-u01..e-u05 | Yes | 5/5 |
| Contradicted (5) | e-c01..e-c05 | e-c01..e-c05 | Yes | 5/5 |
| No evidence (5) | none/invalid/unrelated | none | N/A | 5/5 |

## Critical claims

| Critical case | Gold | Predicted | Shadow action | False allow? |
| --- | --- | --- | --- | --- |
| supported_bp_not_antibiotic | SUPPORTED | SUPPORTED | WOULD_ALLOW | No |
| supported_clindamycin_antibiotic | SUPPORTED | SUPPORTED | WOULD_ALLOW | No |
| supported_pregnancy_contraindication | SUPPORTED | SUPPORTED | WOULD_ALLOW | No |
| supported_emergency_action | SUPPORTED | SUPPORTED | WOULD_ALLOW | No |
| partial_antibiotic_overclaim | PARTIALLY_SUPPORTED | PARTIALLY_SUPPORTED | WOULD_BLOCK_CRITICAL | No |
| unsupported_antibiotic_monotherapy_safety | UNSUPPORTED | UNSUPPORTED | WOULD_BLOCK_CRITICAL | No |
| contradicted_bp_antibiotic | CONTRADICTED | CONTRADICTED | WOULD_BLOCK_CRITICAL | No |
| contradicted_adapalene_antibiotic | CONTRADICTED | CONTRADICTED | WOULD_BLOCK_CRITICAL | No |
| contradicted_clindamycin_retinoid | CONTRADICTED | CONTRADICTED | WOULD_BLOCK_CRITICAL | No |
| contradicted_pregnancy | CONTRADICTED | CONTRADICTED | WOULD_BLOCK_CRITICAL | No |
| contradicted_emergency | CONTRADICTED | CONTRADICTED | WOULD_BLOCK_CRITICAL | No |
| no_evidence_pregnancy | NO_EVIDENCE | NO_EVIDENCE | WOULD_BLOCK_CRITICAL | No |
| no_evidence_antibiotic | NO_EVIDENCE | NO_EVIDENCE | WOULD_BLOCK_CRITICAL | No |
| no_evidence_source_claim | NO_EVIDENCE | NO_EVIDENCE | WOULD_BLOCK_CRITICAL | No |

## Shadow answer comparison

- Fully supported Tazorac case: original claim and verified projection are identical; action `WOULD_ALLOW`.
- Retinoid one-week cure overclaim: original claim is retained for production because mode is shadow; verified projection excludes it; action `WOULD_REWRITE_PARTIAL`.
- Pregnancy contradiction: original claim is retained for production solely for calibration; verified projection excludes it; action `WOULD_BLOCK_CRITICAL`. Existing P0/P3 safety remains authoritative in production.
- No-evidence source claim: verified projection is empty; action `WOULD_BLOCK_CRITICAL`.

Shadow action totals: 12 allow, 4 rewrite-partial, 10 block-critical, 6 abstain, 0 verifier-unavailable. Production answer modifications: 0/32.
