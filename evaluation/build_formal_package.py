"""Build the researcher-reviewed benchmark artifacts from the active corpus.

This operator utility is deliberately separate from production runtime. It reads
the parsed corpus, embedding cache, and active Qdrant alias; it never parses,
embeds, indexes, activates, or writes a datastore. The generated benchmark stays
pending until a researcher reviews its questions, gold claims, and absence audits.
"""

# ruff: noqa: E402 -- direct script execution needs repository-root bootstrap.

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

from evaluation.formal_evaluation_support import canonical_json_sha256, write_pretty_json
from src.database.vector_store import qdrant_client_kwargs
from src.ingestion.bm25 import BM25_VECTOR_NAME, bm25_document
from src.ingestion.embedding import EmbeddingCache
from src.ingestion.parser import artifact_path, load_parsed_artifact
from src.ingestion.pipeline import (
    DEFAULT_EMBEDDING_CACHE,
    DEFAULT_PARSED_CACHE,
    DEFAULT_SOURCE_DIR,
    DEFAULT_SOURCE_MANIFEST,
    DEFAULT_TAXONOMY,
    _compile_prepared_knowledge,
)
from src.ingestion.source_manifest import load_source_manifest


OUTPUT_DIR = ROOT / "evaluation"
BASE_SHA = "6a1809c4ddedbccab986ec76eb730321686ff3ff"
KB_BUILD_ID = "94d613bc9b33628de3ef"
EVALUATOR_MODEL = "gpt-5.4-mini-2026-03-17"
RAGCHECKER_VERSION = "0.1.9"
RAGCHECKER_COMMIT = "9017b3263b82ea5354948c4db40a65bba94d779e"
RAGCHECKER_PACKAGE = "ragchecker==0.1.9"
KNOWLEDGE_ALIAS = "acne_knowledge"
DEVELOPMENT_REGISTRY_PATHS = ("tests", "docs", "scripts")
MANUAL_DEVELOPMENT_EXCLUSIONS = [
    {"pattern_id": "repeated_referral", "query": "Khi nào tình trạng mụn cần đi khám bác sĩ da liễu?"},
    {"pattern_id": "adapalene_bp_comparison", "query": "Adapalene và benzoyl peroxide khác nhau thế nào?"},
    {"pattern_id": "absolute_permanent_cure", "query": "Benzoyl peroxide có chữa khỏi mụn vĩnh viễn 100% không?"},
    {"pattern_id": "bp_pronoun_follow_up", "query": "Benzoyl peroxide có phải kháng sinh không? Hoạt chất này có tác dụng gì?"},
    {"pattern_id": "explicit_topic_switch", "query": "Bỏ qua adapalene. Benzoyl peroxide có gây kích ứng không?"},
]


# Claims are source annotations made before any evaluator run. Every claim points
# to one or more active chunk IDs and intentionally preserves source qualifiers.
CLAIMS: dict[str, dict[str, Any]] = {
    "acne_definition": {
        "text": "Mụn trứng cá là bệnh viêm mạn tính của đơn vị nang lông - tuyến bã.",
        "chunks": ["904d057a-aa4d-5d6a-bcb7-7379c5137b60"],
    },
    "acne_lesions": {
        "text": "Tổn thương có thể gồm nhân mụn mở hoặc đóng, sẩn, mụn mủ và nốt, thường ở mặt hoặc thân mình.",
        "chunks": ["904d057a-aa4d-5d6a-bcb7-7379c5137b60"],
    },
    "acne_lesion_types": {
        "text": "Tổn thương có thể gồm nhân mụn mở hoặc đóng, sẩn, mụn mủ và nốt.",
        "chunks": ["904d057a-aa4d-5d6a-bcb7-7379c5137b60"],
    },
    "pathogenesis": {
        "text": "Sinh bệnh học mụn có nhiều yếu tố, gồm tăng sừng hóa nang lông, C. acnes, bã nhờn và viêm.",
        "chunks": ["e35a46c1-1733-5225-b02c-9fb0c198d26d"],
    },
    "c_acnes_identity": {
        "text": "Cutibacterium acnes, trước đây gọi là Propionibacterium acnes, là trực khuẩn Gram dương kỵ khí.",
        "chunks": ["056b77fc-8b05-5da9-a46a-7cc57fb32f42"],
    },
    "mild_moderate": {
        "text": "Theo phân loại NICE, mụn nhẹ đến vừa có thể có bất kỳ số nhân mụn, tối đa 34 tổn thương viêm và tối đa 2 nốt.",
        "chunks": ["3a600fda-b952-5feb-bd28-f757f9539a60"],
    },
    "moderate_severe": {
        "text": "Theo phân loại NICE, mụn vừa đến nặng có từ 35 tổn thương viêm trở lên và/hoặc từ 3 nốt trở lên.",
        "chunks": ["8db0c5ee-c2ec-59f2-96ff-9ee31ed59e0e"],
    },
    "comedone_types": {
        "text": "Nhân mụn mở là mụn đầu đen, còn nhân mụn đóng là mụn đầu trắng do lỗ nang lông bị bít hoàn toàn.",
        "chunks": ["ac93995d-741e-53f0-8b8c-72252cf81119"],
    },
    "comedone_names": {
        "text": "Nhân mụn mở là mụn đầu đen, còn nhân mụn đóng là mụn đầu trắng.",
        "chunks": ["ac93995d-741e-53f0-8b8c-72252cf81119"],
    },
    "blackhead_color": {
        "text": "Màu đen của nhân mụn mở liên quan đến melanin chứ không phải bụi bẩn.",
        "chunks": ["ac93995d-741e-53f0-8b8c-72252cf81119"],
    },
    "scar_types": {
        "text": "Sẹo teo do mụn thường được chia thành sẹo đáy nhọn, sẹo lượn sóng và sẹo đáy vuông.",
        "chunks": ["c5a99f3c-abf4-5ed3-be4a-6acdf0b82d3a"],
    },
    "postinflammatory_pigment": {
        "text": "Tăng sắc tố sau viêm xảy ra do melanin lắng đọng trong vùng da vừa bị tổn thương viêm.",
        "chunks": ["df9c7438-a085-5063-9824-a96830e2224a"],
    },
    "topical_mainstay": {
        "text": "Điều trị bôi là nền tảng của điều trị mụn và có thể dùng khi bắt đầu, duy trì hoặc phối hợp với thuốc khác.",
        "chunks": ["f675512f-93f5-572a-b705-35132a1445b7"],
    },
    "topical_multimodal": {
        "text": "Điều trị bôi phối hợp nhiều cơ chế được khuyến nghị để tối ưu hiệu quả và giảm nguy cơ kháng kháng sinh.",
        "chunks": ["f675512f-93f5-572a-b705-35132a1445b7"],
    },
    "bp_type_mechanism": {
        "text": "Benzoyl peroxide là hoạt chất kháng khuẩn bôi không kê đơn.",
        "chunks": ["db4632c5-44ed-5dea-80c2-addb0a2534b3"],
    },
    "bp_antimicrobial_role": {
        "text": "Benzoyl peroxide là hoạt chất kháng khuẩn bôi.",
        "chunks": ["db4632c5-44ed-5dea-80c2-addb0a2534b3"],
    },
    "bp_oxygen_mechanism": {
        "text": "Benzoyl peroxide giải phóng các gốc oxy tự do.",
        "chunks": ["db4632c5-44ed-5dea-80c2-addb0a2534b3"],
    },
    "bp_comedolytic": {
        "text": "Benzoyl peroxide có tác dụng tiêu nhân mụn nhẹ.",
        "chunks": ["db4632c5-44ed-5dea-80c2-addb0a2534b3"],
    },
    "bp_resistance": {
        "text": "Tài liệu không ghi nhận C. acnes kháng benzoyl peroxide.",
        "chunks": ["db4632c5-44ed-5dea-80c2-addb0a2534b3"],
    },
    "bp_adverse": {
        "text": "Benzoyl peroxide có thể gây rát, châm chích, khô, đỏ, đau, bong tróc, kích ứng và làm bạc màu vải.",
        "chunks": ["db4632c5-44ed-5dea-80c2-addb0a2534b3"],
    },
    "retinoid_role": {
        "text": "Retinoid bôi có tác dụng tiêu nhân mụn và chống viêm, hỗ trợ giảm rối loạn sắc tố và duy trì độ sạch mụn.",
        "chunks": ["850609c9-471a-54cc-934f-478ad48d826d"],
    },
    "retinoid_comedolytic_antiinflammatory": {
        "text": "Retinoid bôi có tác dụng tiêu nhân mụn và chống viêm.",
        "chunks": ["850609c9-471a-54cc-934f-478ad48d826d"],
    },
    "retinoid_agents": {
        "text": "Các retinoid bôi được nêu gồm tretinoin, adapalene, tazarotene và trifarotene.",
        "chunks": ["850609c9-471a-54cc-934f-478ad48d826d"],
    },
    "avoid_antibiotic_mono": {
        "text": "Không nên dùng kháng sinh bôi hoặc kháng sinh uống làm đơn trị liệu cho mụn.",
        "chunks": ["a9d1f7ea-e047-506d-b951-1f80bf38f31a"],
    },
    "avoid_topical_antibiotic_mono": {
        "text": "Không nên dùng kháng sinh bôi làm đơn trị liệu cho mụn.",
        "chunks": ["a9d1f7ea-e047-506d-b951-1f80bf38f31a"],
    },
    "bp_with_antibiotic": {
        "text": "Dùng benzoyl peroxide cùng kháng sinh bôi có thể hạn chế sự phát triển kháng kháng sinh.",
        "chunks": ["e4e54d11-c8bf-56eb-8e41-b7be9ecea11e"],
    },
    "oral_antibiotic_course": {
        "text": "Kháng sinh uống nên dùng trong thời gian ngắn nhất có thể, thường không quá 3 đến 4 tháng.",
        "chunks": ["47e2209f-efbe-5807-b7a9-f48df661923e"],
    },
    "oral_antibiotic_combination": {
        "text": "Khi dùng kháng sinh uống trị mụn, hướng dẫn khuyến nghị dùng cùng benzoyl peroxide và thuốc bôi khác.",
        "chunks": ["47e2209f-efbe-5807-b7a9-f48df661923e"],
    },
    "first_line_combinations": {
        "text": "NICE liệt kê phối hợp cố định adapalene bôi với benzoyl peroxide bôi là một lựa chọn đầu tay cho mọi mức độ mụn.",
        "chunks": ["9f2c542c-d365-51c5-be89-55c0ee80d22a"],
    },
    "combination_advantage": {
        "text": "Các sản phẩm phối hợp adapalene-benzoyl peroxide, clindamycin-tretinoin hoặc erythromycin-isotretinoin có hiệu quả cao hơn đơn trị một thành phần theo nguồn Bộ Y tế.",
        "chunks": ["12259038-373a-5ead-a76c-6e536aef00b9"],
    },
    "maintenance": {
        "text": "Sau khi hoàn tất điều trị, điều trị duy trì không phải lúc nào cũng cần.",
        "chunks": ["6f0ad48a-89fd-522b-8651-ad13e98e24c9"],
    },
    "maintenance_relapse": {
        "text": "Có thể cân nhắc điều trị duy trì ở người có tiền sử thường xuyên tái phát sau điều trị.",
        "chunks": ["6f0ad48a-89fd-522b-8651-ad13e98e24c9"],
    },
    "maintenance_adapalene_bp": {
        "text": "Có thể cân nhắc phối hợp adapalene-benzoyl peroxide bôi làm điều trị duy trì.",
        "chunks": ["6f0ad48a-89fd-522b-8651-ad13e98e24c9"],
    },
    "isotretinoin_indication": {
        "text": "Có thể cân nhắc isotretinoin uống cho người trên 12 tuổi bị mụn nặng kháng các liệu trình chuẩn, như mụn nốt nang, mụn conglobata, mụn fulminans hoặc nguy cơ sẹo vĩnh viễn.",
        "chunks": ["26c4c35a-1421-5b5e-a331-f08bee903d74"],
    },
    "isotretinoin_mechanism": {
        "text": "Isotretinoin làm giảm kích thước và bài tiết tuyến bã, giảm C. acnes phụ thuộc bã nhờn, điều chỉnh sừng hóa và có tác dụng chống viêm.",
        "chunks": ["f6b3f02e-cba0-58d8-bc6b-38ba398679e3"],
    },
    "isotretinoin_monitoring": {
        "text": "Isotretinoin cần tư vấn, đánh giá và theo dõi các nguy cơ tâm thần theo hướng dẫn MHRA.",
        "chunks": ["26c4c35a-1421-5b5e-a331-f08bee903d74"],
    },
    "isotretinoin_pregnancy_prevention": {
        "text": "Người có khả năng mang thai dùng isotretinoin cần tránh thai và tuân thủ chương trình phòng ngừa thai của MHRA.",
        "chunks": ["93c0a3e3-816e-5930-871c-5e772b08dd4d"],
    },
    "spironolactone_role": {
        "text": "Spironolactone có thể là một lựa chọn điều trị mụn hiệu quả, đặc biệt ở phụ nữ.",
        "chunks": ["baa3e271-8367-5aba-a547-9149c0a41fc0"],
    },
    "spironolactone_mechanism": {
        "text": "Spironolactone đối kháng aldosterone, giảm testosterone và ức chế androgen gắn lên thụ thể.",
        "chunks": ["41e041af-9428-54cb-a20e-d29617f316b5"],
    },
    "spironolactone_pregnancy_precaution": {
        "text": "Người có khả năng mang thai cần dùng biện pháp tránh thai khi điều trị mụn bằng spironolactone.",
        "chunks": ["4a825443-b33e-5e7b-a3d0-c40a31615b0f"],
    },
    "coc_mechanism": {
        "text": "Thuốc tránh thai phối hợp hỗ trợ mụn thông qua tác dụng kháng androgen.",
        "chunks": ["ccc72a29-3f9a-574c-aa41-b3ad050a673f"],
    },
    "coc_role": {
        "text": "Thuốc tránh thai phối hợp có thể dùng cùng thuốc trị mụn bôi hoặc uống khác.",
        "chunks": ["56e503dd-292a-51c5-a569-97c223c6c095"],
    },
    "coc_precaution": {
        "text": "Thuốc tránh thai phối hợp chứa ethinyl estradiol có liên quan đến nguy cơ huyết khối tĩnh mạch.",
        "chunks": ["aeeeeaa2-a2ba-5d0c-80b5-1e6384e5a96f"],
    },
    "salicylic_role": {
        "text": "Salicylic acid là hoạt chất tiêu nhân mụn bôi không kê đơn và giúp thông thoáng lỗ chân lông.",
        "chunks": ["7bbeb95e-489e-5583-ae76-b128853168e6", "6d6de22f-f537-5336-9812-1133ba9fbc7b"],
    },
    "salicylic_limits": {
        "text": "Salicylic acid không làm giảm bã nhờn và không diệt vi khuẩn.",
        "chunks": ["6d6de22f-f537-5336-9812-1133ba9fbc7b"],
    },
    "salicylic_unclog": {
        "text": "Salicylic acid giúp làm thông thoáng lỗ chân lông.",
        "chunks": ["6d6de22f-f537-5336-9812-1133ba9fbc7b"],
    },
    "salicylic_comedolytic_unclog": {
        "text": "Salicylic acid là hoạt chất tiêu nhân mụn bôi và giúp làm thông thoáng lỗ chân lông.",
        "chunks": ["7bbeb95e-489e-5583-ae76-b128853168e6", "6d6de22f-f537-5336-9812-1133ba9fbc7b"],
    },
    "salicylic_no_bacteria": {
        "text": "Salicylic acid không diệt vi khuẩn.",
        "chunks": ["6d6de22f-f537-5336-9812-1133ba9fbc7b"],
    },
    "azelaic_option": {
        "text": "Azelaic acid bôi nồng độ 15% hoặc 20% là một lựa chọn điều trị được nêu trong hướng dẫn NICE.",
        "chunks": ["5a192e07-bed0-5192-8156-6f45ad41f944"],
    },
    "clascoterone_role": {
        "text": "Clascoterone là thuốc bôi chẹn androgen dùng trong điều trị mụn.",
        "chunks": ["8f7614fe-0375-5441-9822-3773a0cc4e07"],
    },
    "clascoterone_precaution": {
        "text": "Nguồn Bộ Y tế nêu clascoterone dùng điều trị mụn từ 12 tuổi trở lên.",
        "chunks": ["8f7614fe-0375-5441-9822-3773a0cc4e07"],
    },
    "clascoterone_irritation": {
        "text": "Clascoterone bôi có thể gây kích ứng da.",
        "chunks": ["8f7614fe-0375-5441-9822-3773a0cc4e07"],
    },
    "topical_irritation_start": {
        "text": "Để giảm kích ứng từ benzoyl peroxide hoặc retinoid bôi, có thể bắt đầu cách ngày hoặc tiếp xúc ngắn rồi tăng dần nếu dung nạp.",
        "chunks": ["a9d1f7ea-e047-506d-b951-1f80bf38f31a"],
    },
    "bp_irritation_start": {
        "text": "Để giảm kích ứng từ benzoyl peroxide, có thể bắt đầu cách ngày hoặc tiếp xúc ngắn rồi tăng dần nếu dung nạp.",
        "chunks": ["a9d1f7ea-e047-506d-b951-1f80bf38f31a"],
    },
    "pregnancy_contraindication": {
        "text": "Retinoid bôi và tetracycline uống chống chỉ định trong thai kỳ và khi dự định mang thai.",
        "chunks": ["a9d1f7ea-e047-506d-b951-1f80bf38f31a"],
    },
    "pregnancy_limited_topicals": {
        "text": "Nguồn AAD cho biết azelaic acid, benzoyl peroxide, erythromycin và clindamycin bôi không được kỳ vọng gây hại thai do hấp thu toàn thân hạn chế.",
        "chunks": ["4f1be5ba-64c7-5227-b2c8-6ecada57c10c"],
    },
    "azelaic_pregnancy_limited": {
        "text": "Nguồn AAD không kỳ vọng azelaic acid bôi gây hại thai do hấp thu toàn thân hạn chế.",
        "chunks": ["4f1be5ba-64c7-5227-b2c8-6ecada57c10c"],
    },
    "minocycline_precaution": {
        "text": "Minocycline có nguy cơ tác dụng phụ cao hơn doxycycline, gồm các biến cố hiếm như lupus, viêm gan và tăng sắc tố.",
        "chunks": ["501d81ae-61be-5cef-a75e-fec4b6110edd"],
    },
    "minocycline_comparison": {
        "text": "Lymecycline hoặc doxycycline có nguy cơ tác dụng phụ thấp hơn minocycline; minocycline có thể liên quan đến lupus ban đỏ, viêm gan và tăng sắc tố.",
        "chunks": ["501d81ae-61be-5cef-a75e-fec4b6110edd"],
    },
    "skin_care": {
        "text": "NICE khuyên dùng sản phẩm rửa không kiềm, pH trung tính hoặc hơi acid hai lần mỗi ngày trên vùng da dễ nổi mụn.",
        "chunks": ["ffb2b115-4fcd-5f55-adc4-6aa1206eaa95"],
    },
    "makeup_care": {
        "text": "Nên tránh mỹ phẩm nền dầu hoặc gây bít tắc và tẩy trang vào cuối ngày.",
        "chunks": ["ffb2b115-4fcd-5f55-adc4-6aa1206eaa95"],
    },
    "picking_risk": {
        "text": "Cạy, gãi hoặc nặn tổn thương mụn kéo dài làm tăng nguy cơ để lại sẹo.",
        "chunks": ["ffb2b115-4fcd-5f55-adc4-6aa1206eaa95"],
    },
    "diet_uncertain": {
        "text": "Bằng chứng về chế độ ăn tải đường huyết thấp trong điều trị mụn còn mâu thuẫn.",
        "chunks": ["92509d81-1e6f-5fa2-a250-46ea4daf49b4"],
    },
    "diet_insufficient_recommendation": {
        "text": "Bằng chứng hiện có chưa đủ để đưa ra khuyến nghị về chế độ ăn trong điều trị mụn.",
        "chunks": ["92509d81-1e6f-5fa2-a250-46ea4daf49b4"],
    },
    "scarring_risk": {
        "text": "Nguy cơ sẹo tăng theo mức độ nặng và thời gian kéo dài của mụn, dù bằng chứng về các yếu tố này còn có bất định.",
        "chunks": ["a9d1f7ea-e047-506d-b951-1f80bf38f31a", "3922b8bd-661e-5a4d-99c5-5487728abde6"],
    },
    "urgent_fulminans": {
        "text": "Mụn fulminans cần được chuyển khẩn trong ngày để đội da liễu bệnh viện đánh giá trong vòng 24 giờ.",
        "chunks": ["2326dc1b-cf4b-520b-8798-f9617fafc0b0"],
    },
    "refer_severe_forms": {
        "text": "Nên chuyển chuyên khoa khi có mụn conglobata hoặc mụn nốt nang.",
        "chunks": ["2326dc1b-cf4b-520b-8798-f9617fafc0b0"],
    },
    "refer_diagnostic_uncertainty": {
        "text": "Nên chuyển chuyên khoa khi không chắc chắn chẩn đoán mụn.",
        "chunks": ["2326dc1b-cf4b-520b-8798-f9617fafc0b0"],
    },
    "refer_failures": {
        "text": "Có thể cân nhắc chuyển chuyên khoa khi mụn không đáp ứng các liệu trình chuẩn phù hợp.",
        "chunks": ["2326dc1b-cf4b-520b-8798-f9617fafc0b0"],
    },
    "refer_scarring": {
        "text": "Có thể cân nhắc chuyển chuyên khoa khi mụn gây sẹo hoặc thay đổi sắc tố kéo dài.",
        "chunks": ["2326dc1b-cf4b-520b-8798-f9617fafc0b0"],
    },
    "refer_persistent_severe_scar": {
        "text": "Sẹo mụn nặng còn tồn tại một năm sau khi hết mụn nên được chuyển đội da liễu có chuyên môn về sẹo.",
        "chunks": ["e65855e1-b148-585f-8984-a9383edfdd24"],
    },
    "refer_psychological": {
        "text": "Mụn gây đau khổ tâm lý đáng kể hoặc lo ngại sức khỏe tâm thần nghiêm trọng là lý do cần cân nhắc hỗ trợ hoặc chuyển dịch vụ phù hợp.",
        "chunks": ["2326dc1b-cf4b-520b-8798-f9617fafc0b0", "f15b032f-02c6-51d8-8147-589bac07ad8f"],
    },
}


# These are short verbatim spans from the referenced frozen chunks. They are
# intentionally separate from the Vietnamese gold-claim paraphrases so a
# researcher can inspect source support without treating the annotation as a
# human approval. Formatting differences are validated after normalization.
CLAIM_EVIDENCE: dict[str, list[tuple[str, str]]] = {
    "acne_definition": [("904d057a-aa4d-5d6a-bcb7-7379c5137b60", "Acne vulgaris is a chronic, inflammatory skin disease of the pilosebaceous unit.")],
    "acne_lesions": [("904d057a-aa4d-5d6a-bcb7-7379c5137b60", "Acne primarily presents with open or closed comedones, papules, pustules, or nodules on the face or trunk")],
    "acne_lesion_types": [("904d057a-aa4d-5d6a-bcb7-7379c5137b60", "open or closed comedones, papules, pustules, or nodules")],
    "pathogenesis": [("e35a46c1-1733-5225-b02c-9fb0c198d26d", "The multifactorial pathogenesis of acne involves: - Follicular hyperkeratinization - Microbial colonization with Cutibacterium acnes - Sebum production - Complex inflammatory mechanisms")],
    "c_acnes_identity": [("056b77fc-8b05-5da9-a46a-7cc57fb32f42", "C. acnes (formerly Propionibacterium acnes) is a Gram-positive anaerobic rod")],
    "mild_moderate": [("3a600fda-b952-5feb-bd28-f757f9539a60", "For mild to moderate acne, this includes people who have 1 or more of: any number of non-inflammatory lesions (comedones) up to 34 inflammatory lesions (with or without non-inflammatory lesions) up to 2 nodules.")],
    "moderate_severe": [("8db0c5ee-c2ec-59f2-96ff-9ee31ed59e0e", "For moderate to severe acne this includes people who have either or both of: 35 or more inflammatory lesions (with or without non-inflammatory lesions) 3 or more nodules.")],
    "comedone_types": [("ac93995d-741e-53f0-8b8c-72252cf81119", "Open comedones are blackheads; black because of surface pigment (melanin), rather than dirt Closed comedones are whiteheads; the follicle is completely blocked")],
    "comedone_names": [
        ("ac93995d-741e-53f0-8b8c-72252cf81119", "Open comedones are blackheads"),
        ("ac93995d-741e-53f0-8b8c-72252cf81119", "Closed comedones are whiteheads"),
    ],
    "blackhead_color": [("ac93995d-741e-53f0-8b8c-72252cf81119", "Open comedones are blackheads; black because of surface pigment (melanin), rather than dirt")],
    "scar_types": [("c5a99f3c-abf4-5ed3-be4a-6acdf0b82d3a", "Atrophic scars can be classified by three main types: ice-pick scars, rolling scars, and box-car scars.")],
    "postinflammatory_pigment": [("df9c7438-a085-5063-9824-a96830e2224a", "Postinflammatory hyperpigmentation (more common in darker skin types) occurs due to the deposition of melanin within the keratinocytes of skin that had recent inflammatory damage.")],
    "topical_mainstay": [("f675512f-93f5-572a-b705-35132a1445b7", "Topical therapies are the mainstay of acne treatment: they may be used for acne initial treatment and maintenance as monotherapy (except topical antibiotics) or used in combination with other topical or oral agents.")],
    "topical_multimodal": [("f675512f-93f5-572a-b705-35132a1445b7", "multimodal therapy combining multiple mechanisms of actions is recommended as a good practice statement to optimize efficacy and to reduce the risk of antibiotic resistance")],
    "bp_type_mechanism": [("db4632c5-44ed-5dea-80c2-addb0a2534b3", "Type: Over-the-counter topical antimicrobial agent.")],
    "bp_antimicrobial_role": [("db4632c5-44ed-5dea-80c2-addb0a2534b3", "topical antimicrobial agent")],
    "bp_oxygen_mechanism": [("db4632c5-44ed-5dea-80c2-addb0a2534b3", "Mechanism: Releases free oxygen radicals")],
    "bp_comedolytic": [("db4632c5-44ed-5dea-80c2-addb0a2534b3", "is mildly comedolytic")],
    "bp_resistance": [("db4632c5-44ed-5dea-80c2-addb0a2534b3", "No resistance of C. acnes to BP has been reported.")],
    "bp_adverse": [("db4632c5-44ed-5dea-80c2-addb0a2534b3", "Can cause burning sensation, stinging, dryness, erythema, pain, peeling, irritation, fabric staining, and bleaching.")],
    "retinoid_role": [("850609c9-471a-54cc-934f-478ad48d826d", "Topical retinoids are vitamin A derivatives and serve as the cornerstone of acne treatment since they are comedolytic and anti-inflammatory, improve dyspigmentation, and enable maintenance of acne clearance.")],
    "retinoid_comedolytic_antiinflammatory": [("850609c9-471a-54cc-934f-478ad48d826d", "they are comedolytic and anti-inflammatory")],
    "avoid_antibiotic_mono": [("a9d1f7ea-e047-506d-b951-1f80bf38f31a", "Do not use the following to treat acne: monotherapy with a topical antibiotic monotherapy with an oral antibiotic")],
    "avoid_topical_antibiotic_mono": [("a9d1f7ea-e047-506d-b951-1f80bf38f31a", "monotherapy with a topical antibiotic")],
    "bp_with_antibiotic": [("e4e54d11-c8bf-56eb-8e41-b7be9ecea11e", "Concomitant use of BP can prevent the development of antibiotic resistance.")],
    "oral_antibiotic_course": [("47e2209f-efbe-5807-b7a9-f48df661923e", "Systemic antibiotic use should also be limited to the shortest duration possible, typically no more than 3-4 months")],
    "oral_antibiotic_combination": [("47e2209f-efbe-5807-b7a9-f48df661923e", "When treating acne with systemic antibiotics, we recommend concomitant use of benzoyl peroxide (BP) and other topical therapies")],
    "first_line_combinations": [
        ("9f2c542c-d365-51c5-be89-55c0ee80d22a", "Offer people with acne a 12-week course of 1 of the following first-line treatment options"),
        ("9f2c542c-d365-51c5-be89-55c0ee80d22a", "a fixed combination of topical adapalene with topical benzoyl peroxide for any acne severity")
    ],
    "combination_advantage": [("12259038-373a-5ead-a76c-6e536aef00b9", "Sản phẩm kết hợp các thuốc bôi như: Adapalene + Benzoyl Peroxide Clindamycin + Tretinoin Erythromycin + Isotretinoin Hiệu quả: Cao hơn so với điều trị đơn độc một thành phần.")],
    "maintenance": [("6f0ad48a-89fd-522b-8651-ad13e98e24c9", "maintenance treatment is not always necessary")],
    "maintenance_relapse": [("6f0ad48a-89fd-522b-8651-ad13e98e24c9", "Consider maintenance treatment in people with a history of frequent relapse after treatment")],
    "maintenance_adapalene_bp": [("6f0ad48a-89fd-522b-8651-ad13e98e24c9", "Consider a fixed combination of topical adapalene and topical benzoyl peroxide as maintenance treatment for acne")],
    "isotretinoin_indication": [
        ("26c4c35a-1421-5b5e-a331-f08bee903d74", "Consider oral isotretinoin for people older than 12 years who have a severe form of acne that is resistant to adequate courses of standard therapy with systemic antibiotics and topical therapy"),
        ("26c4c35a-1421-5b5e-a331-f08bee903d74", "nodulo-cystic acne acne conglobata acne fulminans acne at risk of permanent scarring")
    ],
    "isotretinoin_mechanism": [("f6b3f02e-cba0-58d8-bc6b-38ba398679e3", "Reducing the size and secretion of sebaceous glands. Decreasing surface and ductal levels of sebum-dependent C. acnes. Inhibiting comedogenesis by normalizing keratinocyte keratinization. Possessing anti-inflammatory properties.")],
    "isotretinoin_monitoring": [("26c4c35a-1421-5b5e-a331-f08bee903d74", "requirements for counselling people about potential mental health and sexual function side effects requirements for assessing and monitoring mental health and sexual function")],
    "isotretinoin_pregnancy_prevention": [("93c0a3e3-816e-5930-871c-5e772b08dd4d", "people of childbearing potential have to use contraception and need to follow the recommended MHRA pregnancy prevention programme")],
    "spironolactone_role": [("baa3e271-8367-5aba-a547-9149c0a41fc0", "Spironolactone may be an effective treatment option for acne, particularly in women")],
    "spironolactone_mechanism": [("41e041af-9428-54cb-a20e-d29617f316b5", "Spironolactone is an aldosterone receptor antagonist that decreases testosterone production and competitively inhibits testosterone and dihydrotestosterone binding to androgen receptors in the skin.")],
    "spironolactone_pregnancy_precaution": [("4a825443-b33e-5e7b-a3d0-c40a31615b0f", "If you can get pregnant, you’ll need to use birth control while taking spironolactone.")],
    "coc_mechanism": [("ccc72a29-3f9a-574c-aa41-b3ad050a673f", "COCs treat acne through their overall anti-androgenic properties")],
    "coc_role": [("56e503dd-292a-51c5-a569-97c223c6c095", "COCs may be combined with other oral or topical acne medications")],
    "coc_precaution": [("aeeeeaa2-a2ba-5d0c-80b5-1e6384e5a96f", "Typical use of Ethinyl Estradiol (EE) with daily doses below 50 µg is associated with lower venous thromboembolism (VTE) risks compared to historical use of EE with daily doses above 50 µg.")],
    "salicylic_role": [
        ("7bbeb95e-489e-5583-ae76-b128853168e6", "Salicylic acid is a topical comedolytic agent available over the counter"),
        ("6d6de22f-f537-5336-9812-1133ba9fbc7b", "salicylic acid helps unclog pores to resolve and prevent lesions")
    ],
    "salicylic_limits": [("6d6de22f-f537-5336-9812-1133ba9fbc7b", "It does not have any effect on sebum production and does not kill bacteria.")],
    "salicylic_unclog": [("6d6de22f-f537-5336-9812-1133ba9fbc7b", "salicylic acid helps unclog pores to resolve and prevent lesions")],
    "salicylic_comedolytic_unclog": [
        ("7bbeb95e-489e-5583-ae76-b128853168e6", "Salicylic acid is a topical comedolytic agent"),
        ("6d6de22f-f537-5336-9812-1133ba9fbc7b", "salicylic acid helps unclog pores to resolve and prevent lesions"),
    ],
    "salicylic_no_bacteria": [("6d6de22f-f537-5336-9812-1133ba9fbc7b", "does not kill bacteria")],
    "azelaic_option": [
        ("5a192e07-bed0-5192-8156-6f45ad41f944", "Formulation with either of these 2 concentrations: 15% azelaic acid"),
        ("5a192e07-bed0-5192-8156-6f45ad41f944", "20% azelaic acid")
    ],
    "azelaic_pregnancy_limited": [("4f1be5ba-64c7-5227-b2c8-6ecada57c10c", "the risk of fetal harm from topical azelaic acid, BP, erythromycin, and clindamycin is not expected based on limited systemic absorption")],
    "clascoterone_role": [("8f7614fe-0375-5441-9822-3773a0cc4e07", "Loại thuốc: Thuốc kháng androgen tại chỗ.")],
    "clascoterone_precaution": [("8f7614fe-0375-5441-9822-3773a0cc4e07", "Đã được FDA phê duyệt để điều trị mụn trứng cá ở người từ 12 tuổi trở lên")],
    "clascoterone_irritation": [("8f7614fe-0375-5441-9822-3773a0cc4e07", "Tác dụng phụ: Có thể kích ứng da")],
    "topical_irritation_start": [
        ("a9d1f7ea-e047-506d-b951-1f80bf38f31a", "To reduce the risk of skin irritation associated with topical treatments, such as benzoyl peroxide or retinoids, start with alternate-day or short-contact application"),
        ("a9d1f7ea-e047-506d-b951-1f80bf38f31a", "If tolerated, progress to using a standard application")
    ],
    "bp_irritation_start": [
        ("a9d1f7ea-e047-506d-b951-1f80bf38f31a", "To reduce the risk of skin irritation associated with topical treatments, such as benzoyl peroxide or retinoids, start with alternate-day or short-contact application"),
        ("a9d1f7ea-e047-506d-b951-1f80bf38f31a", "If tolerated, progress to using a standard application")
    ],
    "pregnancy_contraindication": [("a9d1f7ea-e047-506d-b951-1f80bf38f31a", "topical retinoids and oral tetracyclines are contraindicated during pregnancy and when planning a pregnancy")],
    "pregnancy_limited_topicals": [("4f1be5ba-64c7-5227-b2c8-6ecada57c10c", "the risk of fetal harm from topical azelaic acid, BP, erythromycin, and clindamycin is not expected based on limited systemic absorption")],
    "minocycline_precaution": [("501d81ae-61be-5cef-a75e-fec4b6110edd", "Lymecycline or doxycycline have a lower risk of side effects than minocycline (which may, for example, be associated with lupus erythematosus, hepatitis and pigmentation)")],
    "minocycline_comparison": [("501d81ae-61be-5cef-a75e-fec4b6110edd", "Lymecycline or doxycycline have a lower risk of side effects than minocycline (which may, for example, be associated with lupus erythematosus, hepatitis and pigmentation)")],
    "skin_care": [("ffb2b115-4fcd-5f55-adc4-6aa1206eaa95", "use a non-alkaline (skin pH neutral or slightly acidic) synthetic detergent (syndet) cleansing product twice daily on acne-prone skin")],
    "makeup_care": [("ffb2b115-4fcd-5f55-adc4-6aa1206eaa95", "Advise people with acne who use make-up to avoid oil-based and comedogenic products, and to remove make-up at the end of the day.")],
    "picking_risk": [("ffb2b115-4fcd-5f55-adc4-6aa1206eaa95", "persistent picking or scratching of acne lesions can increase the risk of scarring")],
    "diet_uncertain": [("92509d81-1e6f-5fa2-a250-46ea4daf49b4", "Available evidence is conflicting on low-glycemic-load diet for acne treatment.")],
    "diet_insufficient_recommendation": [("92509d81-1e6f-5fa2-a250-46ea4daf49b4", "Available evidence is insufficient to develop a recommendation")],
    "refer_severe_forms": [("2326dc1b-cf4b-520b-8798-f9617fafc0b0", "they have acne conglobata they have nodulo-cystic acne")],
    "refer_diagnostic_uncertainty": [("2326dc1b-cf4b-520b-8798-f9617fafc0b0", "there is diagnostic uncertainty about their acne")],
    "urgent_fulminans": [("2326dc1b-cf4b-520b-8798-f9617fafc0b0", "Urgently refer people with acne fulminans on the same day to the on-call hospital dermatology team, to be assessed within 24 hours.")],
    "refer_failures": [
        ("2326dc1b-cf4b-520b-8798-f9617fafc0b0", "mild to moderate acne that has not responded to 2 completed courses of treatment"),
        ("2326dc1b-cf4b-520b-8798-f9617fafc0b0", "moderate to severe acne which has not responded to previous treatment that contains an oral antibiotic")
    ],
    "refer_scarring": [("2326dc1b-cf4b-520b-8798-f9617fafc0b0", "acne that is leading to scarring acne with persistent pigmentary changes")],
    "refer_persistent_severe_scar": [("e65855e1-b148-585f-8984-a9383edfdd24", "If a person's acne-related scarring is severe and persists a year after their acne has cleared: refer the person to a consultant dermatologist-led team with expertise in scarring management")],
    "refer_psychological": [("2326dc1b-cf4b-520b-8798-f9617fafc0b0", "Consider referral to mental health services if a person with acne experiences significant psychological distress or a mental health disorder")],
}


def _single_specs() -> list[tuple[str, str, str, list[str]]]:
    return [
        ("ANS-DEF-001", "definition_classification", "Mụn trứng cá được định nghĩa là bệnh gì và liên quan cấu trúc da nào?", ["acne_definition"]),
        ("ANS-DEF-002", "definition_classification", "Những dạng tổn thương cơ bản nào có thể xuất hiện trong mụn trứng cá?", ["acne_lesion_types"]),
        ("ANS-DEF-003", "definition_classification", "Nhân mụn mở và nhân mụn đóng được gọi thông thường là gì?", ["comedone_names"]),
        ("ANS-DEF-004", "definition_classification", "Vì sao đầu của mụn đầu đen có màu sẫm, có phải do da bẩn không?", ["blackhead_color"]),
        ("ANS-DEF-005", "definition_classification", "NICE phân loại mụn nhẹ đến vừa dựa trên số tổn thương viêm và nốt như thế nào?", ["mild_moderate"]),
        ("ANS-DEF-006", "definition_classification", "Khi nào số tổn thương được xếp vào mức mụn vừa đến nặng theo NICE?", ["moderate_severe"]),
        ("ANS-DEF-007", "definition_classification", "Các yếu tố chính tham gia sinh bệnh học mụn trứng cá gồm những gì?", ["pathogenesis"]),
        ("ANS-DEF-008", "definition_classification", "Vi khuẩn C. acnes từng được gọi bằng tên nào và có đặc điểm vi sinh gì?", ["c_acnes_identity"]),
        ("ANS-DEF-009", "definition_classification", "Ba kiểu sẹo teo thường gặp sau mụn được phân loại ra sao?", ["scar_types"]),
        ("ANS-DEF-010", "definition_classification", "Tăng sắc tố sau viêm sau mụn hình thành do cơ chế nào?", ["postinflammatory_pigment"]),

        ("ANS-TRT-001", "treatment_use_combination", "Vai trò chung của thuốc bôi trong điều trị mụn là gì?", ["topical_mainstay", "topical_multimodal"]),
        ("ANS-TRT-002", "treatment_use_combination", "Phối hợp adapalene với benzoyl peroxide được NICE đặt ở vị trí nào trong điều trị ban đầu?", ["first_line_combinations"]),
        ("ANS-TRT-003", "treatment_use_combination", "Vì sao kháng sinh bôi không nên dùng một mình và benzoyl peroxide hỗ trợ thế nào?", ["avoid_topical_antibiotic_mono", "bp_with_antibiotic"]),
        ("ANS-TRT-004", "treatment_use_combination", "Nguyên tắc về thời gian và phối hợp khi dùng kháng sinh uống trị mụn là gì?", ["oral_antibiotic_course", "oral_antibiotic_combination"]),
        ("ANS-TRT-005", "treatment_use_combination", "Sau khi mụn đã ổn, ai có thể cần điều trị duy trì và lựa chọn nào được cân nhắc?", ["maintenance", "maintenance_relapse", "maintenance_adapalene_bp"]),
        ("ANS-TRT-006", "treatment_use_combination", "Trong những tình huống nào hướng dẫn cân nhắc isotretinoin uống?", ["isotretinoin_indication"]),
        ("ANS-TRT-007", "treatment_use_combination", "Spironolactone có thể phù hợp với nhóm người bệnh mụn nào?", ["spironolactone_role"]),
        ("ANS-TRT-008", "treatment_use_combination", "Thuốc tránh thai phối hợp có vai trò gì trong điều trị mụn và có thể kết hợp với gì?", ["coc_mechanism", "coc_role"]),
        ("ANS-TRT-009", "treatment_use_combination", "Salicylic acid bôi hỗ trợ mụn theo cách nào và giới hạn tác dụng của nó là gì?", ["salicylic_comedolytic_unclog", "salicylic_limits"]),
        ("ANS-TRT-010", "treatment_use_combination", "Hướng dẫn NICE nêu những nồng độ azelaic acid bôi nào?", ["azelaic_option"]),
        ("ANS-TRT-011", "treatment_use_combination", "Một routine làm sạch cơ bản cho da mụn nên tuân theo nguyên tắc nào?", ["skin_care"]),
        ("ANS-TRT-012", "treatment_use_combination", "Có thể chỉ dựa vào chế độ ăn tải đường huyết thấp để điều trị mụn không?", ["diet_uncertain", "diet_insufficient_recommendation"]),

        ("ANS-MEC-001", "mechanism_role", "Benzoyl peroxide có vai trò kháng khuẩn và tác động lên mụn bằng những cơ chế chính nào?", ["bp_antimicrobial_role", "bp_oxygen_mechanism", "bp_comedolytic"]),
        ("ANS-MEC-002", "mechanism_role", "Retinoid bôi góp phần kiểm soát nhân mụn và viêm ra sao?", ["retinoid_comedolytic_antiinflammatory"]),
        ("ANS-MEC-003", "mechanism_role", "Isotretinoin ảnh hưởng đến tuyến bã và quá trình hình thành mụn như thế nào?", ["isotretinoin_mechanism"]),
        ("ANS-MEC-004", "mechanism_role", "Spironolactone tác động lên con đường androgen bằng cách nào?", ["spironolactone_mechanism"]),
        ("ANS-MEC-005", "mechanism_role", "Tác dụng kháng androgen giải thích vai trò của thuốc tránh thai phối hợp trong mụn thế nào?", ["coc_mechanism"]),
        ("ANS-MEC-006", "mechanism_role", "Salicylic acid làm thông thoáng lỗ chân lông nhưng không làm được những việc gì?", ["salicylic_unclog", "salicylic_limits"]),
        ("ANS-MEC-007", "mechanism_role", "Clascoterone thuộc nhóm tác động nội tiết tại chỗ nào và được dùng cho độ tuổi nào theo nguồn Bộ Y tế?", ["clascoterone_role", "clascoterone_precaution"]),
        ("ANS-MEC-008", "mechanism_role", "Vì sao benzoyl peroxide thường được đưa vào phác đồ có kháng sinh bôi?", ["bp_with_antibiotic", "bp_resistance"]),

        ("ANS-ADV-001", "adverse_effects_precautions", "Những kích ứng và ảnh hưởng lên đồ vải nào có thể gặp với benzoyl peroxide?", ["bp_adverse"]),
        ("ANS-ADV-002", "adverse_effects_precautions", "Có thể bắt đầu benzoyl peroxide hoặc retinoid bôi thế nào để giảm kích ứng?", ["topical_irritation_start"]),
        ("ANS-ADV-003", "adverse_effects_precautions", "Những nhóm thuốc trị mụn nào được nêu là chống chỉ định khi mang thai?", ["pregnancy_contraindication"]),
        ("ANS-ADV-004", "adverse_effects_precautions", "Các thuốc bôi nào được nguồn AAD xem là ít khả năng gây hại thai do hấp thu hạn chế?", ["pregnancy_limited_topicals"]),
        ("ANS-ADV-005", "adverse_effects_precautions", "Khi cân nhắc isotretinoin cần tư vấn và theo dõi những nhóm nguy cơ nào?", ["isotretinoin_monitoring", "isotretinoin_pregnancy_prevention"]),
        ("ANS-ADV-006", "adverse_effects_precautions", "Minocycline có những lưu ý tác dụng phụ nào so với doxycycline?", ["minocycline_precaution"]),
        ("ANS-ADV-007", "adverse_effects_precautions", "Nguy cơ quan trọng nào cần cân nhắc khi dùng thuốc tránh thai phối hợp cho mụn?", ["coc_precaution"]),
        ("ANS-ADV-008", "adverse_effects_precautions", "Clascoterone bôi có lưu ý về độ tuổi và kích ứng ra sao?", ["clascoterone_precaution", "clascoterone_irritation"]),
        ("ANS-ADV-009", "adverse_effects_precautions", "Thói quen cạy hoặc gãi tổn thương làm thay đổi nguy cơ sẹo thế nào?", ["picking_risk"]),

        ("ANS-CMP-001", "comparison_integration", "Đối chiếu nhân mụn mở với nhân mụn đóng về tên gọi và hình thái bít tắc.", ["comedone_types"]),
        ("ANS-CMP-002", "comparison_integration", "Hai ngưỡng NICE cho mụn nhẹ-vừa và vừa-nặng khác nhau ở số tổn thương nào?", ["mild_moderate", "moderate_severe"]),
        ("ANS-CMP-003", "comparison_integration", "Benzoyl peroxide có vai trò kháng khuẩn gì, C. acnes có được ghi nhận kháng hoạt chất này không, và vì sao nó được phối hợp với kháng sinh bôi?", ["bp_antimicrobial_role", "bp_resistance", "bp_with_antibiotic"]),
        ("ANS-CMP-004", "comparison_integration", "Retinoid bôi và benzoyl peroxide bổ sung nhau ở những vai trò nào?", ["retinoid_comedolytic_antiinflammatory", "bp_antimicrobial_role", "bp_comedolytic"]),
        ("ANS-CMP-005", "comparison_integration", "Kháng sinh bôi và kháng sinh uống giống nhau ở nguyên tắc tránh đơn trị liệu, và kháng sinh uống được khuyến nghị dùng trong thời gian bao lâu?", ["avoid_antibiotic_mono", "oral_antibiotic_course"]),
        ("ANS-CMP-006", "comparison_integration", "Phác đồ phối hợp nhiều hoạt chất có lợi thế gì so với một thành phần đơn lẻ?", ["combination_advantage"]),
        ("ANS-CMP-007", "comparison_integration", "Vì sao doxycycline hoặc lymecycline thường được ưu tiên hơn minocycline trong hướng dẫn NICE?", ["minocycline_comparison"]),
        ("ANS-CMP-008", "comparison_integration", "Isotretinoin khác điều trị chuẩn ban đầu ở chỉ định và yêu cầu giám sát thế nào?", ["isotretinoin_indication", "isotretinoin_monitoring", "isotretinoin_pregnancy_prevention"]),
        ("ANS-CMP-009", "comparison_integration", "Sản phẩm rửa và mỹ phẩm cho da mụn nên chọn khác sản phẩm nền dầu, gây bít tắc như thế nào?", ["skin_care", "makeup_care"]),
        ("ANS-CMP-010", "comparison_integration", "Tăng sắc tố sau viêm hình thành do gì, còn sẹo teo sau mụn thường được chia thành những dạng nào?", ["postinflammatory_pigment", "scar_types"]),

        ("ANS-REF-001", "referral_care_seeking", "Trường hợp mụn fulminans cần được chuyển khám nhanh đến mức nào?", ["urgent_fulminans"]),
        ("ANS-REF-002", "referral_care_seeking", "Những thể mụn nặng nào là lý do chuyển đội chuyên khoa da liễu?", ["refer_severe_forms", "urgent_fulminans"]),
        ("ANS-REF-003", "referral_care_seeking", "Nếu chưa chắc tổn thương có thật sự là mụn, hướng dẫn xử trí tuyến chuyên môn ra sao?", ["refer_diagnostic_uncertainty"]),
        ("ANS-REF-004", "referral_care_seeking", "Khi các liệu trình chuẩn phù hợp đều thất bại, có nên cân nhắc chuyển chuyên khoa không?", ["refer_failures"]),
        ("ANS-REF-005", "referral_care_seeking", "Sẹo hoặc thay đổi sắc tố kéo dài liên quan đến quyết định chuyển chuyên khoa như thế nào?", ["refer_scarring"]),
        ("ANS-REF-006", "referral_care_seeking", "Ảnh hưởng tâm lý do mụn khi nào cần được xem là lý do tìm hỗ trợ chuyên môn?", ["refer_psychological"]),
    ]


def _multi_specs() -> list[dict[str, Any]]:
    return [
        _multi("ANS-MUL-001", "pronoun_coreference", "Thuốc bôi đó tác động lên androgen bằng cách nào?", ["clascoterone_role"], [("user", "Tôi muốn hỏi về thuốc bôi clascoterone."), ("assistant", "Được, bạn muốn biết thêm điều gì về thuốc đó?")]),
        _multi("ANS-MUL-002", "pronoun_coreference", "Hoạt chất này làm thông thoáng lỗ chân lông nhưng không diệt vi khuẩn đúng không?", ["salicylic_unclog", "salicylic_no_bacteria"], [("user", "Salicylic acid có vai trò gì trong trị mụn?"), ("assistant", "Đó là một hoạt chất tiêu nhân mụn dùng tại chỗ.")]),
        _multi("ANS-MUL-003", "pronoun_coreference", "Thuốc đó ảnh hưởng tuyến bã bằng những cơ chế nào và được cân nhắc cho mức độ nào?", ["isotretinoin_mechanism", "isotretinoin_indication"], [("user", "Isotretinoin thường được cân nhắc trong trường hợp nào?"), ("assistant", "Thuốc được cân nhắc cho một số trường hợp mụn nặng kháng điều trị chuẩn.")]),
        _multi("ANS-MUL-004", "pronoun_coreference", "Liệu pháp này phù hợp với nhóm nào và cần lưu ý gì nếu có thể mang thai?", ["spironolactone_role", "spironolactone_pregnancy_precaution"], [("user", "Spironolactone tác động lên androgen ra sao?"), ("assistant", "Thuốc đối kháng aldosterone và làm giảm tác động androgen.")]),
        _multi("ANS-MUL-005", "pronoun_coreference", "Lựa chọn bôi đó có những nồng độ nào và nguồn thai kỳ nhận định ra sao?", ["azelaic_option", "azelaic_pregnancy_limited"], [("user", "Có thể cân nhắc azelaic acid cho mụn không?"), ("assistant", "Azelaic acid bôi là một lựa chọn được nêu trong hướng dẫn.")]),
        _multi("ANS-MUL-006", "follow_up_continuity", "Nếu da dễ kích ứng thì nên bắt đầu nhóm này ra sao?", ["topical_irritation_start"], [("user", "Retinoid bôi có vai trò gì trong trị mụn?"), ("assistant", "Retinoid bôi giúp tiêu nhân mụn và chống viêm.")]),
        _multi("ANS-MUL-007", "follow_up_continuity", "Còn thời gian dùng và yêu cầu phối hợp của thuốc uống này thì sao?", ["oral_antibiotic_course", "oral_antibiotic_combination"], [("user", "Kháng sinh uống có thể dùng cho mụn viêm không?"), ("assistant", "Có thể được dùng trong phác đồ phù hợp, nhưng không nên là đơn trị liệu.")]),
        _multi("ANS-MUL-008", "follow_up_continuity", "Với mỹ phẩm trang điểm thì áp dụng nguyên tắc chăm sóc nào?", ["makeup_care"], [("user", "Da mụn nên chọn sữa rửa mặt như thế nào?"), ("assistant", "Nên dùng sản phẩm rửa không kiềm, pH trung tính hoặc hơi acid.")]),
        _multi("ANS-MUL-009", "follow_up_continuity", "Nếu dấu này tồn tại lâu sau khi hết mụn thì lúc nào cần chuyên khoa?", ["refer_persistent_severe_scar"], [("user", "Sẹo teo sau mụn có những dạng nào?"), ("assistant", "Ba dạng thường nêu là đáy nhọn, lượn sóng và đáy vuông.")]),
        _multi("ANS-MUL-010", "explicit_topic_switch", "Còn isotretinoin thì vì sao phải quản lý nguy cơ thai kỳ và tâm thần?", ["isotretinoin_monitoring", "isotretinoin_pregnancy_prevention"], [("user", "Chế độ ăn tải đường huyết thấp có chắc chắn trị hết mụn không?"), ("assistant", "Bằng chứng còn mâu thuẫn và chưa đủ để đưa ra khuyến nghị.")]),
        _multi("ANS-MUL-011", "explicit_topic_switch", "Chuyển sang chuyện đi khám: thể nào cần chuyển ngay và sẹo kéo dài liên quan chuyên khoa ra sao?", ["urgent_fulminans", "refer_persistent_severe_scar"], [("user", "Nhân mụn mở khác nhân mụn đóng thế nào?"), ("assistant", "Nhân mở là đầu đen, nhân đóng là đầu trắng.")]),
        _multi("ANS-MUL-012", "explicit_topic_switch", "Riêng routine hằng ngày, nên rửa mặt thế nào và nếu dễ kích ứng thì bắt đầu BP ra sao?", ["skin_care", "bp_irritation_start"], [("user", "Benzoyl peroxide tác động lên C. acnes như thế nào?"), ("assistant", "Đây là hoạt chất kháng khuẩn bôi giải phóng gốc oxy tự do.")]),
        _multi("ANS-MUL-013", "repeated_question_history_isolation", "Nhắc lại giúp tôi: vì sao không nên cạy hoặc gãi các nốt mụn?", ["picking_risk"], [("user", "Cạy mụn thường xuyên có hại gì?"), ("assistant", "Việc này có thể làm tăng nguy cơ để lại sẹo."), ("user", "Tôi hiểu rồi.")]),
        _multi("ANS-MUL-014", "repeated_question_history_isolation", "Tóm lại sau khi khỏi mụn có bắt buộc phải dùng thuốc duy trì mãi không?", ["maintenance"], [("user", "Điều trị duy trì có luôn cần sau khi mụn đã sạch không?"), ("assistant", "Không phải ai cũng cần; thường cân nhắc khi hay tái phát."), ("user", "Bạn nói rõ kết luận một lần nữa nhé.")]),
        _multi("ANS-MUL-015", "repeated_question_history_isolation", "Trả lời lại thật gọn: màu đen đó có phải là bụi bẩn không?", ["blackhead_color"], [("user", "Vì sao nhân mụn mở có đầu màu đen?"), ("assistant", "Màu đen liên quan đến melanin, không phải do bụi bẩn."), ("user", "Tôi muốn xác nhận lại.")]),
    ]


def _multi(case_id: str, category: str, query: str, claims: list[str], history: list[tuple[str, str]]) -> dict[str, Any]:
    return {"case_id": case_id, "family": "answerable_multi_turn", "category": category, "query": query, "claim_keys": claims, "history": [{"role": role, "content": content} for role, content in history]}


def _gap_specs() -> list[tuple[str, str, str, list[str], str]]:
    rows = [
        ("GAP-ABS-001", "unsupported_absolute_certainty_guarantee", "Benzoyl peroxide có bảo đảm chữa khỏi mụn vĩnh viễn cho mọi người không?", ["bảo đảm", "vĩnh viễn", "mọi người"], "bp_type_mechanism"),
        ("GAP-ABS-002", "unsupported_absolute_certainty_guarantee", "Dùng adapalene đúng cách có chắc chắn ngăn mụn tái phát suốt đời không?", ["chắc chắn", "suốt đời", "tái phát"], "retinoid_role"),
        ("GAP-ABS-003", "unsupported_absolute_certainty_guarantee", "Azelaic acid có cam kết không gây kích ứng ở bất kỳ loại da nào không?", ["cam kết", "không gây kích ứng", "bất kỳ"], "azelaic_option"),
        ("GAP-ABS-004", "unsupported_absolute_certainty_guarantee", "Isotretinoin có bảo đảm người bị mụn nặng sẽ không để lại sẹo không?", ["bảo đảm", "không để lại sẹo"], "isotretinoin_indication"),
        ("GAP-ABS-005", "unsupported_absolute_certainty_guarantee", "Salicylic acid có hiệu quả với toàn bộ người bị mụn đầu đen không?", ["toàn bộ", "hiệu quả", "mụn đầu đen"], "salicylic_role"),
        ("GAP-ABS-006", "unsupported_absolute_certainty_guarantee", "Chế độ ăn tải đường huyết thấp có chắc chắn chữa dứt điểm mụn mà không cần thuốc không?", ["chắc chắn", "chữa dứt điểm", "không cần thuốc"], "diet_uncertain"),
        ("GAP-ABS-007", "unsupported_absolute_certainty_guarantee", "Dưỡng ẩm không gây bít tắc có bảo đảm không bao giờ nổi mụn mới không?", ["bảo đảm", "không bao giờ", "mụn mới"], "skin_care"),
        ("GAP-ABS-008", "unsupported_absolute_certainty_guarantee", "Kem chống nắng không dầu có chắc chắn ngăn hoàn toàn mọi vết thâm sau mụn không?", ["chắc chắn", "ngăn hoàn toàn", "mọi vết thâm"], "skin_care"),
        ("GAP-ABS-009", "unsupported_absolute_certainty_guarantee", "Phối hợp clindamycin với benzoyl peroxide có bảo đảm không bao giờ xuất hiện kháng kháng sinh không?", ["bảo đảm", "không bao giờ", "kháng kháng sinh"], "bp_with_antibiotic"),
        ("GAP-ABS-010", "unsupported_absolute_certainty_guarantee", "Quang trị liệu có chắc chắn làm sạch mụn vĩnh viễn sau một liệu trình không?", ["chắc chắn", "vĩnh viễn", "một liệu trình"], "topical_mainstay"),

        ("GAP-EXA-001", "unsupported_exact_quantity_time", "Chính xác bao nhiêu ngày adapalene sẽ làm hết toàn bộ nhân mụn?", ["bao nhiêu ngày", "hết toàn bộ", "adapalene"], "retinoid_role"),
        ("GAP-EXA-002", "unsupported_exact_quantity_time", "Sau đúng bao nhiêu phút benzoyl peroxide bắt đầu tiêu diệt C. acnes?", ["bao nhiêu phút", "bắt đầu", "C. acnes"], "bp_type_mechanism"),
        ("GAP-EXA-003", "unsupported_exact_quantity_time", "Mỗi lần bôi azelaic acid cần chính xác bao nhiêu gam cho toàn mặt?", ["bao nhiêu gam", "mỗi lần", "toàn mặt"], "azelaic_option"),
        ("GAP-EXA-004", "unsupported_exact_quantity_time", "Mỗi lần rửa da mụn cần chính xác bao nhiêu mililit sữa rửa mặt?", ["bao nhiêu mililit", "sữa rửa mặt"], "skin_care"),
        ("GAP-EXA-005", "unsupported_exact_quantity_time", "Nước rửa mặt cho da mụn phải đúng bao nhiêu độ C?", ["bao nhiêu độ C", "nước rửa mặt"], "skin_care"),
        ("GAP-EXA-006", "unsupported_exact_quantity_time", "Da mụn cần chính xác bao nhiêu gam kem chống nắng cho mỗi lần bôi mặt?", ["bao nhiêu gam", "kem chống nắng", "mỗi lần"], "skin_care"),
        ("GAP-EXA-007", "unsupported_exact_quantity_time", "Benzoyl peroxide làm giảm chính xác bao nhiêu phần trăm số lượng C. acnes sau một ngày?", ["bao nhiêu phần trăm", "C. acnes", "một ngày"], "bp_type_mechanism"),
        ("GAP-EXA-008", "unsupported_exact_quantity_time", "Sau một tuần adapalene làm giảm chính xác bao nhiêu phần trăm bã nhờn?", ["một tuần", "bao nhiêu phần trăm", "bã nhờn"], "retinoid_role"),
        ("GAP-EXA-009", "unsupported_exact_quantity_time", "Ba buổi điều trị sẹo sẽ cải thiện chính xác bao nhiêu milimet độ sâu sẹo?", ["ba buổi", "bao nhiêu milimet", "độ sâu sẹo"], "scar_types"),
        ("GAP-EXA-010", "unsupported_exact_quantity_time", "Một nhân mụn đóng mất chính xác bao nhiêu giờ để chuyển thành sẩn viêm?", ["bao nhiêu giờ", "chuyển thành", "sẩn viêm"], "comedone_types"),

        ("GAP-REL-001", "unsupported_comparison_relationship_specificity", "Nhãn adapalene nào có hiệu quả cao nhất tuyệt đối trong tất cả sản phẩm đang bán?", ["nhãn", "cao nhất", "tất cả sản phẩm"], "retinoid_agents"),
        ("GAP-REL-002", "unsupported_comparison_relationship_specificity", "Benzoyl peroxide 2,5% vượt 5% chính xác bao nhiêu phần trăm về hiệu quả?", ["2,5%", "5%", "bao nhiêu phần trăm"], "bp_type_mechanism"),
        ("GAP-REL-003", "unsupported_comparison_relationship_specificity", "Azelaic acid hay salicylic acid đứng hạng nhất về giảm sẩn viêm, với chênh lệch chính xác bao nhiêu?", ["đứng hạng nhất", "chênh lệch", "sẩn viêm"], "salicylic_role"),
        ("GAP-REL-004", "unsupported_comparison_relationship_specificity", "Gel adapalene hấp thu qua da gấp chính xác bao nhiêu lần dạng kem?", ["hấp thu", "bao nhiêu lần", "dạng kem"], "retinoid_agents"),
        ("GAP-REL-005", "unsupported_comparison_relationship_specificity", "Ánh sáng xanh có vượt ánh sáng đỏ một tỷ lệ chính xác nào trong trị mụn không?", ["ánh sáng xanh", "ánh sáng đỏ", "tỷ lệ chính xác"], "topical_mainstay"),
        ("GAP-REL-006", "unsupported_comparison_relationship_specificity", "Thương hiệu kem dưỡng nào được chứng minh tương thích nhất riêng với Epiduo?", ["thương hiệu", "tương thích nhất", "Epiduo"], "first_line_combinations"),
        ("GAP-REL-007", "unsupported_comparison_relationship_specificity", "Biến thể gene cụ thể nào dự đoán chắc chắn đáp ứng với benzoyl peroxide?", ["biến thể gene", "dự đoán", "benzoyl peroxide"], "bp_type_mechanism"),
        ("GAP-REL-008", "unsupported_comparison_relationship_specificity", "Chủng hệ vi sinh nào dự đoán chính xác kháng clindamycin ở từng người?", ["chủng hệ vi sinh", "dự đoán", "clindamycin"], "bp_with_antibiotic"),
        ("GAP-REL-009", "unsupported_comparison_relationship_specificity", "Bao nhiêu phần trăm retinol tương đương chính xác adapalene 0,1% về hiệu lực?", ["retinol", "tương đương", "adapalene 0,1%"], "retinoid_agents"),
        ("GAP-REL-010", "unsupported_comparison_relationship_specificity", "Loại chocolate và số gam mỗi ngày nào chắc chắn làm xuất hiện mụn?", ["chocolate", "số gam", "chắc chắn"], "diet_uncertain"),
    ]
    return rows


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _similarity(left: str, right: str) -> float:
    a, b = set(_normalize(left).split()), set(_normalize(right).split())
    return (2 * len(a & b) / (len(a) + len(b))) if a and b else 0.0


def _load_corpus() -> tuple[list[dict[str, Any]], str]:
    sources = load_source_manifest(DEFAULT_SOURCE_MANIFEST)
    artifacts = {
        source.source_id: load_parsed_artifact(artifact_path(DEFAULT_PARSED_CACHE, source), source)
        for source in sources
    }
    prepared = _compile_prepared_knowledge(
        sources=sources,
        artifacts=artifacts,
        source_dir=DEFAULT_SOURCE_DIR,
        source_manifest_path=DEFAULT_SOURCE_MANIFEST,
        taxonomy_path=DEFAULT_TAXONOMY,
    )
    return list(prepared["compiled"].records), prepared["identity"].build_id


def _development_registry() -> list[dict[str, str]]:
    roots = [ROOT / relative for relative in DEVELOPMENT_REGISTRY_PATHS]
    candidates: dict[str, str] = {}
    string_pattern = re.compile(r"[\"']([^\"'\r\n]{12,240}\?)[\"']")
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in {".py", ".json", ".md"} or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError:
                continue
            for match in string_pattern.finditer(text):
                question = " ".join(match.group(1).split())
                normalized = _normalize(question)
                if len(normalized) >= 12:
                    candidates.setdefault(normalized, path.relative_to(ROOT).as_posix())
    return [{"normalized_query": q, "source_path": candidates[q]} for q in sorted(candidates)]


def _provenance(record_by_id: dict[str, dict[str, Any]], chunk_id: str) -> dict[str, Any]:
    record = record_by_id[chunk_id]
    text = " ".join(str(record.get("text") or "").split())
    return {
        "chunk_id": chunk_id,
        "source_id": record.get("source_id"),
        "source_title": record.get("source_title"),
        "source_authority": record.get("source_authority"),
        "source_url": record.get("source_url"),
        "section_path": record.get("section_path") or [],
        "record_id": record.get("record_id"),
        "excerpt": text[:900],
    }


def _claim_evidence_snippets(
    claim_key: str,
    claim: dict[str, Any],
    record_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    evidence = CLAIM_EVIDENCE.get(claim_key) or []
    if not evidence:
        raise RuntimeError(f"Gold claim has no evidence snippet: {claim_key}")
    output: list[dict[str, str]] = []
    for chunk_id, snippet in evidence:
        if chunk_id not in claim["chunks"]:
            raise RuntimeError(f"Evidence chunk is not a claim source: {claim_key}, {chunk_id}")
        source_text = str(record_by_id[chunk_id].get("text") or "")
        if _normalize(snippet) not in _normalize(source_text):
            raise RuntimeError(f"Evidence snippet is not present in frozen chunk: {claim_key}, {chunk_id}")
        output.append({"chunk_id": chunk_id, "text": " ".join(snippet.split())})
    return output


def _answerable_case(spec: dict[str, Any], record_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gold_claims = []
    provenance_ids: list[str] = []
    for index, key in enumerate(spec["claim_keys"], start=1):
        claim = CLAIMS[key]
        provenance_ids.extend(claim["chunks"])
        gold_claims.append({
            "claim_id": f"{spec['case_id']}-C{index:02d}",
            "text": claim["text"],
            "source_chunk_ids": claim["chunks"],
            "evidence_snippets": _claim_evidence_snippets(key, claim, record_by_id),
            "annotation_status": "source_annotated_pending_researcher_review",
        })
    unique_ids = list(dict.fromkeys(provenance_ids))
    return {
        "case_id": spec["case_id"],
        "family": spec["family"],
        "category": spec["category"],
        "query": spec["query"],
        "history": spec.get("history", []),
        "expected": {"action": "generate", "reason": None},
        "gold_claims": gold_claims,
        "gold_answer": " ".join(claim["text"] for claim in gold_claims),
        "provenance": [_provenance(record_by_id, cid) for cid in unique_ids],
    }


def _cached_dense_vectors(records: list[dict[str, Any]]) -> tuple[dict[str, list[float]], list[str]]:
    cache = EmbeddingCache(DEFAULT_EMBEDDING_CACHE)
    vectors: dict[str, list[float]] = {}
    misses: list[str] = []
    for record in records:
        vector = cache.get(str(record.get("text") or ""))
        if vector is None:
            misses.append(str(record["chunk_id"]))
        else:
            vectors[str(record["chunk_id"])] = vector
    return vectors, misses


async def _qdrant_absence_audits(
    gaps: list[tuple[str, str, str, list[str], str]],
    record_by_id: dict[str, dict[str, Any]],
    dense_vectors: dict[str, list[float]],
    dense_misses: list[str],
) -> dict[str, dict[str, Any]]:
    def review_candidate(chunk_id: str) -> dict[str, Any]:
        record = record_by_id[chunk_id]
        return {
            "chunk_id": chunk_id,
            "source_id": record.get("source_id"),
            "section_path": record.get("section_path") or [],
            "snippet": " ".join(str(record.get("text") or "").split())[:500],
        }

    def unsupported_requirement(category: str, terms: list[str]) -> str:
        if category == "unsupported_absolute_certainty_guarantee":
            kind = "Cam kết hoặc kết luận tuyệt đối cần được nguồn hỗ trợ trực tiếp"
        elif category == "unsupported_exact_quantity_time":
            kind = "Con số hoặc thời điểm chính xác cần được nguồn nêu trực tiếp"
        else:
            kind = "So sánh, quan hệ hoặc mức chênh lệch cụ thể cần được nguồn nêu trực tiếp"
        return f"{kind}: {', '.join(terms)}."

    client = AsyncQdrantClient(**qdrant_client_kwargs())
    audits: dict[str, dict[str, Any]] = {}
    try:
        aliases = await client.get_aliases()
        alias_map = {item.alias_name: item.collection_name for item in aliases.aliases}
        expected_physical = f"acne_knowledge__{KB_BUILD_ID}"
        if alias_map.get(KNOWLEDGE_ALIAS) != expected_physical:
            raise RuntimeError(
                f"Active knowledge alias mismatch: {alias_map.get(KNOWLEDGE_ALIAS)}"
            )
        info = await client.get_collection(KNOWLEDGE_ALIAS)
        if int(info.points_count or 0) != 512:
            raise RuntimeError("Active knowledge alias does not contain 512 points")
        for case_id, category, query, terms, seed_claim_key in gaps:
            seed_chunk = CLAIMS[seed_claim_key]["chunks"][0]
            seed_vector = dense_vectors.get(seed_chunk)
            if seed_vector is None:
                raise RuntimeError(f"Dense seed vector missing for {case_id}: {seed_chunk}")
            dense = await client.query_points(
                collection_name=KNOWLEDGE_ALIAS,
                query=seed_vector,
                using="dense",
                limit=20,
                with_payload=True,
            )
            sparse = await client.query_points(
                collection_name=KNOWLEDGE_ALIAS,
                query=bm25_document(query),
                using=BM25_VECTOR_NAME,
                limit=20,
                with_payload=True,
            )
            lexical = []
            for record in record_by_id.values():
                normalized_text = _normalize(str(record.get("text") or ""))
                matched = [term for term in terms if _normalize(term) in normalized_text]
                if matched:
                    lexical.append({
                        "chunk_id": record["chunk_id"],
                        "matched_terms": matched,
                        "source_id": record.get("source_id"),
                        "section_path": record.get("section_path") or [],
                        "snippet": " ".join(str(record.get("text") or "").split())[:500],
                    })
            sparse_ids = [str(point.id) for point in sparse.points]
            dense_ids = [str(point.id) for point in dense.points]
            likely_ids = list(dict.fromkeys([
                *[item["chunk_id"] for item in lexical[:5]],
                *sparse_ids[:5],
                *dense_ids[:5],
            ]))
            audits[case_id] = {
                "scope": "entire_active_frozen_corpus",
                "unsupported_factual_requirement": unsupported_requirement(category, terms),
                "candidate_search_completed": True,
                "absence_review_status": "pending_researcher_review",
                "lexical_corpus_scan": {
                    "records_scanned": len(record_by_id),
                    "terms": terms,
                    "candidate_matches": lexical[:25],
                    "candidate_match_count": len(lexical),
                },
                "bm25_candidate_search": {
                    "engine": "qdrant_native_bm25_read_only",
                    "top_k": 20,
                    "candidate_chunk_ids": sparse_ids,
                    "review_candidates": [review_candidate(chunk_id) for chunk_id in sparse_ids[:5]],
                },
                "related_topic_dense_probe": {
                    "engine": "qdrant_cosine_read_only",
                    "probe_method": "cached_related_source_chunk_vector_not_query_embedding",
                    "probe_chunk_id": seed_chunk,
                    "top_k": 20,
                    "candidate_chunk_ids": dense_ids,
                    "review_candidates": [review_candidate(chunk_id) for chunk_id in dense_ids[:5]],
                    "cached_vectors_available": len(dense_vectors),
                    "cached_vectors_missing": len(dense_misses),
                },
                "likely_relevant_sources_and_sections": [review_candidate(chunk_id) for chunk_id in likely_ids[:10]],
                "provider_calls": 0,
                "datastore_writes": 0,
                "review_note": (
                    "Candidate search đã hoàn thành, nhưng kết luận 'không tìm thấy support trong frozen KB' "
                    "chưa được researcher xác nhận; điều này không nói gì về toàn bộ y văn."
                ),
            }
        return audits
    finally:
        await client.close()


def _calibration(record_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def snippet(chunk_id: str, limit: int = 700) -> str:
        return " ".join(str(record_by_id[chunk_id].get("text") or "").split())[:limit]

    extraction = [
        _ex("CAL-EXT-01", "simple_factual", "Mụn trứng cá là bệnh viêm mạn tính của đơn vị nang lông - tuyến bã.", [
            _ref("R01", "Mụn trứng cá là bệnh viêm mạn tính của đơn vị nang lông - tuyến bã.", [["mụn trứng cá"], ["viêm mạn tính", "viêm mạn"], ["đơn vị nang lông tuyến bã", "nang lông tuyến bã"]]),
        ], []),
        _ex("CAL-EXT-02", "simple_factual", "Benzoyl peroxide là hoạt chất kháng khuẩn bôi và có tác dụng tiêu nhân mụn nhẹ.", [
            _ref("R01", "Benzoyl peroxide là hoạt chất kháng khuẩn bôi.", [["benzoyl peroxide"], ["kháng khuẩn bôi", "kháng khuẩn dùng tại chỗ", "topical antimicrobial"]]),
            _ref("R02", "Benzoyl peroxide có tác dụng tiêu nhân mụn nhẹ.", [["benzoyl peroxide"], ["tiêu nhân mụn", "comedolytic"]], [["nhẹ", "mildly"]]),
        ], []),
        _ex("CAL-EXT-03", "qualifier_sensitive", "Bằng chứng về chế độ ăn tải đường huyết thấp còn mâu thuẫn và chỉ gợi ý khả năng giảm mụn.", [
            _ref("R01", "Bằng chứng về chế độ ăn tải đường huyết thấp còn mâu thuẫn.", [["tải đường huyết thấp", "low glycemic"], ["mâu thuẫn", "conflicting"]]),
            _ref("R02", "Bằng chứng chỉ gợi ý chế độ ăn tải đường huyết thấp có thể giảm mụn.", [["tải đường huyết thấp", "low glycemic"], ["giảm mụn", "reduce acne"]], [["gợi ý", "suggest"], ["có thể", "khả năng", "may"]]),
        ], ["chắc chắn", "chữa khỏi"]),
        _ex("CAL-EXT-04", "qualifier_sensitive", "Điều trị duy trì không phải lúc nào cũng cần và có thể cân nhắc khi thường xuyên tái phát.", [
            _ref("R01", "Điều trị duy trì không phải lúc nào cũng cần.", [["điều trị duy trì"], ["không phải lúc nào cũng cần", "không luôn cần"]]),
            _ref("R02", "Có thể cân nhắc điều trị duy trì khi thường xuyên tái phát.", [["điều trị duy trì"], ["tái phát thường xuyên", "thường xuyên tái phát"]], [["có thể cân nhắc", "cân nhắc"]]),
        ], ["bắt buộc"]),
        _ex("CAL-EXT-05", "multi_clause_treatment", "Không dùng kháng sinh bôi đơn trị liệu; phối hợp benzoyl peroxide có thể hạn chế kháng kháng sinh.", [
            _ref("R01", "Không dùng kháng sinh bôi làm đơn trị liệu.", [["kháng sinh bôi"], ["không dùng", "không nên dùng"], ["đơn trị liệu", "một mình"]]),
            _ref("R02", "Phối hợp benzoyl peroxide có thể hạn chế kháng kháng sinh.", [["benzoyl peroxide"], ["kháng kháng sinh"], ["hạn chế", "giảm nguy cơ"]], [["có thể", "giúp"]]),
        ], []),
        _ex("CAL-EXT-06", "multi_clause_treatment", "Isotretinoin làm giảm tuyến bã, điều chỉnh sừng hóa và có tác dụng chống viêm.", [
            _ref("R01", "Isotretinoin làm giảm hoạt động tuyến bã.", [["isotretinoin"], ["giảm tuyến bã", "giảm hoạt động tuyến bã", "giảm bài tiết tuyến bã"]]),
            _ref("R02", "Isotretinoin điều chỉnh quá trình sừng hóa.", [["isotretinoin"], ["điều chỉnh sừng hóa", "bình thường hóa sừng hóa"]]),
            _ref("R03", "Isotretinoin có tác dụng chống viêm.", [["isotretinoin"], ["chống viêm", "kháng viêm"]]),
        ], []),
        _ex("CAL-EXT-07", "comparison_mixed_terminology", "Open comedone là mụn đầu đen, còn closed comedone là mụn đầu trắng.", [
            _ref("R01", "Open comedone là mụn đầu đen.", [["open comedone"], ["mụn đầu đen", "blackhead"]]),
            _ref("R02", "Closed comedone là mụn đầu trắng.", [["closed comedone"], ["mụn đầu trắng", "whitehead"]]),
        ], []),
        _ex("CAL-EXT-08", "comparison_mixed_terminology", "Benzoyl peroxide là antimicrobial chứ không phải topical antibiotic; clindamycin mới là kháng sinh bôi.", [
            _ref("R01", "Benzoyl peroxide là antimicrobial, không phải topical antibiotic.", [["benzoyl peroxide"], ["antimicrobial", "kháng khuẩn"], ["không phải topical antibiotic", "không phải kháng sinh bôi"]]),
            _ref("R02", "Clindamycin là kháng sinh bôi.", [["clindamycin"], ["kháng sinh bôi", "topical antibiotic"]]),
        ], []),
    ]
    checking_rows = [
        ("CAL-CHK-01", "vi_to_vi", snippet("1b0c75be-1e53-505c-a3bd-bc0f85b7e2ad"), "Người bị mụn được khuyên dùng sản phẩm rửa mặt và dưỡng ẩm phù hợp.", "SUPPORTED", ["entity_distinction"]),
        ("CAL-CHK-02", "vi_to_vi", snippet("460222d6-c4ba-56f9-a26c-bf255b6afb39"), "Có thể giảm tần suất retinoid bôi trong vài tuần đầu để hạn chế kích ứng.", "SUPPORTED", ["qualifier_modality"]),
        ("CAL-CHK-03", "vi_to_vi", snippet("dbeb735c-2e0e-57a5-af78-eb38ecdc0a24"), "Nguồn không khuyến nghị dùng kháng sinh làm điều trị đơn độc.", "SUPPORTED", ["negation"]),
        ("CAL-CHK-04", "vi_to_vi", snippet("1a5c4052-4796-5a98-b56a-331ed3d98d51"), "Mọi nhân mụn đóng đều có kích thước chính xác 4 mm.", "NOT_SUPPORTED", ["exact_number"]),
        ("CAL-CHK-05", "vi_to_vi", snippet("a5d5921e-d85b-5554-89c6-6b090b63cc9f"), "Benzoyl peroxide chắc chắn không bao giờ gây kích ứng.", "NOT_SUPPORTED", ["negation", "absolute_claim"]),
        ("CAL-CHK-06", "vi_to_vi", snippet("530439f4-0d52-505c-9ced-492c20a38f51"), "Mọi người đều bắt buộc dùng điều trị duy trì suốt đời.", "NOT_SUPPORTED", ["qualifier_modality", "absolute_claim"]),
        ("CAL-CHK-07", "en_to_vi", snippet("904d057a-aa4d-5d6a-bcb7-7379c5137b60"), "Mụn trứng cá là bệnh viêm mạn tính của đơn vị nang lông - tuyến bã.", "SUPPORTED", ["cross_language_entailment"]),
        ("CAL-CHK-08", "en_to_vi", snippet("db4632c5-44ed-5dea-80c2-addb0a2534b3"), "Benzoyl peroxide là hoạt chất kháng khuẩn bôi và có thể làm bạc màu vải.", "SUPPORTED", ["entity_distinction", "cross_language_entailment"]),
        ("CAL-CHK-09", "en_to_vi", snippet("2326dc1b-cf4b-520b-8798-f9617fafc0b0"), "Mụn fulminans cần được đánh giá trong vòng 24 giờ.", "SUPPORTED", ["exact_time", "cross_language_entailment"]),
        ("CAL-CHK-10", "en_to_vi", snippet("7bbeb95e-489e-5583-ae76-b128853168e6"), "Salicylic acid diệt C. acnes mạnh hơn benzoyl peroxide.", "NOT_SUPPORTED", ["unsupported_comparison", "entity_distinction"]),
        ("CAL-CHK-11", "en_to_vi", snippet("850609c9-471a-54cc-934f-478ad48d826d"), "Adapalene chắc chắn làm sạch toàn bộ mụn trong đúng 3 ngày.", "NOT_SUPPORTED", ["exact_time", "absolute_claim"]),
        ("CAL-CHK-12", "en_to_vi", snippet("5a192e07-bed0-5192-8156-6f45ad41f944"), "Azelaic acid 20% luôn hiệu quả gấp đôi azelaic acid 15%.", "NOT_SUPPORTED", ["exact_number", "unsupported_comparison"]),
    ]
    checking = [{"item_id": i, "task": "claim_checking", "language_direction": lang, "evidence": evidence, "claim": claim, "expected_label": label, "critical_dimensions": dimensions, "annotation_status": "source_annotated_pending_researcher_review"} for i, lang, evidence, claim, label, dimensions in checking_rows]
    return {
        "schema_version": "evaluator_calibration_contract",
        "evaluator_model": EVALUATOR_MODEL,
        "annotation_authority": "frozen_kb_source_annotation_not_human_review",
        "counts": {"total": 20, "claim_extraction": 8, "claim_checking": 12, "supported": 6, "not_supported": 6, "vi_to_vi": 6, "en_to_vi": 6},
        "claim_extraction": extraction,
        "claim_checking": checking,
    }


def _ref(
    reference_id: str,
    text: str,
    required_concept_groups: list[list[str]],
    qualifier_concept_groups: list[list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "reference_id": reference_id,
        "text": text,
        "required_concept_groups": required_concept_groups,
        "qualifier_concept_groups": qualifier_concept_groups or [],
    }


def _ex(
    item_id: str,
    category: str,
    text: str,
    reference_claims: list[dict[str, Any]],
    forbidden: list[str],
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "task": "claim_extraction",
        "category": category,
        "input_text": text,
        "reference_claims": reference_claims,
        "acceptance": {
            "forbidden_inventions": forbidden,
            "criteria": ["factual_coverage", "atomicity", "no_invention", "qualifier_preservation"],
            "automatic_decision_scope": "reference-aware lexical concepts with one-to-one atomic matching",
            "unresolved_semantic_cases": "pending_researcher_review",
        },
        "annotation_status": "source_annotated_pending_researcher_review",
    }


async def build() -> None:
    load_dotenv(dotenv_path=ROOT / ".env")
    records, build_id = _load_corpus()
    if build_id != KB_BUILD_ID or len(records) != 512:
        raise RuntimeError(f"Unexpected corpus identity: {build_id}, {len(records)} records")
    record_by_id = {str(record["chunk_id"]): record for record in records}
    missing_claim_chunks = sorted({cid for claim in CLAIMS.values() for cid in claim["chunks"] if cid not in record_by_id})
    if missing_claim_chunks:
        raise RuntimeError(f"Gold claim chunks are absent: {missing_claim_chunks}")

    specs = [{"case_id": case_id, "family": "answerable_single_turn", "category": category, "query": query, "claim_keys": claim_keys, "history": []} for case_id, category, query, claim_keys in _single_specs()]
    specs.extend(_multi_specs())
    answerable = [_answerable_case(spec, record_by_id) for spec in specs]

    dense_vectors, dense_misses = _cached_dense_vectors(records)
    gap_specs = _gap_specs()
    absence = await _qdrant_absence_audits(gap_specs, record_by_id, dense_vectors, dense_misses)
    evidence_gap = [{"case_id": case_id, "family": "evidence_gap", "category": category, "query": query, "history": [], "expected": {"action": "abstain", "reason": "evidence_gap"}, "gold_claims": [], "gold_answer": None, "provenance": [], "absence_verification": absence[case_id]} for case_id, category, query, _terms, _seed in gap_specs]
    cases = [*answerable, *evidence_gap]

    internal_near_duplicates = []
    for left_index, left in enumerate(cases):
        for right in cases[left_index + 1:]:
            score = _similarity(left["query"], right["query"])
            if score >= 0.80:
                internal_near_duplicates.append({
                    "left": left["case_id"],
                    "right": right["case_id"],
                    "token_dice": round(score, 4),
                })
    claim_set_cases: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    for case in answerable:
        claim_set = tuple(sorted(claim["text"] for claim in case["gold_claims"]))
        claim_set_cases[claim_set].append(case["case_id"])
    repeated_claim_sets = [case_ids for case_ids in claim_set_cases.values() if len(case_ids) > 1]
    if internal_near_duplicates:
        raise RuntimeError(
            "Internal benchmark deduplication failed: "
            f"near={internal_near_duplicates}"
        )

    registry = _development_registry()
    registry_questions = [entry["normalized_query"] for entry in registry]
    exact = [case["case_id"] for case in cases if _normalize(case["query"]) in registry_questions]
    near = []
    for case in cases:
        best = max((_similarity(case["query"], question) for question in registry_questions), default=0.0)
        if best >= 0.86:
            near.append({"case_id": case["case_id"], "max_token_dice": round(best, 4)})
    if exact or near:
        raise RuntimeError(f"Development prompt contamination detected: exact={exact}, near={near}")
    manual_matches = []
    for case in cases:
        for pattern in MANUAL_DEVELOPMENT_EXCLUSIONS:
            score = _similarity(case["query"], pattern["query"])
            if score >= 0.86:
                manual_matches.append({
                    "case_id": case["case_id"],
                    "pattern_id": pattern["pattern_id"],
                    "token_dice": round(score, 4),
                })
    if manual_matches:
        raise RuntimeError(f"Known development-pattern contamination detected: {manual_matches}")

    category_counts = Counter(case["category"] for case in cases)
    source_cases: defaultdict[str, set[str]] = defaultdict(set)
    source_claims: Counter[str] = Counter()
    for case in answerable:
        for provenance in case["provenance"]:
            source_cases[str(provenance["source_id"])].add(case["case_id"])
        for claim in case["gold_claims"]:
            for cid in claim["source_chunk_ids"]:
                source_claims[str(record_by_id[cid]["source_id"])] += 1
    source_coverage = [{"source_id": source_id, "benchmark_cases": len(source_cases[source_id]), "gold_claim_references": source_claims[source_id]} for source_id in sorted(source_cases)]
    gold_claim_counts = Counter(len(case["gold_claims"]) for case in answerable)
    gold_claim_distribution = {
        "1": gold_claim_counts[1],
        "2": gold_claim_counts[2],
        "3": gold_claim_counts[3],
        "4_or_more": sum(count for size, count in gold_claim_counts.items() if size >= 4),
        "mean": round(sum(size * count for size, count in gold_claim_counts.items()) / len(answerable), 4),
    }

    benchmark = {
        "schema_version": "formal_benchmark_contract",
        "language": "vi",
        "researcher_review_status": "pending",
        "gold_truth_authority": "active_frozen_kb_and_source_documents",
        "cases": cases,
        "coverage": {
            "category_counts": dict(sorted(category_counts.items())),
            "source_coverage": source_coverage,
            "gold_claim_count_distribution": gold_claim_distribution,
            "source_distribution_note": (
                "Phân bố phản ánh nguồn phù hợp trong frozen KB và không được cân bằng nhân tạo; "
                "đây không phải phân bố nguồn không thiên lệch."
            ),
        },
        "anti_contamination": {
            "registry_source_paths": list(DEVELOPMENT_REGISTRY_PATHS),
            "development_registry_entries": len(registry),
            "similarity_method": "accent-insensitive unique-token Dice",
            "automated_registry_near_match_threshold": 0.86,
            "exact_matches": 0,
            "near_matches_at_or_above_0_86": 0,
            "manual_exclusion_patterns": MANUAL_DEVELOPMENT_EXCLUSIONS,
            "manual_pattern_matches_at_or_above_0_86": 0,
            "internal_query_near_duplicates_at_or_above_0_80": 0,
            "repeated_gold_claim_sets": len(repeated_claim_sets),
            "repeated_gold_claim_set_cases": repeated_claim_sets,
            "repeated_gold_claim_set_note": (
                "Repeated minimal claim sets are reported for review and are not padded with unrelated claims."
            ),
            "scope_limitation": (
                "Kết quả chỉ bao phủ string registry trong các path đã liệt kê và các pattern thủ công đã biết; "
                "cùng behavior class vẫn có thể xuất hiện và đây không phải chứng minh contamination tuyệt đối bằng 0."
            ),
        },
        "absence_audit_environment": {"active_alias": KNOWLEDGE_ALIAS, "active_physical_collection": f"acne_knowledge__{KB_BUILD_ID}", "corpus_records": 512, "cached_dense_vectors": len(dense_vectors), "cached_dense_misses": len(dense_misses), "provider_calls": 0, "datastore_writes": 0},
    }
    benchmark_sha = canonical_json_sha256(benchmark)

    manifest = {
        "schema_version": "formal_benchmark_manifest_contract",
        "evaluation_base_sha": BASE_SHA,
        "active_kb_build_id": KB_BUILD_ID,
        "benchmark_sha256": benchmark_sha,
        "benchmark_counts": {"total": 100, "answerable": 70, "single_turn": 55, "multi_turn": 15, "evidence_gap": 30},
        "category_counts": dict(sorted(category_counts.items())),
        "evaluator_model": EVALUATOR_MODEL,
        "ragchecker": {
            "version": RAGCHECKER_VERSION,
            "commit": RAGCHECKER_COMMIT,
            "package": RAGCHECKER_PACKAGE,
            "repository": "https://github.com/amazon-science/RAGChecker",
            "paper": "https://arxiv.org/abs/2408.08067",
            "checker_output_contract": ["Entailment", "Neutral", "Contradiction"],
            "per_case_metric_scale": "ratio_0_1",
            "aggregate_metric_scale": "percent_0_100",
            "headline_reporting_scale": "percent_0_100",
        },
        "negative_rejection_reference": {"method": "RGB-inspired structured-action adaptation", "paper": "https://doi.org/10.1609/aaai.v38i16.29728", "arxiv": "https://arxiv.org/abs/2309.01431"},
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "researcher_review_status": "pending",
        "production_behavior_changed": False,
    }
    calibration = _calibration(record_by_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results" / "figures").mkdir(parents=True, exist_ok=True)
    write_pretty_json(OUTPUT_DIR / "benchmark_100.json", benchmark)
    write_pretty_json(OUTPUT_DIR / "benchmark_manifest.json", manifest)
    write_pretty_json(OUTPUT_DIR / "evaluator_calibration.json", calibration)
    print(json.dumps({"benchmark_sha256": benchmark_sha, "cases": len(cases), "answerable": len(answerable), "evidence_gap": len(evidence_gap), "category_counts": dict(sorted(category_counts.items())), "source_coverage": source_coverage, "qdrant_dense_audits": len(absence), "provider_calls": 0, "datastore_writes": 0}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build formal evaluation benchmark artifacts read-only from the active corpus.")
    parser.parse_args()
    subprocess.run(["git", "diff", "--quiet", "--", "src", "scripts"], cwd=ROOT, check=True)
    asyncio.run(build())


if __name__ == "__main__":
    main()
