# Báo cáo kiểm toán và tinh gọn repository

## Phạm vi và kết luận

- Ngày audit: 2026-10-12 (Asia/Saigon).
- Repository: `MinhHoang1403/Acne_Advisor_AI`.
- Baseline sản phẩm: `366f3ad836c36be36565caad4a94ad17a7a6278f`.
- Nhánh thực hiện: `maintenance/deep-repo-cleanup`.
- Pull request: [#119 — chore: deep repository cleanup and audit](https://github.com/MinhHoang1403/Acne_Advisor_AI/pull/119).
- Final maintenance SHA: PR #119 head (Git cannot self-embed the SHA của commit chứa chính tài liệu này); merge SHA được ghi trong final handoff sau CI.
- Baseline: 271 file tracked, 45 thư mục tracked.
- Sau cleanup: 246 file tracked, 37 thư mục tracked.
- Thay đổi production: không.
- Public API, prompt y khoa, safety, retrieval, KB identity, vector schema, database schema: không đổi.

Inventory kiểm toán ở [`repository_inventory.csv`](repository_inventory.csv) chứa một
disposition cho toàn bộ 271 file baseline và hai artifact audit mới. Inventory dùng union
trước/sau nên vẫn giữ hàng cho mọi file đã xóa.

## Restore point

Trước cleanup, working tree sạch và không có merge/rebase/cherry-pick dang dở. Restore point
đã được tạo và xác minh:

- Source branch: `research/local-ragchecker-extraction-contract`.
- Source HEAD: `c4a0cf7a704cf74dca88f97b82084787929547ef`.
- Safety tag local: `safety/deep-repo-cleanup-20261012-142421`.
- Bundle ngoài repository: `C:\Study\Acne Advisor AI\repository-backups\deep-repo-cleanup-20261012-142421\acne-agent-system-all-refs.bundle`.
- SHA-256: `2DF5EFE1F8EB8A7E5DB3A00435F1C7B1740E086D93542E302A7E4C34761939C6`.
- `git bundle verify`: pass; bundle ghi complete history, 129 refs.
- Phạm vi bundle: `main`, toàn bộ branch local, remote-tracking refs, tag, `refs/stash`,
  safety tag và 57 dangling commits được gắn ref tạm trong bundle.
- Hướng dẫn restore và checksum nằm cạnh bundle, không nằm trong repository.

Git bundle không chứa `.env`, database, model, cache, Qdrant, PostgreSQL, Redis, Neo4j hoặc
file ignored. Vì vậy audit không xóa hoặc mutate bất kỳ dữ liệu runtime nào.

## Inventory filesystem và Git

### Git topology tại baseline

- Remote: `origin` trỏ đến repository GitHub canonical.
- Remote branches: chỉ `main`.
- Open PR: 0.
- Worktree: một worktree hiện hành; không detached HEAD.
- Stash: 3; đều được giữ và nằm trong bundle.
- Submodule: không có.
- Git LFS: client có sẵn nhưng repository không có LFS object tracked.
- Local branch có trước audit ngoài `main`: 20.
- Pending Git operation: không có.
- `git fsck --full --no-reflogs`: repository đọc được; phát hiện 57 dangling commit và
  các object dangling khác. Không object nào bị prune; dangling commits có ref trong bundle.

### Vùng ignored/local được giữ nguyên

| Vùng | Quan sát | Quyết định |
|---|---:|---|
| `data/` | khoảng 14,55 GB, gồm backup/cache/datastore local | Giữ nguyên; không có backup runtime riêng trong task này |
| `.venv/` | khoảng 3,48 GB | Giữ nguyên; dependency local |
| `reports/` | khoảng 171,7 MB | Giữ nguyên; ignored runtime output |
| `src/frontend/node_modules/`, `dist/` | generated/ignored | Giữ; `npm ci` và build có thể tái tạo |
| `.env` | ignored secret-bearing config | Không đọc giá trị, không sao chép, không commit |
| logs, caches, IDE state, egg-info | generated/ignored | Không commit; không cần xóa để Git tree sạch |

Một backup Qdrant cũ dưới `data/backups/` có đường dẫn vượt giới hạn Win32 khi Git liệt kê
ignored files. Đây là local runtime backup, không phải tracked source; không được xóa khi chưa
có backup dữ liệu riêng và owner approval.

## Reachability và dependency audit

### Production/backend

- 83 Python module dưới `src/` được parse bằng AST và lần theo từ FastAPI, graph, CLI và
  operational script roots.
- Không có substantive Python module nào được xác nhận orphan. Các `__init__.py` không có
  incoming import trực tiếp là package boundary/exports, không phải dead implementation.
- FastAPI đăng ký 10 route hiện hành.
- LangGraph đăng ký đúng 8 node: `prepare`, `guard`, `decide`, `retrieve`, `assess`,
  `generate`, `abstain`, `finalize`.
- Dynamic/lazy consumer đã được kiểm tra cho reranker (`sentence_transformers`, `torch`),
  parser (`llama_parse`), evaluator (`openai`, `ragchecker`), Langfuse và package entrypoint.
- `ruff check src scripts tests`: pass tại baseline.

### Frontend

- 26 file JS/JSX đều reachable từ `src/main.jsx`, config hoặc Node test roots.
- Không phát hiện component, API wrapper hay utility orphan.
- `favicon.svg` được `index.html` và test sử dụng.
- `public/icons.svg` là sprite social/documentation từ commit đầu, không có import, URL,
  test, CSS hoặc HTML reference; file này được xóa.

### Dependencies và config

- Toàn bộ direct runtime dependency có consumer trực tiếp, lazy consumer, CLI usage hoặc
  driver/transitive role rõ ràng. `uv pip check` báo 147 package tương thích.
- `uvicorn` là runtime CLI; `asyncpg`/`psycopg2-binary` là database drivers;
  `hiredis` là Redis extra; không được coi là unused chỉ vì thiếu direct import.
- `openai`, `ragchecker`, `jupyterlab` thuộc evaluation requirements riêng, không nằm trong
  production lock; xóa chúng sẽ làm mất evaluator hiện hành nên chưa được phép.
- Không xóa dependency Python hoặc npm nào.
- Frontend `npm audit`: 0 vulnerability.
- Secret scan trong release-readiness checker: pass; không ghi raw secret vào artifact audit.
- Compose production config: pass. Compose observability yêu cầu file env secret riêng đúng
  như `docs/OBSERVABILITY.md`; config không-secret trần bị fail-closed là hành vi dự kiến.

## Cleanup đã thực hiện

### Development evaluation archive

`research_archive/` không có runtime, script, test hoặc CI consumer. Backlink duy nhất từ
README/evaluation README đã được cập nhật. Dự án không còn nghĩa vụ giữ toàn bộ artifact luận
văn trong working tree; các bytes cũ, SHA và manifest vẫn phục hồi được qua Git history,
safety tag và bundle đã verify.

26 file đã xóa:

1. `research_archive/README.md`
2. `research_archive/development_evaluation/archive_manifest.json`
3. `research_archive/development_evaluation/aborted_formal_run_978093b9/calibration_adjudication.json`
4. `research_archive/development_evaluation/aborted_formal_run_978093b9/evaluator_calibration_results.json`
5. `research_archive/development_evaluation/aborted_formal_run_978093b9/raw_results.json`
6. `research_archive/development_evaluation/audits/post_improvement_47b10954/audit_validation_report.md`
7. `research_archive/development_evaluation/audits/post_improvement_47b10954/audit_validation_summary.json`
8. `research_archive/development_evaluation/audits/post_improvement_47b10954/case_forensic_audit.csv`
9. `research_archive/development_evaluation/audits/post_improvement_47b10954/forensic_analysis.py`
10. `research_archive/development_evaluation/audits/post_improvement_47b10954/forensic_audit.md`
11. `research_archive/development_evaluation/audits/post_improvement_47b10954/forensic_summary.json`
12. `research_archive/development_evaluation/audits/post_improvement_47b10954/validate_forensic_audit.py`
13. `research_archive/development_evaluation/audits/post_improvement_47b10954/validated_case_forensic_audit.csv`
14. `research_archive/development_evaluation/formal_run_baseline/case_metrics.csv`
15. `research_archive/development_evaluation/formal_run_baseline/evaluator_calibration_results.json`
16. `research_archive/development_evaluation/formal_run_baseline/metrics_summary.csv`
17. `research_archive/development_evaluation/formal_run_baseline/ragchecker_checkpoint.json`
18. `research_archive/development_evaluation/formal_run_baseline/raw_results.json`
19. `research_archive/development_evaluation/historical_notebooks/formal_evaluation_manual_output_before_evaluator_config.ipynb`
20. `research_archive/development_evaluation/post_improvement_47b10954/calibration_adjudication.json`
21. `research_archive/development_evaluation/post_improvement_47b10954/case_metrics.csv`
22. `research_archive/development_evaluation/post_improvement_47b10954/evaluator_calibration_results.json`
23. `research_archive/development_evaluation/post_improvement_47b10954/metrics_summary.csv`
24. `research_archive/development_evaluation/post_improvement_47b10954/ragchecker_checkpoint.json`
25. `research_archive/development_evaluation/post_improvement_47b10954/raw_results.json`
26. `research_archive/development_evaluation/post_improvement_47b10954/researcher_notebook_calibration_review_required.ipynb`

### Unused frontend asset

27. `src/frontend/public/icons.svg` — zero references; không phải favicon đang dùng.

Tổng giảm: 27 file, 11.547.753 byte tracked. `.gitignore`, `README.md` và
`evaluation/results/README.md` được cập nhật để không còn exception/backlink stale.

## Các file trông cũ nhưng phải giữ

- `evaluation/formal_evaluation.ipynb`: được byte/identity contract test kiểm tra.
- `evaluation/formal_evaluation_support.py`, `build_formal_package.py`,
  `create_formal_notebook.py`: evaluator/reproducibility package có test consumer.
- `evaluation/results/formal_run_2d5f0124/**`: canonical historical result có hash và identity;
  không phải production truth hiện tại nhưng vẫn là reusable research evidence.
- `evaluation/benchmark_100.json`, calibration và adjudication files: benchmark/gold boundary.
- Compatibility fields, legacy cache/version promotion và legacy embedding-reuse guards:
  đều có regression tests hoặc migration/fail-closed consumer.
- `src/frontend/src/utils/sessionMerge.js`, storage/history code, model selector và source
  presentation: đều reachable và được test; không xóa vì UI path ít dùng.
- `data/knowledge_build_manifest.json`: giữ nguyên tuyệt đối; không rewrite để hợp cleanup.

## Duplicate/dead-code findings

- Không xác nhận duplicate production implementation đủ bằng chứng để xóa.
- Các basename `contracts.py`, `provider.py`, `versioning.py` giống nhau nhưng thuộc bounded
  context khác nhau, không phải duplicate.
- Formal/development result filenames giống nhau nhưng đại diện run identity khác nhau; archive
  phát triển bị externalize vì lifecycle, không tuyên bố byte-duplicate.
- Không sửa formatter, fallback hoặc safety compatibility path.
- Không thay đổi TODO/deprecated marker vì các marker quan sát được đều mô tả compatibility
  boundary có test, không phải TODO bỏ quên.

## Audit Phase 1 và Phase 2

Phase 1 được lần theo từ manifest/source catalog qua parser, normalization, chunking, filtering,
embedding, BM25, Qdrant indexing/activation, taxonomy, EntityCards và Neo4j. Không tìm thấy
builder song song đã được xác nhận obsolete. Không parse, embed, rebuild, activate hoặc rollback KB.

Phase 2 được lần theo `React -> chatApi -> FastAPI -> run_clinical_agent -> LangGraph ->
retrieval -> reranker -> packer -> ActionDecision -> generation/finalize -> API -> React` và các
nhánh safety, cache, retry, provider/reranker fallback, timeout, persistence. Không có module nào
trong chuỗi này bị xóa hoặc refactor.

## Branch disposition

Mọi branch dưới đây có unique commits theo ancestry, nên không dùng `git branch -D`. Restore
bundle giữ nguyên tất cả branch. Không branch experimental nào được merge vào maintenance/main.

| Branch | Ahead/behind so với main | Disposition | Bằng chứng/quyết định |
|---|---:|---|---|
| `architecture-responsibility-audit` | 1/16 | SUPERSEDED | audit-only; trách nhiệm canonical đã đi qua PR #111, #112 và docs #117 |
| `audit/runtime-architecture-checkpoint` | 1/4 | SUPERSEDED | checkpoint research; main #117 là tài liệu runtime cuối |
| `evaluation/thesis-final-evaluation` | 3/0 | RESEARCH_ONLY | protocol/output thesis; evaluator quota blocker, không đổi production |
| `evidence-path-audit` | 1/19 | ALREADY_MERGED/SUPERSEDED | hai fix production tương ứng đã ở main qua `d683abe` và `001eb4f`; artifact audit không cần merge |
| `evidence-selection-research` | 1/19 | FAILED_EXPERIMENT | candidates reject/inconclusive, cần label review |
| `evidence-selector-focused-research` | 1/16 | RESEARCH_ONLY | adjudication packet chưa có owner decision |
| `information-need-decomposition-research` | 1/16 | FAILED_EXPERIMENT | gate fail 3/13 GOOD, stability/critical gate fail |
| `requested-information-research` | 1/16 | FAILED_EXPERIMENT | schema validity và critical-case gate fail |
| `research/benchmark-current-kb-adjudication` | 1/12 | RESEARCH_ONLY | current-KB review hữu ích nhưng kết thúc `STEP3_ARCHITECTURE_BLOCKER` |
| `research/local-ragchecker-evaluator` | 5/0 | FAILED_EXPERIMENT | `LOCAL_EVALUATOR_QUALITY_FAILED` |
| `research/local-ragchecker-evaluator-9b` | 7/0 | FAILED_EXPERIMENT | Qwen 9B `LOCAL_EVALUATOR_QUALITY_FAILED` |
| `research/local-ragchecker-extraction-contract` | 12/0 | RESEARCH_ONLY/REVIEW_REQUIRED | development paused for independent owner adjudication; không tuyên bố evaluator hoàn chỉnh |
| `research/ragas-shadow-evaluation` | 3/0 | FAILED_EXPERIMENT | calibration gate không đạt; formal shadow không chạy |
| `research/reranker-performance-study` | 1/10 | ALREADY_MERGED | BF16 finding đã được production hóa qua PR #115 |
| `research/reranker-precision-retry-validation` | 1/10 | ALREADY_MERGED | full-union BF16 evidence đã được dùng cho PR #115 |
| `research/reranker-retry-validation` | 1/10 | ALREADY_MERGED | depth-24 rejected; production giữ full union |
| `research/retrieval-first-boundary-study` | 2/12 | FAILED_EXPERIMENT | `RETRIEVAL_FIRST_QUERY_REWRITE_BLOCKER`; không đổi routing |
| `research/retry-evidence-loss` | 1/6 | ALREADY_MERGED | root cause/fix đã được đưa qua PR #116 |
| `research/simple-path-baseline` | 1/12 | RESEARCH_ONLY | baseline/gates chưa freeze do safety/label blocker |
| `retrieval-research` | 1/21 | RESEARCH_ONLY | method study read-only; không có production adoption commit |

Branch đã tích hợp mới trong cleanup: không. Branch đã xóa: không. Branch còn cần owner review:
`research/local-ragchecker-extraction-contract`, `evidence-selector-focused-research` và ba stash.
Các branch ALREADY_MERGED chỉ có kết luận/fix tương ứng ở main dưới SHA khác; branch research vẫn
được giữ vì `git branch -d` không được phép xóa unique ancestry và task cấm force deletion.

## Verification

### Baseline trước cleanup

| Gate | Kết quả |
|---|---|
| GitHub Actions tại `366f3ad` | `Reproducible Python environment`: success; `Frontend contract and build`: success |
| Python version | 3.11.9 |
| Dependency consistency | pass qua `uv pip check` (147 packages); local venv không cài module `pip` |
| Reproducibility checker local | fail duy nhất `runtime_imports`: Windows Application Control chặn `torch/lib/caffe2_nvrtc.dll` |
| Release readiness offline local | 16/17; fail kế thừa từ runtime import trên |
| Ruff lint | pass |
| Pytest local | 1.065 pass, 11 fail; mọi failure là import torch/sentence-transformers bị App Control chặn hoặc checker kế thừa; coverage 82,80% |
| Ruff format check | fail tồn tại sẵn: 88 file would be reformatted; không bulk-format code ổn định trong cleanup |
| Compileall | pass |
| Frontend | 52/52 pass; build pass; lint pass; audit 0 vulnerability |
| Production Compose config | pass |
| Whitespace | pass |

Baseline GitHub CI là môi trường sạch canonical và đã pass đúng SHA. Lỗi local được giữ làm
baseline so sánh, không sửa bằng cách tắt test, giảm coverage hoặc thay dependency.

### Sau cleanup

Full Python verification được chạy trong image khóa `acne-thesis-evaluation:366f3ad8`,
repository mount read-only, network bị tắt và toàn bộ paid-provider key bị làm rỗng. Cách chạy
này tránh Windows Application Control nhưng không thể ghi source, gọi provider hay mutate KB.

| Gate | Kết quả sau cleanup |
|---|---|
| `python -m pip check` | pass |
| Reproducibility checker | pass, gồm toàn bộ runtime imports |
| Release readiness offline | pass 17/17; tracked-secret scan pass |
| Ruff lint | pass |
| Full pytest | 1.076/1.076 pass; coverage 83,20% |
| Compileall | pass |
| FastAPI/LangGraph import smoke | pass; 14 framework routes gồm docs/OpenAPI và `CompiledStateGraph` |
| Targeted formal-evaluation/method tests | 97/97 pass (`--no-cov`) |
| Markdown local links | pass trên toàn bộ file tracked |
| JSON/YAML/notebook/CSV parse | pass trên toàn bộ file tracked |
| Frontend clean install/tests/build/lint/audit | 52/52 pass; build/lint pass; 0 vulnerability |
| Production + observability Compose config | pass; observability dùng secret ngẫu nhiên chỉ trong process |
| Git whitespace check | pass |
| Ruff format check | vẫn fail đúng baseline: 88 file would be reformatted, 77 file formatted |

Không có post-cleanup regression. Format debt tồn tại trước task và không được che bằng config
hoặc một diff format hàng loạt. Required CI của PR #119 và final merge SHA được ghi trong final
handoff sau khi GitHub chạy hai jobs trên final PR head.

## Behavioral invariants

Cleanup chỉ xóa artifact không được load, do đó:

- API `/chat` và response schema không đổi.
- Session/history, model selection và source presentation không đổi.
- LangGraph vẫn 8 node; ActionPolicy không đổi.
- Dense + BM25 + RRF, BGE policy, retry retained candidates và context budget không đổi.
- Safety-before-cache, evidence gap, source allowlist, cache identity và provider fallback không đổi.
- PostgreSQL persistence behavior và Qdrant/Neo4j active state không đổi.
- Không đổi KB content, knowledge build ID hoặc pipeline fingerprint.

## Technical debt và owner decisions còn lại

1. Windows Application Control trên máy này chặn CUDA PyTorch DLL; cần allow policy hoặc môi
   trường sạch để local full suite giống CI.
2. 88 file chưa đồng nhất với `ruff format --check`; CI hiện chỉ enforce `ruff check`. Không
   format hàng loạt trong task cleanup vì sẽ tạo diff rộng không có giá trị hành vi.
3. Local evaluator 4B/9B không đạt quality gate; extraction-contract study còn cần independent
   adjudication. Không được mô tả repository là có local semantic evaluator hoàn chỉnh.
4. Sáu benchmark policy labels còn cần owner choice; không âm thầm sửa gold.
5. Formal evaluator hiện hành giữ dependency OpenAI/RAGChecker và có thể cần paid provider khi
   chạy thật; cleanup không gọi provider. Xóa nó sẽ làm mất semantic metric capability nên cần
   owner approval riêng.
6. Ba stash và các unique research branch được bundle bảo vệ nhưng chưa xóa; owner có thể phê
   duyệt một vòng archival/deletion riêng nếu muốn giảm local branch clutter.
7. Local runtime data/backup lớn và một cây Qdrant có đường dẫn dài; cần kế hoạch backup dữ liệu
   riêng trước bất kỳ filesystem cleanup nào.

## Checklist trả lời yêu cầu cuối

1. Vấn đề ban đầu: archive 11,8 MB trong active tree, một asset orphan, stale backlink/ignore,
   20 branch research và 3 stash chưa có disposition tập trung.
2. Dead files thực sự: ít; xác nhận 27 file, không xác nhận dead production module.
3. File xóa: danh sách 27 file ở mục Cleanup.
4. Lý do: zero active consumer, đã có restore verified, không còn nghĩa vụ thesis.
5. File cũ còn cần: formal evaluator/notebook/result, benchmark/gold, compatibility paths.
6. Module refactor: không; đây là cleanup behavior-preserving.
7. Duplicate implementation: không có duplicate production được chứng minh.
8. Dependency bỏ: 0.
9. Coupling: ActionDecision/evaluation debt đã được các branch nghiên cứu ghi nhận nhưng không
   có thay đổi an toàn mới để tích hợp trong task này.
10. Phase 1 ảnh hưởng: không.
11. Phase 2 ảnh hưởng: không.
12. Public API thay đổi: không.
13. KB identity thay đổi: không.
14. Local branch kiểm tra: 20 branch có trước audit ngoài `main`.
15. Branch đã merged/superseded: ghi trong bảng disposition.
16. Valuable commits: các kết luận reranker/retry/evidence-path đã nằm ở main dưới PR #110,
   #115, #116; không còn production commit chưa tích hợp được chứng minh.
17. Branch tích hợp trong cleanup: 0.
18. Failed experiments: evidence selection/decomposition/requested-information, local evaluators,
   RAGAS shadow và retrieval-first.
19. Branch có thể archival/delete sau approval: các branch SUPERSEDED/ALREADY_MERGED/FAILED,
   nhưng không force-delete trong task này.
20. Owner review: extraction contract, focused adjudication, stashes, benchmark policy labels.
21. File giảm ròng dự kiến: 25 (27 xóa, 2 artifact audit thêm).
22. Dependency giảm: 0.
23. Test trước/sau: baseline và post-cleanup được ghi đầy đủ ở mục Verification; container sạch
   đạt 1.076 test, frontend đạt 52 test.
24. CI: baseline success; PR #119 chỉ được merge sau khi hai required jobs success trên final head.
25. Backup: bundle verify pass, checksum ghi ở mục Restore point.
26. Maintenance commits: xem Git log của branch/PR.
27. PR URL: <https://github.com/MinhHoang1403/Acne_Advisor_AI/pull/119>.
28. Final main SHA: ghi trong final handoff sau merge.
29. Git status: phải sạch sau sync cuối.
30. Technical debt: liệt kê trong mục Technical debt.
