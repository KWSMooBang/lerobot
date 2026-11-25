# VLA-shortcut-exps — RESEARCH

설계 원문: "LIBERO Shortcut Learning 관측 실험 설계" (https://claude.ai/artifact/7jaV1qVgL53Dx2ZsjaSbr9, 2026-10-25).
아래는 실행용 요약이며, 조건·지표의 세부 정의는 원문 1–9장을 따른다. 결과는 같은 문서의 모델별 탭에 정리한다
(개요 · π0.5 LIBERO · π0.5 base · OpenVLA-OFT(보류), rollout 영상 위주).
문서 id 3688b476-b3d8-4faa-9cbe-648f12a5d2a2 · 탭 body: 개요 148053fd-b595, π0.5 LIBERO a53f724e-8515,
π0.5 base 7c071bcc-f214, OpenVLA-OFT(보류) 9f0d56b1-1013. 영상은 문서 artifact 에 asset 업로드 → blob → 탭에 삽입 (mp4 + 3MB 이하 gif).

## 브랜치 구조
- `main`: upstream lerobot fork 기준선. 연구 코드는 넣지 않는다.
- `shortcut-learning-exps`: 연구 통합 브랜치 (main d3793b7e 에서 분기, 2026-10-25). L2 는 여기서 worktree 를 따고
  `exp/<slug>` 로 push, codex 리뷰 후 `shortcut-learning-exps` 로 병합한다.
- 재사용: `origin/exp/pid-demo` 의 `scripts/pid_demo/{common,setup_libero}.py` (π0.5 LIBERO 로딩·환경 설정).

## 목표
LIBERO 로 학습한 VLA 가 instruction 이 아니라 장면 레이아웃에서 태스크를 추론하는지(shortcut) 정량·정성으로 확인한다.
"새 태스크를 못 함"(SR_new 낮음)과 "원래 태스크를 고집함"(Shortcut Rate 높음)을 분리해 측정한다.

## 가설 (기준값은 제안, C0 재현 후 확정)
| ID | 내용 | 조건 | 지지 기준 |
|---|---|---|---|
| H1 언어 무시 | Spatial/Object 는 instruction 없이도 수행 | C1, C2 | C0 대비 하락 ≤ 10%p |
| H2 원래 태스크 고집 | 지시를 바꿔도 원래 태스크 수행 | C4, C5 | Shortcut Rate > SR_new 이고 ≥ 50% |
| H3 suite 의존성 | shortcut 은 레이아웃-태스크 1:1 suite 에서만 | C1, C6 × 전 suite | Goal 만 C1 에서 ≥ 50%p 하락 |
| H4 위치 암기 | 물체 의미가 아니라 위치를 기억 | C7, C8 | 첫 접근 지점이 원래 위치인 비율 ≥ 50% |

## 모델
| 모델 | 체크포인트 | 학습 방식 | 실행 환경 |
|---|---|---|---|
| π0.5 LIBERO | lerobot 공식 π0.5 LIBERO finetune | 4 suite 혼합 (L2 가 확인·기록) | 이 repo (lerobot) |
| π0.5 base | lerobot/pi05_base | LIBERO 미학습 — shortcut 이 LIBERO 학습에서 생기는지 보는 대조 | 이 repo, LIBERO stats 로 전처리 |
| (보류) π0, OpenVLA-OFT | — | — | π0.5 결과 후 결정 |

## Testbed (2026-10-28 사람 지시)
임의 모델(lerobot 내장·직접 만든 플러그인)을 LIBERO 학습 데이터로 학습하고, 우리 instruction 변환 + LIBERO-Plus + LIBERO-PRO 로
shortcut robustness 를 바로 평가하는 환경. 학습·대규모 평가 = 4×A100 서버, 검증 = GPU 0. 설계: /outputs/VLA-shortcut-exps/tasks/TESTBED_DESIGN.md, 문서 "Shortcut Testbed 설계" https://claude.ai/artifact/N5RnamMWSYwxoQ54RsEfXF.
작업: T009 평가 env 수정 (병합) · T010 policy server + battery + report · T011 학습 harness + 플러그인 템플릿 → T012 LIBERO-Plus · T013 LIBERO-PRO adapter.

## 단계와 관문
- P0 코드 (GPU 거의 없음): T001 평가 harness (override·이중 goal·로깅·영상), T002 instruction 세트·swap BDDL·feasibility
- P1 파일럿: π0.5 × Spatial·Goal × {C0, C1, C4, C5}, 태스크당 10 ep → harness 검증, 시간 실측
  - 관문 1: C0 성공률이 보고값 ±3%p (π0.5 LeRobot 보고: Spatial 98.8, Object 98.2, Goal 98.0, 10 92.4 근방), ep 당 시간 실측
- P2 본 평가: {π0.5 LIBERO, π0.5 base} × 4 suite × C0–C6, 태스크당 50 ep (C0 3회 반복)
- P3 시각 조건 C7–C8 (T003, P1 이후), OpenVLA-OFT 추가
- P4 분석: 지표·Wilson CI·McNemar, 대표 rollout 영상 (C0 vs C4/C5 같은 초기 상태 나란히) → 문서 모델별 탭
- C9 (vision-only 학습) 는 선택, 보류

## GPU 예산
- 2026-11-23 사람 지시: zest kubectl 이 읽기 전용이 되어, T022 basket far 실험은 VGI-102(이 컨테이너 호스트) GPU 0 에서 실행 (10/30 'VGI-102 GPU 금지' 의 예외, ≈1.3 GPU-h).
- 2026-11-23 사람 지시: VGI-102 GPU 1 에서 T023 (pi05lib × LIBERO-PRO 전체 3,400 ep + activation 기록) 실행, 예상 ≈11–14 GPU-h, 16 GPU-h 넘으면 멈춤. GPU 0 은 T022.
- 2026-11-25 사람 지시: VGI-102 GPU 0 에서 T028 (기존 에피소드 rich 재추출: 벡터·token 축·denoise 경로·잡음 16개 probe) 진행, 약 5 GPU-h, 6 GPU-h 넘으면 멈춤.
- 2026-11-25 T028 디스크: spec 그대로 105 GB > 60 → 사람 결정 A안 (≤80 GB, 무손실 압축) → 실측 ≈90 GB (층 17 massive activation 때문에 fp32) → 사람 결정 95 GB 까지 승인. GPU 예상 ≈2 GPU-h.
- kubectl 서버 사용량: T015 smoke 2.0 GPU-h (2026-10-31), VLM SAE ≈14 GPU-h 승인 → 실제 합계 ≈11.2 GPU-h (smoke 포함).
- **2026-10-30 사람 지시: GPU 를 쓰는 일은 smoke 포함 모두 kubectl 서버에서 (baccus-VLA-shortcut-exps 세션에 넘김). 이 컨테이너 GPU 0 은 쓰지 않는다.** L2 명세는 CPU 테스트 + 서버용 smoke 스크립트로.
- 2026-10-25 사람 지시: 당분간 GPU 0번만 사용. launch.sh 는 GPU 를 고를 수 없으므로 run 은 CUDA_VISIBLE_DEVICES=0 에서 순서대로 직접 실행
  (출력은 /outputs/VLA-shortcut-exps/runs/<run_id>/ 동일). 동시 run 1개 → 벽시계 시간 = GPU-h.
- P2 범위 결정 대기 (2026-10-26, 사람에게 질문). T001 smoke 실측 (RTX 4090, GPU 0): ≈45 ms/step, 모델 로드 60–90 s/run.
  ep 당: pi05lib C0 5.5 s (조기 종료), 300 step 조건 ≈14 s, pi05base 는 항상 max step (≈13–14 s, l10 은 ≈24 s).
  → P1 파일럿 (Spatial·Goal × C0/C1/C4/C5 × 10 task × 10 ep, 두 모델) ≈ 9–10 GPU-h.
  → P2 전체 (4 suite × 17 조건-변형 × 50 ep) 는 pi05lib ≈ 80 h, pi05base ≈ 130 h → GPU 1개로는 축소 필요 (P1 후 결정).

## 백로그
- **CPU 아키텍처 재현성 (2026-11-20 baccus)**: ferrari (EPYC 7763 Zen3) 와 이 컨테이너 호스트 VGI-102 (EPYC 9554 Zen4, AVX-512) 사이에서 기본 OpenBLAS 커널이면 sim 이 갈라짐 (기록 action replay 6개 중 2개 eef 0.6–1.4 mm 차이, patch init_states 바이트 다름). OPENBLAS_CORETYPE=ZEN 이면 바이트 동일·replay 차이 0. 두 머신 결과를 비트 단위로 맞추려면 이 env 를 설정 (setup·실행 스크립트 기본값 후보). 렌더는 GPU 간 픽셀 1 레벨 차이 소수.
- ferrari 실험 환경 사본: /workspace/VLA-shortcut-exps-kubectl (cdd3aec6 clone + lerobot-l90 worktree, 자체 .venv; 이 repo 와 별개). GPU 스크립트는 GPUS=<ids> 필요.
- ~~common.resolve_path 의 libero import 버그~~ → T019 에서 다시 고침 (2026-11-10).: 평가 계획 (ours/adapter plan → run_eval.plan_variants → common.variants_for → resolve_path) 이 ensure_libero_config 전에 `from libero.libero import get_libero_path` 를 해서 ~/.libero/config.yaml 이 없으면 input() 에서 멈추고, 있으면 다른 venv 경로를 씀 (이 컨테이너의 ~/.libero 는 RACA venv; BDDL 130개 동일이라 과거 결과 영향 없음). 수정 (find_spec 로 bddl_files 찾기) 은 삭제된 exp/sc-starvla 에 있었음. 우회: LIBERO_CONFIG_PATH 를 fix config 로.
- **SAE 분석 (T015 뒤)**: 참고 arXiv 2603.19183 (π0.5 PG5 위주, mean-pool, TopK+AuxK ER=1). 후보: feature 별 지시/장면 민감도 (C0 vs C4/C5/C1 같은 init), shortcut feature ablation·steering 으로 SR_new 변화, 패치 단위 위치 feature (H4), 층별 지시문 정보 소실, base vs finetune diffing, generality 비율 ↔ testbed shortcut 점수 상관 (T011 변형), 대조군 (norm·빈도 맞춘 무작위 방향 ablation, splice-in 성공률).
- **T014 Object 조건 반영 (T009 뒤)**: 문서 "Shortcut Testbed 설계" > 테스트 예시 > "Object 에서의 C0–C8" (다른 세션 작성, 2026-10-28). C4-a·b·c 는 instruction 세트와 같음. 바꿀 것: C6 = 장면에 없고 반대쪽 대상 자리에서 학습한 물체의 문장 (지금 cross_task 는 다음 태스크 = C4 와 같은 문장), C7 = 두 대상 자리 물체 맞바꿈, C8 = 대상 제거 (init state free joint 편집), 물체별 In(basket) + 첫 파지로 장면/문장 자리/지시/그 밖 4분류. Object quick = C0·C1·C4 (+C7).
- **LIBERO-Plus·PRO 도 MuJoCo 3.8.1 + 자체 config → T009 수정 미적용, 같은 렌더·튕김 문제 가능성** (T009 병합 후 확인).
- ~~평가 환경 ≠ 학습 데이터~~ → T009 병합으로 해결 (아래 결정 로그). Object·LIBERO-10 평가 가능. 이후 run 은 run_meta.libero_env 로 raw/fix 구분: LIBERO-10 t0·t1·t7 (LIVING_ROOM_SCENE1/2) 은 평가 렌더에서 milk·orange juice·ketchup 이 테이블 위에 없음 (sim z≈0.9–1.05 m), Object 는 바닥 텍스처·조명이 다름. Spatial·Goal 은 일치. 원인(자산·BDDL 버전) 조사 후 Object·LIBERO-10 평가.
- paraphrase 6개 사람 검수 필요 (spatial 2, goal 2, l10 2; instructions/<suite>.json 의 review 필드). C3 실행 전에.
- libero_10 t9 C4-a feasible=false 는 스크립트 정책 한계 (C0 도 실패). 포함하려면 teleop 확인 필요.
- submit.sh (T003) codex 리뷰 3건, 파일럿 후 수정: driver 가 죽으면 run 이 running 으로 남음(check-runs 에선 stale 로 보임),
  같은 run 동시 제출 경합, --hours 검증 없음. **파일럿 중에는 .claude/worktrees/sc-pilot 의 submit.sh 를 고치거나 worktree 를 지우지 말 것**
  (driver bash 가 그 파일을 읽으며 실행 중).
- T004 batched 경로: npz·mp4 교체가 서로 독립이라 이론상 두 시도의 파일이 섞일 수 있음 (commit 된 npz 가 깨져 있어야 생기므로 사실상 불가, codex 2차). attempt 별 경로로 바꿀 때 같이.
- metrics.py C0 repeat 키에 base seed 가 없음 (codex 2차 리뷰). 지금은 seed 7 하나라 무해. 여러 seed 를 쓰기 전에 고칠 것.
- π0, OpenVLA-OFT: π0.5 결과 후 결정 (OFT 는 별도 환경 필요)
- 자체 학습 모델(suite별 학습) 포함 여부 (설계 3장 열린 질문)
- C9 vision-only 학습

## 결정 로그
- 2026-10-25 설계 문서 채택, 통합 브랜치 `shortcut-learning-exps` 생성. 1차 모델은 lerobot 내장 π0.5·π0.
- 2026-10-25 사람 지시: 1차 모델을 π0.5 base vs π0.5 LIBERO finetuned 로 변경. π0·OpenVLA-OFT 는 보류. GPU 예산은 파일럿 실측 후 확정.
- 2026-10-25 T001 harness 병합 (exp/sc-harness 899e6a40). smoke: pi05lib goal C0 4/4, C4 새 지시 3/4·원래 goal 0/4, pi05base 전부 0/4.
- 2026-10-25 사람 지시: A100 80GB × 4 서버용 고속 실행 경로를 코드만 구현 (T004, 실행 안 함). 결과는 exec_mode·GPU 별로 구분해 단일 GPU 결과와 짝짓지 않는다.
- 2026-10-25 사람 지시: 파일럿이 끝나면 결과 문서에 영상 위주로 정리. T005(analysis) 가 영상·manifest 를 만들고 lead 가 문서에 올린다.
- 2026-10-25 T002 병합 (exp/sc-instructions 4d97228c). swap 102개 중 101 feasible. codex 리뷰: goal t3 swap 의 Open 누락 지적은 원본 goal 도 In 만 쓰므로 유지.
- 2026-10-25 T004 병합 (exp/sc-multigpu 1b7a67f1, CPU 테스트 25/25). P2 추정: 46,100 ep, 단일 경로 ≈162 GPU-h, 4×A100 ≈6 h (범위 3–11 h, 미측정 β 가정).
- 2026-10-26 P1 완료 (16 run, 7.04 GPU-h). pi05lib: Spatial 은 지시를 바꿔도 원래 goal (C5 94.0% vs 새 0.5%, C4 61.1% vs 30.0%), Goal 은 언어 사용 (C1 13.0%, C4 새 40.0%). pi05base 전 조건 0% → 대조군 역할 못 함.
- 2026-10-26 T005 결과 문서 반영 (개요·π0.5 LIBERO·π0.5 base 탭, 영상 18개, SR 차트). T006 GPU 확인 (4090): batched N=4 가 single 대비 3.4× (70 vs 20.8 step/s), β=0.27. single vs batched 일치: success_orig 24/24, grasp 24/24, C4 success_new 13/16 (bf16 배치로 step 0 부터 action 차이 ≈1e-3, C4 에서 궤적이 갈라짐) → batched 결과는 single 과 절대 짝짓지 않는다 (metrics 기본 차단). P2 추정: single ≈173 GPU-h, 4×A100 P=2 N=8–16 ≈5–6 h (3–7 h, A100 값 미측정).
- 2026-10-26 사람 지시: pi05lib 의 지시 미이행 케이스가 학습 instruction (완전·부분 일치, 같은/다른 장면) 과 관련 있는지 분석 → T007 (analysis, CPU).
- 2026-10-26 T007 결과: 학습 데이터 = HuggingFaceVLA/libero 40 문장/1,693 ep (LIBERO-90 없음). swap 60개 중 완전 일치 10 (전부 Spatial C4-a, 다른 장면), 같은 장면 0. SR_new: Spatial L2 56.0% / L4 3.3% / L5 0.4%, Goal L4 34.7%. 처음 보는 구성요소(특히 목적지)는 거의 항상 실패, 그러나 학습 문장 그대로도 Spatial 4/10 은 원래 bowl ("next to …" 3개 모두 실패) → 미이행은 학습 노출 부족만으로 설명 안 됨. 문서 π0.5 LIBERO 탭에 섹션·차트(widget 4e7b2319-b42d) 추가.
- 2026-10-26 사람 지시: LIBERO 학습 데이터(HuggingFaceVLA/libero) 를 빠짐없이 시각화 → T008 (analysis). 문서에 "학습 데이터" 탭 + suite 별 하위 탭으로 정리 예정.
- 2026-10-26 T008 학습 데이터 시각화 완료 (HuggingFaceVLA/libero rev 86958911, 34.9 GB → /outputs/VLA-shortcut-exps/data/hf_libero/). 첫 프레임 → task 정확도 (DINOv2+logreg): Spatial 90.0 / Object 100 / Goal 8.9 / LIBERO-10 89.7% (배치 상한 91.9/100/11.7/91.3). Goal 도 진행 10% 에 71%. Spatial t7·t9 는 같은 물리 배치. 문서 '학습 데이터' 탭 + suite 하위 탭 4개에 197 그림 반영.
- 2026-10-28 사람 결정: testbed 구축. 학습 GPU = 4×A100 서버 + GPU 0, 벤치마크 1차 = LIBERO-Plus·PRO (X·MAX 보류), 모델 = lerobot 내장 + 플러그인 템플릿 (외부 모델 보류). T009–T011 동시 시작.
- 2026-10-28 T011 병합 (exp/sc-train ce58993e). preset: sc_template·smolvla·pi05·pi0, effective batch 고정 (pi05 256, 6k update), 변형 instr=blank|paraphrase (frame 별 결정적)·suites=, --variant-eval same|none. ckpt_registry.tsv = /outputs/VLA-shortcut-exps/ckpt_registry.tsv. W&B 키 거부 → offline.
- 2026-10-28 T010 관찰: 기존 in-process 경로도 긴 C4 에피소드에서 action 이 bit 단위로 재현되지 않음 (같은 설정 반복 8/12 bit 동일, 결과 수준은 일치). 짝 비교는 결과 수준(성공·파지)으로만 해석.
- 2026-10-28 T010 병합 (exp/sc-bench-core 458606fd). 프로토콜 sc-policy/1, server↔in-process 결과 24/24 일치. T012 LIBERO-Plus·T013 LIBERO-PRO 시작.
- 2026-10-28 다른 세션이 테스트 예시 탭에 Object 설계 추가 → lead 가 일관성 정리 (C4-c 추가, C6·C7·C8·판정은 미구현 표시, 한눈에 표·설계 탭 반영, T014 백로그).
- 2026-10-28 T012·T013 [DONE] (수정 중). Plus: 모든 변형이 원래 goal 유지 → Plus 는 겉모습 강건성만 잰다. PRO: task 축 π0.5 새 goal 0/8, 원래 goal 5/8; 단 PRO 지시문·init 혼선 확인 중.
- 2026-10-28 PRO 혼선 정리 (T013): PRO :language 는 40 중 15 만 학습 문장과 같음, 모든 축이 init 재추출 → testbed 기본 = 비언어 축은 학습 문장, task·semantic 축은 clean init. 재검증 task 축 새 goal 0/8, 원래 goal 6/8; position swap 4/8. 문서 테스트 예시 탭 PRO 섹션·한눈에 표 반영 (PRO task 축은 대상 물체를 주로 바꾸고 목적지는 Goal t2·L10 t4 만).
- 2026-10-28 T012 조명 축: fork 조명 변형의 약 절반이 같은 '불 꺼진' 화면 (Spatial t0 23/42, Goal t8 11/26, 레벨 무관; 렌더 경로 정상). 조명 결과는 레벨이 아니라 실제 화면별로 묶어 해석.
- 2026-10-28 T012 병합 (e086ef9f), T013 병합 (a621c4c2). testbed = ours + LIBERO-Plus + LIBERO-PRO. 벤치마크 venv 는 아직 L2 worktree 안에 있음 (Plus 12 GB, PRO 2.4 GB) → quick 을 돌리는 checkout 에서 setup 필요.
- 2026-10-29 T009 [DONE]: 원인은 MuJoCo 버전 (lerobot 3.8.1 vs 학습 데이터 ≤3.3.2; ≥3.3.3 sRGB → Object 바닥·salad dressing 어두움, ≥3.4 box-box 침투 → L10 t0·t1·t7 물체 튕김). 자산·init mirror 로 40 task 일치 27→40/40, Spatial·Goal 바이트 동일 (P1 비교 유지). P1 은 raw env 결과. codex 수정 요청 중.
- 2026-10-29 T009 병합 (7bc7a64f, exp/sc-env-fix fbfb6553; T008 시각화 코드 함께 병합). 기본 = fix env, SC_LIBERO_RAW=1 = P1 env. run_meta·resume identity·ours bench_version 에 libero_env 기록 (없으면 raw), raw↔fix resume 양방향 거부. codex 리뷰 3회 (blocking 3건 수정). mypy 는 기존 파일(p1_videos·train_overlap·train_data_*) 에러로 실패 상태.
- 2026-10-30 사람 지시: π0.5 모든 층 SAE. 이 컨테이너에서는 구현 + GPU 0 smoke 만 (T015, exp/sc-sae), 병합 뒤 전체 activation 수집·SAE 학습은 baccus-VLA-shortcut-exps 세션이 kubectl 서버에서. 기본: PaliGemma 0–17 + embedding (카메라·텍스트 그룹별 평균 저장) + action expert 0–17 (denoising step 별), 논문 B.1 설정.
- 2026-10-30 T015 병합 (exp/sc-sae 12ad3e6a). π0.5 37층 hook (PG emb+0–17 그룹별 평균, AE 0–17 step 별), TopK+AuxK SAE (논문 B.1), pod 분할·resume·서버 smoke.sh. codex 리뷰 4회 (lease 경합은 중복 작업만 생겨 수용). CPU 테스트 38, GPU 미사용.
- 2026-10-31 사람 지시: 용량 때문에 VLM (emb + PG 0–17, ≈64 GB) SAE 부터. AE 는 그 뒤. 서버 용량이 부족하면 끝난 산출물을 로컬로 받고 서버에서 지운 뒤 계속. 실행은 baccus-VLA-shortcut-exps (kubectl).
- 2026-10-31 서버 smoke PASS (exp/sc-sae 5df0e208 병합, A100×1): 수집 실측 17 ts/s. VLM 3분할로 ≈14 GPU-h — 사람 승인. a 배치 23:15 UTC 시작. VLM 만 요청하면 denoising 을 건너뛰는 옵션 (T015 후속, CPU) 을 b·c 에 적용 예정.
- 2026-10-31 T015 prefix-only 수집 병합 (exp/sc-sae 97c6901c): VLM 층만 요청하면 가장 깊은 PG 층 뒤에서 중단, activation bit 동일. A100 실측 full 22.3 → prefix-only pg11 까지 43.5, pg17 까지 32.2 ts/s. VLM a·b 완료 (VE pg00 0.972 → pg11 0.923), c 진행 중.
- 2026-10-31 T015 VLM SAE 완료 (19층, ≈11.2 GPU-h). emb·pg00–16 keep, pg17 보류 (dead 27.6%, action expert 가 읽지 않는 마지막 residual 이라는 가설). 데이터는 Mac mini (baccus). 다음: pg17 유효 rank 확인, SAE 분석 (지시/장면 민감도 등) 명세, action expert SAE 는 용량 확보 뒤.
- 2026-11-01 사람 지시: kubectl 서버의 shortcut-exps 데이터 전부 Mac mini (baccus server-backup/) 로 백업 후 삭제, pod 삭제. 서버에는 모델 캐시 (hf_cache/hub 7.4G) 만. 다음 서버 작업은 환경·데이터셋·Arrow 캐시 재구축 (~75G) 부터.
- 2026-11-09 사람 지시: 진행 중 실험 모두 중지, T016–T018 (StarVLA·π0.5-DROID zero-shot·언어 probe) 결과·로그·브랜치·서버 설치물 전부 삭제. 새 방향: LIBERO-Goal 에서 가능하지만 학습에 없는 새 task instruction 성능 측정 — 후보 초안 tasks/refs/goal-novel-candidates.md (N1 재조합 11 / N2 새 영역 3 / N3 새 동사 5 / N4 조합 4).
- 2026-11-10 사람 목표 확정: 정책 학습 → 같은 체크포인트로 {LIBERO 원본, 직접 설계한 LIBERO 변형, LIBERO-Plus 원하는 perturbation, LIBERO-PRO 원하는 perturbation} 중 골라 평가하는 환경. T019 (선택·묶어 실행·학습→평가 연결·libero 버그·Plus/PRO 렌더 확인·서버 setup·e2e) 와 T020 (변형 설계 도구: YAML → BDDL·init·실행 가능성·미리보기, 다중 목표 추적; 첫 예시 Goal 새 instruction 23 + Object C6–C8) 동시 시작. 경계: adapter catalog() 인터페이스, 병합 T019 → T020.
- 2026-11-10 T019 병합 (exp/sc-testbed-select ac6d3fd7): sc-bench --list / battery 검증 / custom_template, sc-train --then-bench, Plus·PRO render_fix (Object·L10 학습 렌더 일치; bench_version 에 fix 태그, 파일 sha 로 검증), setup_server --with-plus/--with-pro, e2e_smoke. codex 2회. exp/sc-sae 8652608a 도 병합.
- 2026-11-10 T020 병합 (exp/sc-variant-designer-v2 e2cdae1f): 변형 설계 도구 (YAML → BDDL·init → 실행 가능성 → 미리보기), condition custom + 다중 목표 추적·fallback 표 (C0 짝은 source_init_idx), catalog·validate 연결. 예시 103개 (goal15 30 + control 20 + Goal 후보 n01–n23 + Object C6–C8; feasible 78 / infeasible 5 / n/a 20), 갤러리 analysis/variants/gallery.html. TurnOff 는 not TurnOn 으로 판정. codex 3회. 서버 goal15_trainfmt 실행은 이전 commit cdd3aec6 (exp/sc-variant-designer) 로 진행 중 — 그 브랜치는 실행이 끝난 뒤 정리.
- 2026-11-23 사람 지시: basket near(+10 cm)는 위치 변화가 작음 → 바구니를 카메라 쪽으로 20·30 cm 당긴 조건 추가 (T022, baccus 서버, ≈1 GPU-h). reach 확인 후 거리 확정, 이후 거리별 내부 표현 분석.
- 2026-11-23 T022 거리 확정: +x 만 당기면 원근 때문에 바구니가 정책 화면 왼쪽 끝에 잘림 → 화면 오른쪽(world −y) 으로 함께 이동. 사람 승인 far25 = (0.25, 0.20) (원래 (0, 0.26) 에서 25.7 cm, 10 task 화면 안 왼쪽 여유 13 px, 50 init 안정, 물체 최소 거리 5.3 cm). far20 = 같은 방향 선 위 (0.20, 0.21) (20.6 cm, lead 결정, L2 확인 중). 실행 VGI-102 GPU 0, ctl·far20·far25 각 100 ep.
- 2026-11-23 사람 지시: GPU 1 에서 pi05lib 로 LIBERO-PRO·Plus 성능 + activation trajectory 분석, 참고 논문 Churchland 2012 (jPCA 회전 동역학), Gallego 2018 (조건 간 manifold 보존·CCA 정렬), Inagaki 2019 (이산 attractor). 범위는 '일단 LIBERO-PRO 만 전부' (4 suite × 6 교란, Object position 11 level, 340 항목 × init 10). T023: 정책 서버에 activation 기록 옵션 (action 비트 일치 확인) → 본 평가 → 성능·shortcut 비율 + 세 논문 방식 분석. 질문: 교란 실패 때 내부 상태가 학습한 task 궤적으로 끌려 들어가는가 (이산 attractor = '지시문은 스위치' 결론의 내부 근거).
- 2026-11-23 궤적 시각화 확장 (CPU): T024 basket 궤적에 시간·잡기/놓기 표시 + 원래 Object 고정 공간·40 task 공간 투영 (바구니 이동은 같은 공간 안에서 다른 자리, 다른 학습 task 로 바꿔치기 아님), T025 네 suite 같은 방식 (Goal 후보가 배운 task 로 빠질 때 잡기 전부터 그 task 궤적 위, patch_attention 0.79).
- 2026-11-23 사람 지시: patch attention 궤적이 task 구분에 가장 좋아 보임 → composition 성공 후보의 특징 추출 (예: 구성 task 궤적이 교차하는가). T026: cand31 29 후보 × 사전 특징 (구성 task 최근접 거리·교차 길이·시작 겹침·목적지 배타성·기본 task 끌림·attention 지도 겹침) vs 새 task 성공률, 특징 6종 비교, 후보 안 성공/실패 비교. n 작음 → 탐색적, A1 정의 사전 고정.
- 2026-11-24 T026 결과 (탐색적, 성공 후보 6/29): 구성 task 궤적 "교차" (A1) 는 성공과 무관 (patch ρ +0.20, 교차 0). 성공 = 두 원천 task 가 잡기 전부터 붙어 감 (A2_pre patch ρ −0.64 p=0.001, expert −0.52) + 목적지가 공유됨 (A3) + 성공 trial 이 잡은 뒤 물체 원천 → 목적지 원천으로 갈아탐 (90% vs 27%). 해석: 다가가는 구간을 공유하는 "Y 자" 조합만 이어짐. 확인하려면 A2_pre·A3 로 새 후보를 미리 골라 평가 (다음 후보).
- 2026-11-24 사람 지시: 궤적에서 더 의미 있는 해석 → T027 (CPU): (A) RSA — 특징 RDM vs 의미 (동사·물체·목적지·공간 수식·단어)·동작 (eef·gripper·action)·장면 RDM, 부분 상관, (B) 의미·동작이 갈리는 짝 (basket 같은 task 반대 동작, L90 lscene·cand31 대체 = 다른 지시 같은 동작) 에서 특징별 β, (C) 시간별 선형 디코딩·혼동 행렬·학습→대체 trial 일반화 (지시 vs 실제 동작). 주의: kubectl 특징은 크기 요약이라 gemma 의미 검정이 약함 → stage 2 로 PRO (실제 벡터) 반복. 선택지: 기존 실행 벡터 재추출 (GPU ≈3 GPU-h, 사람 결정).
- 2026-11-24 T027 stage 1 결과 (909621e3): 지시 의미를 따르는 것은 gemma 문장 token 뿐 (L90 lscene β 의미 0.49 / 동작 0.22), expert·action·patch_attention 은 실제 동작·장면 (β 동작 0.58–0.68). basket 같은 지시·다른 동작: expert 잡은 뒤 1.0→1.87, 문장 token ≈1. 40 task 분류 gemma 0.98·patch 0.93·expert 0.87·action 0.41, 그러나 처음 보는 task 의 동사·물체·목적지는 우연 수준 = 부품이 아니라 task 통째 저장 (조합 실패와 일치). L90 대체 trial 을 학습 분류기로: expert 실제 0.70 / 지시 0.02, 첫 2 호출부터. 한계: 크기 요약, L90 실제 동작 = 장면 겹침 → PRO stage 2.
- 2026-11-25 사람 지시: 신경과학 논문 (Churchland 2012, Gallego 2018, Inagaki 2019, Elsayed 2016, Kaufman 2014, Ames 2014, Kim 2024) 의 집단 활동 분석을 VLA 에 제대로 적용하려면 크기 요약이 아니라 벡터 (방향+크기) 가 필요 → T028 (rich 재추출, GPU 0) + T029 (분석, CPU) 병행. 핵심 가설: 동작 담당은 이해 담당을 층마다 W_K·W_V (2048→256) 로만 읽으므로, 지시문 차이가 그 read-in null space 에 있으면 "이해 담당은 구별, 동작 담당은 무시" 가 구조로 설명됨 (Kaufman). 그 밖에 token 축 (언어 단어·action 50 = 계획된 미래) 궤적, 준비/움직임 부분공간 (Elsayed), jPCA (token·denoise·호출 축), manifold (Gallego), 잡음 16개 probe 로 이산 attractor (Inagaki), 조합 코드 (Kim). 중간 지시 바꾸기 (Ames) 는 설계만.
- 2026-11-25 T029 stage A (PRO 벡터, Kaufman): "지시 차이가 동작 담당 read-in null space 에 있다" 는 새 목표 지시에서 기각 (potent 비율 0.30 > 무작위 0.25). 바꿔 쓰기 (같은 뜻) 만 null 쪽. 탐색 결과: 대체 에피소드에서도 VLM 은 지시 차이를 읽을 수 있는 공간으로 보내는데, 동작 담당의 상태 변화가 작고 action 출력에 안 닿는 방향 (null) → 병목은 VLM 이 아니라 동작 담당의 변환. stage B 에서 사전 등록해 재검정.
- 2026-11-25 T029 결과 (99bfda3c, 벡터 1,560 ep + PRO 3,390 ep): 운동피질 그림 중 π0.5 에서 보인 것은 Gallego (바구니 20·25 cm = 같은 동역학·다른 읽기, CCA 0.88/0.82; 반대편 = 다른 동역학 0.73) 와 Inagaki 부분 (학습 지시는 두 끝점 85–91%, 새 지시는 39–47% 끝점·20–24% 사이) 뿐. 회전 동역학 (jPCA R² 0.01–0.02)·준비/움직임 직교 (Elsayed, 무작위 수준) 없음. 지시 정보는 동작 담당 read-in 까지 도달 (Kaufman, null 에 숨지 않음; 같은 뜻 바꿔쓰기만 덜 읽힘). stage A 의 "동작 담당 변환 병목" 은 Goal 후보 사전등록 재검정에서 재현 안 됨 (후보와 혼재). Goal 후보 상태 = 원천 task 조합 (R² 0.48, 무작위 대비 +0.29). 잡음 16개는 거의 같은 곳 → 모드는 관찰·지시가 정함. 다음 후보: 내부 상태 직접 교란 (KV 바꿔 끼우기·방향 주입) 과 Ames 중간 지시 바꾸기 (1,080 ep ≈3.6 GPU-h, 설계 완료).
- 2026-11-25 사람 지시: 층별 activation 궤적을 신경 집단 벡터 방식으로 → T030 (CPU, T028 데이터만): 층마다 채널 벡터 (이해 담당 구간별 2048, 동작 담당 1024), 동작 담당 MLP 유닛 4096 (뉴런에 가장 가까움), 토큰 × 채널 (action 50, 언어) 로 궤적·유효 차원·층 × 호출 디코딩 (지시 vs 실제 동작)·β·갈라짐·Kaufman 18층·Gallego/Kim 층별·크기 요약과 비교. massive activation 채널 처리 사전 등록. 2단계 (이미지 포함 모든 토큰 크기 지도, VLM MLP 유닛, A·V, ≈1 GPU-h) 는 보류.
