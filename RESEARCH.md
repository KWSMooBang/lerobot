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
작업: T009 평가 env 수정 · T010 policy server + battery + report · T011 학습 harness + 플러그인 템플릿 → T012 LIBERO-Plus · T013 LIBERO-PRO adapter.

## 단계와 관문
- P0 코드 (GPU 거의 없음): T001 평가 harness (override·이중 goal·로깅·영상), T002 instruction 세트·swap BDDL·feasibility
- P1 파일럿: π0.5 × Spatial·Goal × {C0, C1, C4, C5}, 태스크당 10 ep → harness 검증, 시간 실측
  - 관문 1: C0 성공률이 보고값 ±3%p (π0.5 LeRobot 보고: Spatial 98.8, Object 98.2, Goal 98.0, 10 92.4 근방), ep 당 시간 실측
- P2 본 평가: {π0.5 LIBERO, π0.5 base} × 4 suite × C0–C6, 태스크당 50 ep (C0 3회 반복)
- P3 시각 조건 C7–C8 (T003, P1 이후), OpenVLA-OFT 추가
- P4 분석: 지표·Wilson CI·McNemar, 대표 rollout 영상 (C0 vs C4/C5 같은 초기 상태 나란히) → 문서 모델별 탭
- C9 (vision-only 학습) 는 선택, 보류

## GPU 예산
- 2026-10-25 사람 지시: 당분간 GPU 0번만 사용. launch.sh 는 GPU 를 고를 수 없으므로 run 은 CUDA_VISIBLE_DEVICES=0 에서 순서대로 직접 실행
  (출력은 /outputs/VLA-shortcut-exps/runs/<run_id>/ 동일). 동시 run 1개 → 벽시계 시간 = GPU-h.
- P2 범위 결정 대기 (2026-10-26, 사람에게 질문). T001 smoke 실측 (RTX 4090, GPU 0): ≈45 ms/step, 모델 로드 60–90 s/run.
  ep 당: pi05lib C0 5.5 s (조기 종료), 300 step 조건 ≈14 s, pi05base 는 항상 max step (≈13–14 s, l10 은 ≈24 s).
  → P1 파일럿 (Spatial·Goal × C0/C1/C4/C5 × 10 task × 10 ep, 두 모델) ≈ 9–10 GPU-h.
  → P2 전체 (4 suite × 17 조건-변형 × 50 ep) 는 pi05lib ≈ 80 h, pi05base ≈ 130 h → GPU 1개로는 축소 필요 (P1 후 결정).

## 백로그
- **T014 Object 조건 반영 (T009 뒤)**: 문서 "Shortcut Testbed 설계" > 테스트 예시 > "Object 에서의 C0–C8" (다른 세션 작성, 2026-10-28). C4-a·b·c 는 instruction 세트와 같음. 바꿀 것: C6 = 장면에 없고 반대쪽 대상 자리에서 학습한 물체의 문장 (지금 cross_task 는 다음 태스크 = C4 와 같은 문장), C7 = 두 대상 자리 물체 맞바꿈, C8 = 대상 제거 (init state free joint 편집), 물체별 In(basket) + 첫 파지로 장면/문장 자리/지시/그 밖 4분류. Object quick = C0·C1·C4 (+C7).
- **평가 환경 ≠ 학습 데이터 (T008, P2 전 필수)**: LIBERO-10 t0·t1·t7 (LIVING_ROOM_SCENE1/2) 은 평가 렌더에서 milk·orange juice·ketchup 이 테이블 위에 없음 (sim z≈0.9–1.05 m), Object 는 바닥 텍스처·조명이 다름. Spatial·Goal 은 일치. 원인(자산·BDDL 버전) 조사 후 Object·LIBERO-10 평가.
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
