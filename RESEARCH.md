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
- 미정 — 파일럿 실측 후 확정. T001 smoke 실측 (RTX 4090, GPU 0): ≈45 ms/step, 모델 로드 60–90 s/run.
  ep 당: pi05lib C0 5.5 s (조기 종료), 300 step 조건 ≈14 s, pi05base 는 항상 max step (≈13–14 s, l10 은 ≈24 s).
  → P1 파일럿 (Spatial·Goal × C0/C1/C4/C5 × 10 task × 10 ep, 두 모델) ≈ 9–10 GPU-h.
  → P2 전체 (4 suite × 17 조건-변형 × 50 ep) 는 pi05lib ≈ 80 h, pi05base ≈ 130 h → GPU 1개로는 축소 필요 (P1 후 결정).

## 백로그
- π0, OpenVLA-OFT: π0.5 결과 후 결정 (OFT 는 별도 환경 필요)
- 자체 학습 모델(suite별 학습) 포함 여부 (설계 3장 열린 질문)
- C9 vision-only 학습

## 결정 로그
- 2026-10-25 설계 문서 채택, 통합 브랜치 `shortcut-learning-exps` 생성. 1차 모델은 lerobot 내장 π0.5·π0.
- 2026-10-25 사람 지시: 1차 모델을 π0.5 base vs π0.5 LIBERO finetuned 로 변경. π0·OpenVLA-OFT 는 보류. GPU 예산은 파일럿 실측 후 확정.
