# VLA-shortcut-exps — EXPERIMENTS

run 별 결과와 keep/discard. 숫자는 /outputs/VLA-shortcut-exps/results.tsv 원문 인용.

| 날짜 | task | run_id | 모델 | suite × 조건 | 결과 | 판단 |
|---|---|---|---|---|---|---|
| 2026-10-26 | T003 P1 | sc-p1-pi05lib-goal-C0 | π0.5 LIBERO | Goal × C0 | `n=100;sr_orig=0.970` (보고값 98.0, 관문 1 통과) | keep — 기준선 |
| 2026-10-26 | T003 P1 | sc-p1-pi05lib-goal-C1 | π0.5 LIBERO | Goal × C1 | `n=100;sr_orig=0.130` (C0 대비 −84%p) | keep — Goal 은 언어 필요 (H3 방향 일치) |
| 2026-10-26 | T003 P1 | sc-p1-pi05lib-goal-C4 | π0.5 LIBERO | Goal × C4 | `n=110;sr_orig=0.155;sr_new=0.400` | keep — 새 지시 따름 (chance 10%) |
| 2026-10-26 | T003 P1 | sc-p1-pi05lib-goal-C5 | π0.5 LIBERO | Goal × C5 | `n=100;sr_orig=0.300;sr_new=0.240` | keep — Goal 에서도 target 고집이 따름보다 큼 |
| 2026-10-26 | T003 P1 | sc-p1-pi05lib-spat-C0 | π0.5 LIBERO | Spatial × C0 | `n=100;sr_orig=0.970` (보고값 98.8, 관문 1 통과) | keep — 기준선 |
| 2026-10-26 | T003 P1 | sc-p1-pi05lib-spat-C1 | π0.5 LIBERO | Spatial × C1 | `n=100;sr_orig=0.750` (C0 대비 −22%p) | keep — H1 기준(≤10%p) 미달, Goal 보다 훨씬 작음 |
| 2026-10-26 | T003 P1 | sc-p1-pi05lib-spat-C4 | π0.5 LIBERO | Spatial × C4 | `n=190;sr_orig=0.611;sr_new=0.300` | keep — H2 기준 충족 (Shortcut ≥ 50%, > SR_new) |
| 2026-10-26 | T003 P1 | sc-p1-pi05lib-spat-C5 | π0.5 LIBERO | Spatial × C5 | `n=200;sr_orig=0.940;sr_new=0.005` | keep — 가장 강한 shortcut (H2) |
| 2026-10-26 | T003 P1 | sc-p1-pi05base-goal-C0 | π0.5 base | Goal × C0 | `n=100;sr_orig=0.000` | keep — 정성(영상) 용 |
| 2026-10-26 | T003 P1 | sc-p1-pi05base-goal-C1 | π0.5 base | Goal × C1 | `n=100;sr_orig=0.000` | keep — 정성 |
| 2026-10-26 | T003 P1 | sc-p1-pi05base-goal-C4 | π0.5 base | Goal × C4 | `n=110;sr_orig=0.000;sr_new=0.000` | keep — 정성 |
| 2026-10-26 | T003 P1 | sc-p1-pi05base-goal-C5 | π0.5 base | Goal × C5 | `n=100;sr_orig=0.000;sr_new=0.000` | keep — 정성 |
| 2026-10-26 | T003 P1 | sc-p1-pi05base-spat-C0 | π0.5 base | Spatial × C0 | `n=100;sr_orig=0.000` | keep — 정성 |
| 2026-10-26 | T003 P1 | sc-p1-pi05base-spat-C1 | π0.5 base | Spatial × C1 | `n=100;sr_orig=0.000` | keep — 정성 |
| 2026-10-26 | T003 P1 | sc-p1-pi05base-spat-C4 | π0.5 base | Spatial × C4 | `n=190;sr_orig=0.000;sr_new=0.000` | keep — 정성 |
| 2026-10-26 | T003 P1 | sc-p1-pi05base-spat-C5 | π0.5 base | Spatial × C5 | `n=200;sr_orig=0.000;sr_new=0.000` | keep — 정성 |

**P1 판단 (2026-10-26)**: 16 run 모두 done, 합계 7.04 GPU-h (GPU 0, git 9fd2b5db). pi05lib 관문 1 통과 (Goal 97.0, Spatial 97.0).
Spatial 에서 지시를 바꿔도 원래 태스크를 수행 (C4 Shortcut 61.1%, C5 94.0% vs SR_new 30.0%, 0.5%), Goal 은 언어를 사용 (C1 −84%p, C4 SR_new 40.0%).
pi05base 는 모든 조건 0% — shortcut 수치 비교는 불가, 행동 차이는 영상(T005)으로 본다. next: T005 영상·지표 분석 → 문서 정리, P2 범위 결정.
