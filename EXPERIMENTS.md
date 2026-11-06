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

## T015 π0.5 VLM 층 SAE (kubectl 서버 A100×1, 2026-10-31)
수집: HuggingFaceVLA/libero 전체 (1,693 ep, 273,465 ts), pi05lib, 층별 카메라1·카메라2·텍스트 그룹 평균 (fp16). 디스크 때문에 3배치로 나눠
수집 → 학습 → Mac mini 로 이동 → 서버 삭제. SAE: TopK+AuxK (논문 2603.19183 B.1), 입력 pooled_all, ER 1 (2048), k 100, batch 4096, 100 epoch, seed 0, W&B offline.
수치는 baccus_VLA-shortcut-exps 의 metrics.json 보고 원문 (results.tsv 미기록). 데이터: Mac mini ~/workspace/baccus/VLA-shortcut-exps/t015/pi05lib/{acts,sae}-vlm-{a,b,c} (2,099 파일, 68.6 GB).

| 배치 | run (acts dir) | commit | 층 | 수집 | 층별 nmse / VE / never_fire |
|---|---|---|---|---|---|
| a | acts-vlm-a | 5df0e208 (full forward) | emb, pg00–05 | 3h27m, 22.3 ts/s, 23.55 GB | emb 0.0079/0.981/61 · pg00 0.0190/0.972/4 · pg01 0.0299/0.957/0 · pg02 0.0323/0.954/0 · pg03 0.0369/0.951/0 · pg04 0.0379/0.952/0 · pg05 0.0389/0.951/0 |
| b | acts-vlm-b | 97c6901c (prefix-only) | pg06–11 | 1h47m, 43.5 ts/s, 20.17 GB | pg06 0.0489/0.942/0 · pg07 0.0508/0.941/0 · pg08 0.0551/0.937/0 · pg09 0.0618/0.932/0 · pg10 0.0667/0.928/0 · pg11 0.0731/0.923/0 |
| c | acts-vlm-c | 97c6901c (prefix-only) | pg12–17 | 2h24m, 32.1 ts/s, 20.17 GB | pg12 0.0936/0.904/0 · pg13 0.0984/0.902/0 · pg14 0.1113/0.889/0 · pg15 0.1050/0.901/0 · pg16 0.0481/0.979/39 · pg17 0.0151/0.992/565 |

**판단 (2026-10-31)**: keep — emb·pg00–16 은 분석에 사용 (VE ≥ 0.89, dead ≈ 0; nmse 는 깊이에 따라 증가해 pg14 에서 최대).
**pg17 은 보류**: never_fire 565/2048 (27.6%), 학습 중 dead 비율이 계속 증가 (마지막 구간 10.9%), 에피소드 coverage 중앙값 0.11 (다른 층 ≈0.3), nmse 는 낮음.
hook 정의는 맞다 (pg17 = final norm 입력 = block 출력, T015 hook 검사). 가설 (미검증): action expert 는 각 PG 층의 K/V (그 층 입력에서 계산) 만 읽으므로
pg17 출력은 action 생성에 쓰이지 않고 표현이 소수 방향에 몰렸다. pg16 도 약하게 같은 경향. next: pg17 (·pg16) activation 의 유효 rank·norm 분포 확인 후
제외하거나 작은 dictionary 로 다시 학습. 사용 GPU-h ≈ 11.2 (승인 14; smoke·설치 1.7, idle 낭비 0.5 포함).

## T017 π0.5-DROID zero-shot on LIBERO — smoke (kubectl A100×1, 2026-11-07)
| task | run | 모델 | suite × 조건 | 결과 | 판단 |
|---|---|---|---|---|---|
| T017 smoke | smoke/T017 (e4b9769f) | lerobot/pi05_droid 72824c0a, openpi DROID stats 403b3a22, 15 Hz joint velocity env | 4 suite × 3 task × 3 init × C0 | C0 0/36; 첫 접촉 대상 ~3/36, 파지 3/36 (모두 다른 물체) | discard (zero-shot) — 파이프라인은 정상 (action 유한, 장면 쪽으로 움직임), C0=0 이라 closed-loop shortcut 측정 불가. ~0.3 GPU-h |
