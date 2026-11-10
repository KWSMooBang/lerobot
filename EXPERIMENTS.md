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

## goal15 — LIBERO-Goal 새 task instruction 15개 (T020 custom, kubectl A100×1, 2026-11-10)
사람이 만든 LIBERO-Goal 변형 15개 (학습 task 와 같은 g06·g12 제외) 를 학습 데이터 형식 문장으로, 학습과 같은 장면·init (0–9) 에서 평가. 다중 목표 추적으로 학습 10개 목표 도달도 기록.
results.tsv 원문: `bench-pi05lib-goal15_trainfmt	done	cdd3aec6d7a6	sc-bench-g1xK1	0.84	2026-11-10T02:56:30+00:00	jobs=1;ours-goal-custom-goal15:n=150;sr_orig=0.253;sr_new=0.220`
(bench_version ours-v1/instr-866b36d5/hf-libero-0.1.4/fix-fix1-d18559ff, exec single). 데이터: Mac mini ~/workspace/baccus/VLA-shortcut-exps/goal15/ (서버 같은 경로).

| variant | instruction (학습 형식) | 실행 가능성 | new % | trained % | nothing % | 빠진 학습 task |
|---|---|---|---|---|---|---|
| g17 | put the cream cheese on top of the cabinet | 3/3 | **100** | 0 | 0 | — |
| g15 | put the cream cheese on the stove | 3/3 | **90** | 10 | 0 | t6 10, t7 10 |
| g05 | put the wine bottle on the stove | 3/3 | 50 | 30 | 20 | wine→rack 30 |
| g14 | put the cream cheese on the plate | 2/3 | 50 | 20 | 30 | cheese→bowl 20 |
| g02 | open the top drawer of the cabinet | 3/3 | 20 | 70 | 10 | **open middle drawer 70**, top drawer+bowl 20 |
| g04 | put the wine bottle on the plate | 3/3 | 20 | 30 | 50 | wine→rack 30 |
| g01 | open the bottom drawer of the cabinet | 0/3 | 0 | 100 | 0 | **open middle drawer 100** |
| g03 | put the wine bottle in the bowl | 3/3 | 0 | 100 | 0 | wine→rack 70, wine→cabinet top 30 |
| g07 | open the top drawer and put the wine bottle inside | 3/3 | 0 | 90 | 10 | **top drawer+bowl 80** |
| g08 | open the top drawer and put the cream cheese inside | 1/3 | 0 | 90 | 10 | **top drawer+bowl 90** |
| g09 | open the middle drawer and put the bowl inside | 3/3 | 0 | 100 | 0 | **open middle drawer 90** |
| g10 | open the middle drawer and put the wine bottle inside | 3/3 | 0 | 90 | 10 | **open middle drawer 90** |
| g11 | open the middle drawer and put the cream cheese inside | 1/3 | 0 | 100 | 0 | **open middle drawer 100** |
| g13 | put the bowl on the rack | 3/3 | 0 | 100 | 0 | **bowl→cabinet top 100** |
| g16 | put the cream cheese on the rack | 3/3 | 0 | 10 | 90 | cheese→bowl 10 |

**판단 (2026-11-10)**: keep. 새 지시 성공 22% (150 ep), 학습 task 로 빠짐이 대부분 → Goal 에서도 지시를 '가장 가까운 학습 문장' 으로 바꿔 실행하는 shortcut 이 강하다.
- 서랍 문장은 전부 학습 task 로 붕괴: 'open the middle drawer and put X inside' → X 와 무관하게 가운데 서랍만 열고 멈춤 (학습 t0, 90–100%), 'open the top drawer and put X inside' → X 대신 bowl 을 넣음 (학습 t3, 80–90%),
  'open the top/bottom drawer' → 가운데 서랍 (70/100%). 서랍 위치 단어와 문장 틀이 학습 task 를 고르고, 물체 이름은 무시된다.
- 물체–목적지 조합은 목적지에 따라 갈린다: stove·cabinet top·plate 는 cream cheese 로 일반화 (90/100/50%) 되지만, rack 은 wine bottle 전용 — cheese→rack 0% (대부분 아무것도 못 함), bowl→rack 은 bowl→cabinet top 으로 100% 빠짐.
  wine bottle 은 자기 학습 목적지 (rack·cabinet top) 로 끌려간다 (g03 100%, g04·g05 30%).
- 한계: 변형당 n=10 (예: 20% 의 95% CI ≈ 6–51%), 학습 형식 문장만, pi05lib 하나. g01·g08·g11 은 스크립트 정책 실행 가능성도 낮음. next: BDDL 형식 문장·control (goal15.yaml), Goal 후보 n01–n23, 다른 모델. 사용 0.87 GPU-h.
**재실행 (영상 전체, 2026-11-10)**: `bench-pi05lib-goal15_trainfmt_allvid	done	cdd3aec6d7a6	sc-bench-g2xK1	1.03	2026-11-10T09:03:52+00:00	jobs=1;ours-goal-custom-goal15:n=150;sr_orig=0.253;sr_new=0.220`
(A100×2, 영상 150개 = Mac mini goal15/runs/bench-pi05lib-goal15_trainfmt_allvid). 첫 실행과 에피소드별 일치: success_new 146/150, success_orig 148/150, outcome 144/150 (96%), 도달 목표 집합 145/150, action bit 동일 1/150.
다른 9 ep 는 모두 300 step 까지 간 것 (g05·g07·g14×5·g15×2), 변형 수준 변화는 g05·g07·g14·g15 에서 ±10%p. action 은 대부분 step 0 부터 |Δ|≈1e-3 로 갈라짐 (같은 seed·RNG 복원; GPU/bf16 수치 차이가 closed-loop 에서 커진 것으로 추정, 원인 미확인).
→ 결론 (서랍 문장 붕괴, rack 비일반화, cheese→stove·cabinet top 일반화) 은 재현됨. 변형당 ±1 ep 는 run 간 noise 범위로 해석. 사용 1.09 GPU-h.
