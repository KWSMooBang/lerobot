# T032 sae_vlm_traj — 사전 등록 (PREREG)

작성 2026-11-26, T028 입력을 SAE 에 넣은 결과를 보기 **전에** 커밋한다.
본 시점까지 본 것: T015 학습 지표 (EXPERIMENTS.md, metrics.json), 학습 codes 의 형식 (codes_meta 열, codes_idx/val shape), T030 결과 (README).
T028 를 SAE 로 인코딩한 값은 아직 계산하지 않았다.

## 0. 데이터 · 모델

- SAE: HF `KWSMooBang/pi05-vlm-sae` rev 79f50c90 → /data0/woosung/datasets/VLA-shortcut-exps/vlm_sae/ (ckpt.pt·wandb 제외).
  `sae-vlm-{a,b,c}/pooled_all/{pg00…pg17}/seed0/`. TopK+AuxK, d=2048, n_latents=2048, k=100. emb 는 T028 에 없으므로 제외.
- T028: /outputs/VLA-shortcut-exps/rich_acts/ 1,560 ep, 51,100 call. vlm_seg 는 T030 캐시
  (/outputs/VLA-shortcut-exps/analysis/layer_pop/_cache/vlm_seg.npy, (18, Ntot, 6, 2048) fp16, npz 값의 그대로 복사) 를 읽기 전용으로 쓴다.
  캐시 = npz 인지 무작위 20 ep 로 bit 일치를 확인한다. seg_ntok 은 npz 에서 읽는다.
- 에피소드 meta·라벨 (task, phase, executed/instructed pool, 사례 목록) 은 T030 `layer_pop.py` 의 `meta()`, `labels()`, `cases()` 를 그대로 import 한다.
- CPU only, 작업자 ≤ 8, `OPENBLAS_CORETYPE=SkylakeX`.

## 1. 입력 복원 정의 (T015 `store.pool_pg(..., "pooled_all")` 와 같게)

T015: 그룹 3개 (cam1 = agentview 256, cam2 = wrist 256, text = 패딩 아닌 언어 token 전부) 의 평균과 token 수 → token 수 가중 평균 (fp32).
T028: 세그먼트 6개 (agentview, wrist, BOS, task, state, template) 평균 + `seg_ntok[:, :6]`. BOS+task+state+template = 패딩 아닌 언어 token 전부
(확인: seg_ntok[2:6].sum() = lang_valid.sum(), 예 1+18+28+10 = 57).

  x_l(call) = Σ_{s=0..5} n_s · vlm_seg[call, l, s] / Σ_{s=0..5} n_s   (fp32)

pgNN = T028 vlm_seg[:, NN] (층 NN 출력). 두 수집 모두 pi05lib (lerobot/pi05_libero_finetuned), bf16 모델, 같은 hook 위치 (decoder layer 출력 = 다음 층 입력).
차이: T015 는 학습 데이터 프레임 (HF libero 이미지), T028 은 평가 환경 렌더 (fix env) + closed-loop 궤적 → 분포 이동 예상.

## 2. 인코딩

numpy 로 `TopKSAE.encode` 를 그대로 구현: x̃ = ((x − b_pre) − μ)/‖·‖ (μ = 차원 평균, eps 1e-6), pre = x̃ W_encᵀ, top-100, ReLU.
torch `TopKSAE` (scripts/shortcut/sae/model.py) 와 1,000 call 에서 값 |차이| < 1e-4 · index 집합 일치 (동률 제외) 를 확인한 뒤 쓴다.
feature 가 "켜짐(fire)" = top-k 안에 있고 값 > 0 (학습 stats 와 같은 정의). 활성 값 = ReLU 값 (정규화 공간).

## 3. 분석과 지표

### A1 복원 품질 (층별)
- nmse = mean ‖x̃ − x̃̂‖² / c_mse (config 의 c_mse, 학습 metrics 와 같은 식), VE = 1 − Σ‖x − x̂‖² / Σ‖x − mean_T028(x)‖².
- never-fire 비율 (T028 51,100 call 중 한 번도 안 켜짐), 학습 fire_freq 와 T028 fire_freq 의 Spearman ρ,
  T028 에서 켜진 feature 가 학습에서 드문 (fire_freq < 1e-3) 비율.
- set 별 (libero3 / goal train / cand / basket / l90) VE.
- pg17: massive 차원 (1577, 1959; T030 목록) 이 ‖x − b_pre − μ‖² 에서 차지하는 비율, 그 차원을 뺀 VE. pg17 은 참고로만 쓴다 (T015 보류 층).
- 판단 기준 (사전): 층의 T028 VE ≥ 0.7 이면 그 층 feature 분석에 사용, < 0.7 이면 "분포 이동 큼" 으로 표시하고 결론에서 약하게 다룬다.

### A2 학습 데이터로 feature 라벨
학습 codes (codes_idx/val, codes_meta: episode, frame, task_index, suite) 273,465 frame. task_index → 우리 task 코드 (S1…, O1…, G1…, L1…) 는 문장 일치로.
phase (학습 frame): 데이터셋 action[:, 6] (gripper 명령, >0 = 닫기). g = 첫 닫기 명령 frame, r = g 뒤 첫 열기 명령 frame.
T027 call 정의 × 10 (call = 10 env step):
  reach f < g − 20 · grasp g − 20 ≤ f < g + 20 · carry g + 20 ≤ f < r − 10 · release r − 10 ≤ f < r + 30 · 그 뒤 = 라벨 없음 (r 없으면 g + 20 이후 carry).
진행도 (frame / 에피소드 길이) 5 bin 도 같이.

feature f 마다: 전체 fire 율 φ, task 별 fire 율 r_t (40), scene 그룹 (T027 scene: SPATIAL, OBJECT, GOAL, Long 장면들) · suite 별 율 (그 task 들 평균), phase 별 율 ρ_p,
top-100 activation frame 의 task 분포 (최빈 task 비율) 와 진행도 중앙값.

태그 (사전 임계값, 위에서부터 첫 해당 = 주 태그):
1. **rare**: φ < 1e-3.
2. **always-on**: min_t r_t ≥ 0.5 (모든 task 에서 절반 이상의 frame 에서 켜짐).
3. **task-specific**: 최상위 task t* 에서 r_{t*} ≥ 0.05 이고 r_{t*} ≥ 5 × mean_{t≠t*} r_t.
   - 세부 **instruction (같은 장면)**: t* 의 장면이 같은 배치를 공유하는 학습 task 가 있는 그룹 (GOAL, Long 의 같은 장면 task 쌍) 이고
     r_{t*} ≥ 5 × mean_{같은 장면 다른 task} r_t → 장면이 같은데 지시문만으로 갈리는 feature.
   - 나머지는 "task (장면+지시 혼재)" (SPATIAL·OBJECT 는 task 마다 배치·물체가 달라 분리 불가).
4. **scene-specific**: 최상위 장면 그룹 g* (task ≥ 2 개) 에서 r_{g*} ≥ 0.05, r_{g*} ≥ 5 × mean_{g≠g*} r_g, 그리고 그룹 안에서 고르게: min_{t∈g*} r_t ≥ (1/3) max_{t∈g*} r_t.
5. **suite-specific**: 4 와 같은 규칙을 suite 단위로 (장면 그룹 규칙에 안 걸린 것).
6. **phase-specific**: 학습 frame 4 phase 율에서 max_p ρ_p ≥ 0.05 이고 max_p ρ_p ≥ 3 × min_p ρ_p.
7. **mixed**: 나머지.
phase 조건 (6) 은 주 태그와 별도로 플래그로도 기록한다 (예: task-specific + carry).
객체·목적지 feature (A4 용): T030 primary object / destination (task ≥ 2 개를 가진 값만). 값 o 를 가진 task 들의 평균 율 r_o ≥ 0.05,
r_o ≥ 3 × (o 없는 task 평균), 그리고 o 를 가진 task 중 ≥ 2 개에서 r_t ≥ 2 × (o 없는 task 평균) (한 task 전용이 아님).

### A3 feature 궤적 (사례, T030 과 같은 사례 정의 `cases()`)
basket O9 (A100 ctl/near/opp, 4090 ctl/far20/far25), Goal g10 + G7·G2, L90 lscene fallback (T030 이 고른 task), S1, L1.
- 히트맵: 층 {1, 4, 8, 12, 16} (+ pg17 참고) 마다, 사례 안에서 조건 평균 activation 의 call 간·조건 간 분산이 큰 top-30 feature × call. 행에 A2 태그.
- 고정 PCA 궤적: 층마다 U40 (학습 40 task × 10 init 의 모든 call) dense code (2048) 로 PCA 를 맞춰 PC1–2, 층 small multiples. 시간 옅은→진한 색, ○ 시작, × 첫 grasp, ■ 끝.

### A4 fallback 시행 (핵심)
pool = T030 `labels()["pool"]` 그대로 (cand31 에서 다른 학습 task 를 실행 116, L90 lscene 에서 학습 task 를 실행 78; executed 목록, instructed = cand: base task, L90: 문장이 가장 가까운 학습 task).
- task feature 집합: F_T = {f : r_T ≥ 0.05, r_T ≥ 5 × mean_{t≠T} r_t} (학습 라벨). 시행마다 F_I \ F_E (지시 전용), F_E \ F_I (실행 전용).
- 켜짐 비율 on(F, 시행, call) = F 중 켜진 비율.
- 정규화 지수 (층 × call, 시행 중앙값 + 시행 bootstrap 95% CI):
  IDX_I = (on_I(fallback) − on_I(E 학습 시행)) / (on_I(I 학습 시행) − on_I(E 학습 시행)), IDX_E 는 I·E 를 바꿔서.
  기준 시행 = T028 U40 의 같은 call (T028 안에서 기준을 맞춰 분포 이동을 상쇄). 분모 < 0.02 인 (시행, call) 은 뺀다.
  0 = 실행 task 를 진짜로 할 때만큼 꺼짐, 1 = 지시 task 를 진짜로 할 때만큼 켜짐.
- object / destination feature 도 같은 지수 (지시 값 ≠ 실행 값인 시행만, n 작음 → 참고).
- 같은 관측 쌍 (T029 pairs.json, valid & rich_available; cand vs base 첫 프레임 동일): call 0 에서 feature 활성 차이를 쌍 Wilcoxon (양측) → BH q < 0.05 인 feature 수 (층별).
  그 feature 들의 태그 분포, cand 쪽 증가 feature 가 학습 라벨상 지시 task / base task 중 어느 쪽 feature 인지. call 1–3 에서 같은 feature 의 차이가 유지되는지.

### A5 basket
같은 기계 짝: 4090 ctl4090 vs far20 / far25 (같은 task·init), A100 libero3 Object ctl vs near / opp. 거리: ctl 0, near 10, far20 20.6, far25 25.7, opp 59 cm.
- call 0 (팔이 움직이기 전, 순수 시각) 과 운반 구간 (T030 phase carry, 둘 다 있는 짝만) 에서 feature 활성 차이 Δ.
- 바구니 민감 feature: ctl vs far25 쌍 Wilcoxon BH q < 0.05 (call 0).
- 분류 (사전): λ = mean Δ20 / mean Δ25.
  **graded**: 0.3 ≤ λ ≤ 0.95 이고 부호 같음, **on/off**: λ > 0.95 (20 cm 에서 이미 포화) 또는 fire 율 차이 ≥ 0.5 이면서 λ > 0.8, **기타**: 나머지.
  near (A100 짝) 의 Δ 가 같은 부호·더 작은 크기인지 추가 확인.

### A6 Kaufman 연결
층 l (0–16) feature 의 decoder 방향 d (W_dec 열, 단위) → 층 l+1 read-in (T029 정의: [W_K; W_V] · diag(1 + w_ln), weights.npz) 행공간 P (512/2048, chance 0.25).
PF = ‖Pᵀd‖² . pg17 은 읽는 층이 없음 (L17 자기 read-in 은 참고 proxy 로만). 태그별 PF 분포, 층별 중앙값, 바구니·지시 민감 feature 의 PF.

### A7 비교
- 같은 디코딩 (T030 `fit_predict`: StandardScaler → PCA 50 → 로지스틱, task40·suite 내 task = init leave-one-out, phase = init LOO balanced acc,
  교차 일반화 = U40 학습 → pool 적용, executed / instructed 비율) 을 (a) SAE dense code, (b) 같은 층의 SAE 입력 x (pooled 벡터) 에 적용, T030 call 목록 CALLS.
  T030 v_all6/v_task 결과는 numbers.json 에서 참고로 인용.
- (선택) T031 1단계 expert SAE (exp_mean, 층 {2,5,…,17} × step {0,9}) 가 있으면: U40 call 들에서 VLM feature 활성과 expert feature 활성의 상관,
  VLM feature 마다 가장 상관이 큰 expert feature 와 태그 조합.

## 4. 가설 (방향 사전 지정)

- **H1 (복원)**: T028 VE 는 모든 층에서 학습 VE 보다 낮지만 pg00–16 은 ≥ 0.7. pg17 이 가장 나쁘다 (massive 차원).
- **H2 (라벨)**: pg01–16 에서 활성 feature (rare 제외) 의 과반은 task/scene/suite-specific 이고, always-on 은 소수 (< 10%).
- **H3 (fallback, 핵심)**: T030 과 같은 방향 — pg01 이후 call ≥ 2 에서 IDX_E ≥ 0.5 이고 IDX_I ≤ 0.3 (지시 task feature 는 대체로 꺼져 있고 실행 task feature 가 켜짐).
  pg00 에서는 IDX_I 가 가장 크다 (층 중 최대).
- **H4 (같은 관측 쌍)**: call 0 에서 지시문만 바뀌어 달라지는 feature 수는 깊을수록 많아진다 (pg00–16 Spearman ρ > 0).
- **H5 (basket)**: 바구니 민감 feature 의 과반은 graded (on/off 보다 많다).
- **H6 (Kaufman)**: task-specific (instruction 포함) feature 의 PF 중앙값 > always-on feature 의 PF 중앙값 (층 대부분, MWU).
- **H7 (비교)**: SAE code 의 task40 디코딩은 같은 층 pooled 벡터와 0.05 이내, 교차 일반화의 executed 우세도 같은 방향.

## 5. 통계 · 기타
- 다중 비교: 검정 수 보고 + BH-FDR (q < 0.05).
- 기존 출력 덮어쓰지 않음. 출력: /outputs/VLA-shortcut-exps/analysis/sae_vlm_traj/ (README, PREREG, numbers.json, feature_labels.csv, fig/).
- 사전 등록 뒤 바꾼 것은 README 에 "PREREG 와 다른 점" 으로 적는다.
