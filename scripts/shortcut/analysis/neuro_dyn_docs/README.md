# T029 neuro_dyn: 신경과학 집단 활동 분석을 pi05lib 에 적용

상태: **Stage A (A1, LIBERO-PRO 벡터) 완료**. Stage B (T028 rich re-extraction, A2–A8) 는 데이터 도착 후 진행.

| 경로 | 내용 |
|---|---|
| `PREREG.md` | 가설·지표·기준선. 결과 보기 전에 커밋함 (89b0981b) |
| `numbers.json` | 핵심 수치 |
| `a1/` | A1 그림, 전체 결과 (`a1_results.json`), 탐색적 분할 (`a1_exploratory_fallback_split.json`) |
| `_cache/` | weights.npz (read-in·readout 가중치), 에피소드 벡터 캐시 |

스크립트: `scripts/shortcut/analysis/neuro_dyn.py` (브랜치 `exp/sc-neuro-dyn`).
실행: `OPENBLAS_CORETYPE=SkylakeX <viz venv python> neuro_dyn.py all`.
- 이 numpy 빌드 (scipy-openblas 0.3.34) 는 EPYC 9554 에서 ZEN·Haswell 커널로 실행하면 작은 행렬 곱에서 segfault 가 난다.
- SkylakeX 커널은 모든 점검을 통과했다.

---

## A1. Kaufman et al. 2014: 출력 null / potent 공간

### 무엇을 했나
Kaufman 은 운동 피질 활동을 둘로 나눴다.
- 근육 출력에 닿는 방향 (potent)
- 출력과 무관한 방향 (null)

그리고 "준비 활동은 null 공간에 있어서 움직임을 일으키지 않는다"는 것을 보였다. 여기서는 출력 대신 **가중치로 정확히 정의되는 두 개의 읽기 경로**를 쓴다.

1. **VLM → expert (read-in)**
   - expert 는 VLM 층 j 의 K, V 만 읽는다: K/V = RMSNorm(x)·W.
   - 그래서 VLM 잔차 x 중 expert 가 볼 수 있는 방향은 [W_K; W_V]·diag(1+w_norm) 의 row space 다 (512/2048 차원, 무작위 기준 25 %).
   - 기록된 `vlm_vec[l]` 은 층 l 의 **출력**이므로 층 l+1 의 가중치를 쓴다 (5→6, 11→12).
   - **층 17 의 출력은 expert 가 전혀 읽지 않는다**. 구조적으로 100 % null 이다.
2. **expert → 행동 (readout)**
   - 행동은 action_out_proj · (1+scale_t) 로 나온다. 앞 7 행이 실제 쓰이는 LIBERO 행동 차원이다.
   - scale_t 는 denoise 단계별 AdaRMS scale 로, 가중치에서 직접 계산했다.
   - potent 는 7/1024 차원이다 (무작위 기준 0.68 %).

지표 (차이 벡터 d 에 대해):
- **PF** = 차이의 에너지 중 potent 공간에 들어간 비율
- **GR** = 무작위 방향 대비 증폭률. 1 이면 무작위와 같다.

비교 대상은 call 0 의 짝 차이다. 같은 suite·task·init 에서는 관측이 동일하다 (siglip 특징 100 % 일치 확인). 따라서 paraphrase·new-goal 차이는 순수하게 지시문 때문이다.
- 지시문: paraphrase(semantic) − none, 새 목표(task) − none, task 간 (같은 init)
- 장면: environment, position(swap), object − none
- 기준선:
  - 무작위 부분공간
  - ACT: 전체 none 활동의 분산이 potent 에 들어간 비율. residual 은 비등방이라 이것이 더 공정한 기준이다.
  - MOVE: 에피소드 안에서 call 이 지나며 생기는 변화 (expert)

통계:
- 단위는 (suite, task) 클러스터 40 개, Wilcoxon 검정.
- A1 전체 492 검정에 BH-FDR 을 적용했고, q < 0.05 는 346 개다.

### 그림 읽는 법
- `a1/a1_vlm_potent_fraction.png`
  - 막대는 각 차이의 PF 다.
  - 검은 점선은 무작위 (0.25), 회색 점선은 전체 활동 (ACT).
  - 막대가 회색 선보다 낮으면 그 차이는 expert 가 읽기 어려운 방향 (null 쪽) 에 치우쳐 있다는 뜻이다.
- `a1/a1_expert_potent.png`
  - 왼쪽: denoise 단계별 expert L17 차이의 PF (로그축). 검은 실선 = 에피소드 내 움직임.
  - 오른쪽: task 정체를 potent 7 차원으로 디코딩한 정확도 vs 무작위 null 7 차원.
- `a1/a1_instr_goal_by_call.png`: 새 목표 차이의 PF 가 에피소드 진행 (call) 에 따라 어떻게 변하는지.

### 결과
**사전 가설 (H1) "지시문 차이는 null 에 있어 expert 에 닿지 않는다" 는 의미가 바뀌는 지시문에 대해 기각된다.**

| VLM task 토큰 (rms 정규화) | PF L5→6 | GR L5→6 | PF L11→12 | GR L11→12 |
|---|---|---|---|---|
| 무작위 | 0.250 | 1 | 0.250 | 1 |
| 전체 활동 ACT | 0.286 | 1.32 | 0.300 | 1.33 |
| **새 목표 − none** | **0.286** | 1.30 | **0.304** | **1.36** |
| task 간 | 0.282 | 1.28 | 0.294 | 1.29 |
| 장면 (env/pos/obj) | 0.26–0.28 | 1.16–1.22 | 0.27–0.28 | 1.16–1.18 |
| **paraphrase − none** | **0.230** | **0.86** | 0.263 | 1.00 |

- 새 목표 차이는 L11 에서 장면 차이보다 **더** potent 하다 (PF +0.031, GR +0.19, q = 2e-9). 전체 활동과는 구별되지 않는다 (q = 0.36).
- paraphrase 차이만 null 쪽으로 치우친다. L5 에서는 무작위보다도 낮다 (PF 0.230, q = 2e-4). 의미가 같은 문장 차이는 expert 가 못 읽는 방향으로 밀린다.
- 지시문 정보는 이미지 토큰에도 들어간다 (prefix 양방향 attention).
  - L5 agentview·wrist 에서 지시문 차이의 PF 는 0.21–0.23 으로, 장면 차이 0.17–0.19 보다 높다 (q < 1e-9).
  - 하지만 크기는 아주 작다. 상대 노름은 paraphrase 0.4 %, 새 목표 2 % 이고, 장면 차이는 6 % 다.
- 강건성 검사로, 층 l+1 대신 층 l 의 가중치를 써 봤다 (명세의 "층 l 에서 K = RMSNorm(x)·W_K" 해석).
  - 새 목표 차이 ≥ 장면 차이라는 결론은 같다. L11 task 에서 새 목표 0.233, 장면 0.220.
  - paraphrase 가 null 쪽으로 치우치는 정도는 약해진다 (L5 task 에서 0.238, 장면 0.24).

| expert L17 → 행동 7 차원, denoise step 0 (pooled PF) | 값 |
|---|---|
| 무작위 | 0.0068 |
| 에피소드 내 움직임 MOVE | 0.070 |
| 새 목표 − none | 0.067 |
| task 간 | 0.076 |
| 장면 | 0.018–0.029 |
| paraphrase | 0.016 |

- H2 ("task 차이는 call 0 에서 움직임보다 null 쪽", Kaufman 의 준비 활동 유비) 는 클러스터 중앙값 기준으로 통계적으로 성립한다 (새 목표 PF 차 −0.036, q = 1e-9). 다만 pooled 로 보면 새 목표·task 간 차이는 움직임과 거의 같은 비율로 potent 하다. 강한 "준비 = null" 그림은 아니다.
- 출력에 영향을 주면 안 되는 차이 (paraphrase, 장면 일부) 는 행동 readout 에서 null 에 가깝다 (GR 1–3, 움직임은 10).
- **정보는 null 에도 충분하다.** task 정체 디코딩 (leave-one-init-out) 결과, potent 7 차원과 무작위 null 7 차원이 비슷하다.
  - goal: 0.83 vs 0.86
  - spatial: 0.96 vs 0.89
  - 10: 0.91 vs 0.88
  - Kaufman 처럼 출력과 무관한 공간에도 task 정보가 실려 있다.
  - VLM 디코딩은 potent·null 모두 1.0 으로 포화되어 비교가 무의미했다.
- denoise 가 진행될수록 (t → 0) 모든 차이의 PF 가 줄어든다 (MOVE 0.070 → 0.017). final norm scale 이 바뀌며 readout 이 좁아지는 효과다.

**탐색적 (사전등록 아님): 새 목표를 줬는데 원래 목표로 돌아간 에피소드 (fallback, success_orig, n = 213) vs 새 목표 성공 (n = 91)**

| | VLM L11 task PF | expert L17 PF | expert 차이 노름 |
|---|---|---|---|
| fallback | 0.299 | **0.012** | **85** |
| 새 목표 성공 | 0.297 | 0.052 | 200 |

- fallback 에서도 지시문 차이는 VLM 에서 똑같이 expert 가 읽을 수 있는 방향에 있다.
- 그러나 expert 안에서는 차이가 절반 이하로 작고, 거의 전부 행동 null 에 있다.
- 따라서 **shortcut 의 병목은 VLM 이 지시문을 null 로 버리는 데 있지 않다. expert 가 읽은 지시문 차이를 행동 potent 방향으로 바꾸지 않는 데 있다.**
- 여기서 expert 가 무엇을 읽는지 (query·attention) 는 B 단계에서 볼 일이다.

### 한계
- PRO 벡터는 segment 의 **토큰 평균**이다.
  - RMSNorm 은 토큰별로 작용하므로 rms 정규화는 근사다.
  - expert 는 토큰별 K/V 를 attention 으로 가중해 읽으므로, row space 에 있다고 실제로 읽히는 것은 아니다 (query 가 그 방향을 보는지는 별개).
  - B 데이터 (토큰별, expert q_proj) 로 보강할 예정이다.
- expert 층 5·11 의 readout 은 최종 readout 행렬로 투영한 근사다 (logit-lens 방식). 그래서 step ≥ 5 에서 L5 PF 가 0.5 로 나오는 등 해석하기 어렵다. L17 만 해석한다.
- 모든 PRO 에피소드는 한 대의 4090 (GPU 1) 에서 돌았으므로 same-machine 규칙은 자동으로 만족한다.
- fallback 분할은 탐색적이다. success_orig 와 success 가 겹치는 에피소드가 17 개 있다.

---

## A2–A8
T028 데이터 도착 후 진행.
