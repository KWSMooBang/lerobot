# T029 neuro_dyn: 신경과학 집단 활동 분석을 pi05lib 에 적용

상태: **Stage A (A1, LIBERO-PRO 벡터) 와 Stage B (T028 rich 재추출 1,560 ep: B1 재검정, A2–A7, A8 제안) 모두 완료.**

| 경로 | 내용 |
|---|---|
| `PREREG.md` | 가설·지표·기준선. 결과 보기 전에 커밋함 (89b0981b) |
| `numbers.json` | 핵심 수치 |
| `a1/` | A1 그림, 전체 결과 (`a1_results.json`), 탐색적 분할 (`a1_exploratory_fallback_split.json`) |
| `b/` | Stage B 결과 (b1, a2–a7 의 json·그림, pairs.json, 탐색적 분석 json) |
| `b_pilot/` | pilot 60 ep 로 코드만 점검한 출력. **해석하지 말 것** |
| `A8_ames_proposal.md` | A8 중간 지시문 전환 실험 제안 |
| `_cache/` | weights.npz, 에피소드 벡터 캐시 (stage A), `b/` 압축 캐시 (stage B) |

스크립트 (브랜치 `exp/sc-neuro-dyn`):
- `scripts/shortcut/analysis/neuro_dyn.py`: stage A
- `scripts/shortcut/analysis/neuro_dyn_b.py`: stage B. diptest 가 필요해서 worktree 의 `.venv-nd` (numpy·scipy·scikit-learn·matplotlib·diptest) 로 실행한다.
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

# Stage B: T028 rich 재추출 (1,560 ep, 재생은 모두 4090)

공통 사항
- 사전등록은 `PREREG.md` 의 addendum B (9cf0b203) 와 amendment B.1 (746f16d8) 에 있다. B.1 은 pilot 코드 점검 뒤, 전체 결과를 보기 전에 썼다.
  - 원 규칙: T028 PCA 에 쓰이지 않은 20 % (pca_fit = 0) 만 쓴다.
  - 바뀐 규칙: 이 제한은 T028 PCA 좌표를 쓰는 분석 (A2b, A4 토큰축) 에만 적용한다. raw 키를 쓰는 분석은 전체 에피소드를 쓴다.
  - A3 은 held-out 만으로도 다시 냈다. 결과는 같았다.
- massive activation (VLM 잔차 dim 1577, 1959) 처리:
  - 거리·PCA·CCA 는 토큰별 rms 정규화 후 두 차원을 뺀다.
  - read-in PF 는 전체 벡터로 계산한다 (가중치가 무엇을 읽는지 결정하므로).
  - K/V 는 처리할 필요가 없다.
- T028 의 weights.npz 와 stage A 에서 HF 체크포인트로 만든 read-in 행렬은 최대 차 6.6e-8 로 일치한다.
- A2b·A4 토큰축은 T028 PCA 128 성분 중 앞 10 개만 쓴다. 그래서 128 이후 성분에서 PCA 가 잃는 분산 (held-out 설명 비율 L17 0.937) 은 결과에 닿지 않는다.
- 검정 수와 FDR (가족별 BH):
  - B1 8 / 0
  - A2 10 / 9
  - A3 18 / 0
  - A4 2 / 2
  - A5 5 / 5
  - A6 12 / 11
  - A7 3 / 1

## B1. stage A 탐색 결과의 확증 재검정 (Kaufman, K·V 수준까지)
**무엇을 했나**
- goal cand31 trial 을 같은 init 의 base task 학습 trial 과 짝지었다. call 0 관측이 렌더 잡음 수준까지 같은 경우 (영상 |Δ| ≤ 2) 만 썼다.
  - 310 개 중 150 개가 짝이 된다.
  - 나머지 80 개는 장면이 수정된 후보, 80 개는 base 가 둘인 후보다.
- 비교 집단: FALLBACK (원래 학습 행동, 85) vs NEW (새 행동, 22).
- 지표:
  - 과제 토큰 차이의 read-in PF
  - |ΔK|, |ΔV| 와 그 상대 크기
  - expert attention mass
  - expert L17 차이의 크기와 행동 potent 비율
- p 는 후보 안에서 라벨을 섞는 순열 검정에서 얻었다.

**그림** `b/b1_fallback_vs_new.png`: 왼쪽부터 층별 read-in PF, |ΔV|/|V|, |ΔK|/|K| (중앙값·IQR), 오른쪽은 expert 차이 크기 vs 행동 potent 비율.

**결과: 재현되지 않았다.**
- stage A 가설 "fallback 에서 expert 차이가 작고 null 에 있다" 는 cand 에서는 반대 방향이다.
  - expert 차이 중앙값 FALLBACK 121 vs NEW 95 (p = 0.99)
  - 행동 potent 비율 0.036 vs 0.026 (p = 0.56)
- VLM read-in PF 는 같다 (0.305 vs 0.300, p = 0.12). 단 TOST 동등성은 보이지 못했다 (사전에 정한 margin 이 0.003 으로 매우 작다).
- 그림에서 가장 뚜렷한 것은 K·V 차이다. NEW 쪽이 모든 층에서 더 크다.
  - V 상대 차이 0.79 vs 0.49, K 0.43 vs 0.26
  - 후보 안 순열 검정에서는 유의하지 않다 (p = 0.12–0.26)
  - 후보를 무시하면 p ≈ 1e-6 이다 (탐색적, `b/b1_exploratory_between_within.json`)
- 이유: 두 결과가 모두 나온 후보는 15 개 중 3 개뿐이다. 결과는 거의 후보마다 정해지므로, 이 차이는 "어떤 후보 지시인가" 와 섞여 있다. 후보 수준 상관은 ρ = 0.15–0.21 로 유의하지 않다.
- **해석**:
  - cand 에서 fallback 은 "expert 가 지시 차이를 버린다" 기보다, 지시 차이가 VLM·K·V 단계부터 작은 (base 와 비슷한) 후보에서 일어나는 경향이다.
  - stage A (PRO 새 목표) 에서 본 expert 병목은 이 데이터에서 일반화되지 않는다.
- LIBERO-90 lscene (짝 없음, 보조) 에서도 trained 와 nothing 결과 사이에 attention mass × |V| 차이가 없다 (p = 0.51).

## A2. 토큰 축
**(a) 언어 토큰** (cand vs base, 같은 관측 150 쌍)
- difflib 으로 같은 단어끼리 맞추고 cosine 거리를 쟀다.
- **같은 단어도 바뀐 단어 양쪽에서 갈라진다.** prefix attention 이 양방향이기 때문이다.
  - 바뀐 단어 앞에 있는 같은 단어의 K 거리 중앙값: L0 0 → L2 0.001 → L8 0.030 → L17 0.124
  - 층이 깊어질수록 커진다 (후보별 ρ 중앙값 0.975, q = 7e-4).
  - 거리는 바뀐 단어 바로 뒤 토큰 (+1) 에서 가장 크다 (L14: 0.25). 양쪽으로 갈수록 줄지만 L17 에서는 6 토큰 떨어진 곳도 0.06–0.12 다.
  - 언어 토큰에서 "다시 합쳐지는" 층은 없었다.
- 토큰별 Kaufman 분할: 같은 단어 차이와 바뀐 단어 차이의 read-in PF 는 층마다 부호가 바뀐다 (L0·5·8 은 같은 단어가 +0.03–0.07, L2·14 는 −0.02–0.05). 일관된 null 치우침은 없다.
- 그림 `b/a2_tokens.png` 왼쪽·가운데.

**(b) 50 action 토큰 = 계획된 horizon** (held-out 312 ep)
- 토큰 0 → 49 경로 길이 (L17, PC 10 개) 는 denoise 가 진행될수록 **줄어든다** (step 0 → 4 → 9: 2044 → 1566 → 1181).
  - 사전 예측 (늘어난다) 은 기각이다. 노이즈의 들쭉날쭉한 경로가 매끈한 계획으로 정리되는 과정이다.
- step 9 경로 길이는 같은 call 의 예측 chunk 경로 길이와 거의 같은 순서를 갖는다 (task 별 ρ 중앙값 0.95). 실제 다음 10 step 의 eef 이동 거리와도 ρ 0.69 이다 (둘 다 q = 5e-22).
  - 즉 expert 의 action 토큰 축 궤적은 계획된 움직임의 모양을 그대로 담는다.
- 그림 오른쪽: 조건별 call 0 평균 토큰 궤적 (PC1-2, 점 = 토큰 0).

## A3. Elsayed: 준비 vs 움직임 부분공간
- 준비 = grasp (또는 contact) 이전 call, 움직임 = grasp 부터 release 까지.
- alignment index (AI) 를 공분산을 맞춘 무작위 부분공간 1,000 개와 비교했다.

| 집합 | expert AI (무작위 중앙값) | VLM agentview | VLM task |
|---|---|---|---|
| libero3 | 0.58 (0.65) | 0.48 (0.50) | 0.37 (0.39) |
| goal train | 0.60 (0.70) | 0.52 (0.54) | 0.47 (0.52) |
| basket | 0.64 (0.68) | 0.47 (0.56) | 0.39 (0.38) |

- expert 는 무작위보다 조금 낮지만 (goal p = 0.02) FDR 후에는 유의하지 않다 (q ≥ 0.18).
- **원숭이처럼 준비와 움직임이 직교로 재편되지 않는다.** 두 부분공간이 상당히 겹친다.
- "VLM 이 expert 보다 더 정렬" 이라는 예측도 기각이다. VLM AI 가 오히려 더 낮다.
- VLM call 0 상태로 움직임 구간의 expert 평균 상태를 예측한 R² (task 로 CV):
  - libero3 0.14, basket 0.18
  - expert call 0 으로는 0.00–0.01
  - goal 은 둘 다 음수
  - 움직임에 대한 정보는 시작 시점에 expert 보다 VLM 에 더 있다.
- held-out 만 써도 같다.
- 그림 `b/a3_alignment.png`.

## A4. Churchland jPCA (cross-condition 평균 궤적 제거, task 로 CV)
| 축 | R² 제약 없는 M | R² skew (회전) | 비율 |
|---|---|---|---|
| action 토큰 0→49 | 0.011 | 0.010 | 0.94 |
| denoise step 0→9 | 0.33 | 0.02 | 0.06 |
| call (20 bin) | 0.023 | 0.015 | 0.64 |

- 사전 예측 "토큰축 비율 > 0.5, denoise < 0.5" 는 숫자로는 둘 다 맞다 (조건별 중앙값 0.89 / −0.04, q < 0.002).
- 하지만 **토큰축과 call 축은 선형 동역학 자체가 거의 설명되지 않는다** (R² ≈ 0.01–0.02). 회전 구조가 있다고 말할 수 없다.
- denoise 축은 선형 동역학이 1/3 을 설명하지만 회전이 아니다 (수축·확장형). 노이즈 → 계획으로 끌려가는 흐름이다.
- Churchland 식 회전 동역학은 VLA 에서 확인되지 않았다.
- 그림 `b/a4_jpca.png`.

## A5. Gallego: task 별 manifold
- expert L17 상태로 task 별 10 차원 manifold 를 만들었다.
  - 주각 평균 54° (무작위 62°, 같은 task 반분 32°)
  - 시간 정렬 후 held-out CCA r 은 대부분 0.6–0.95 다.
- pick-and-place 쌍 vs 유형이 다른 쌍: CCA 0.749 vs 0.734 (q = 0.002). 차이가 작고, CCA 는 어느 쌍에서나 높다 ("접근–잡기–이동" 의 공통 시간 구조). 공유 동역학의 강한 증거로 보기 어렵다.
- **basket, 같은 물체 task 에서 near 와 비교**:

| 조건 | 주각 − 반분 주각 | CCA r |
|---|---|---|
| ctl4090 (학습 조건) | +4.9° | 0.88 |
| far20 | +9.1° | 0.88 |
| far25 | +8.4° | 0.82 |
| opp | +15.7° | 0.73 |

  - 주각은 모두 반분보다 크다 (q < 0.01).
  - **far20/far25 는 "같은 동역학, 조금 다른 readout"** 이다. 사전 기준 (CCA 가 ctl 대비 0.1 이내) 을 만족한다.
  - **opp 는 동역학도 다르다** (0.15 낮음).
  - ctl4090 을 near 반분 CCA 의 대리값으로 썼다.
- 그림 `b/a5_manifold.png`.

## A6. Inagaki: multi-noise (같은 관측에서 노이즈 16 개)
- 두 행동 사이 coding direction (CD) 을 학습 fold 에서 만들고, 평가 fold 의 16 개 표본을 투영했다 (0 = 집단 1, 1 = 집단 2).
- 결과는 `b/a6_multinoise.png`.
- **같은 관측의 16 개 표본은 거의 같은 점에 모인다.**
  - 관측 안 SD 는 끝점 사이 거리의 0.01–0.03 이다.
  - 행동 공간 (계획된 이동량) 에서도 관측 안 퍼짐은 trial 간 퍼짐의 3–5 % 다 (`b/a6_action_space_robustness.json`).
  - **노이즈가 모드를 고르지 않는다. 어느 행동으로 갈지는 관측과 지시가 결정한다.**
- trial 사이의 분포:
  - 학습 지시 기준선 (두 학습 task): 끝점 근처 85–91 %, 가운데 7–10 %. 뚜렷한 쌍봉이다.
  - 새 지시:
    - cand (fallback vs new): 끝점 39 %, 가운데 24 %
    - basket far (성공 vs 밖에 놓음): 43 % / 20 %
    - lscene (trained vs nothing): 47 % / 23 %
  - dip 검정은 모두 유의하다 (q < 4e-4). 그러나 표본이 수천 개라 민감한 것이고, 모양은 기준선보다 훨씬 연속적이다 (넓은 단봉 + 어깨).
  - **새 지시에서는 학습된 두 끝점 사이 중간 상태가 흔하다. 순수한 discrete attractor 는 아니다.**
- trial 평균 투영으로 결과를 예측한 AUC (후보/task 로 CV):
  - lscene 0.88 (q = 9e-4)
  - basket far 0.64 (q = 0.002)
  - cand 0.58 (q = 0.06)
- 한계: seed 별로 최종 chunk 만 저장했기 때문에, denoise 경로를 따라 노이즈 차이가 줄어드는지는 잴 수 없다. A2b·A4 의 단일 seed 경로가 정리되는 것만 확인했다.

## A7. 조합 코드 (Kim 2024 의 CD 아이디어 + NNLS 확장)
- Kim 등은 "같은 동작이라도 문맥마다 별도의 준비 활동이 생기고 (병렬 기억), 옛 문맥으로 돌아가면 옛 준비 활동이 다시 켜진다" 를 coding direction 으로 보였다. NNLS 분해는 그 논문에 없는 내 확장이다.
- 방법: cand trial 의 call 별 expert 상태 (전체 평균 뺌) 를 source task 들 (roles.csv 의 base, 물체 source, 목적지 source) 의 시간 정렬 중심의 비음수 조합으로 맞췄다.
- 결과:
  - source 로 맞춘 R² 는 0.48 로, 무작위 비-source task 3 개보다 0.29 높다 (q = 7e-8). **cand 상태는 실제로 source task 상태들의 조합으로 잘 설명된다.**
  - NEW 에서는 초반 물체 source 가중치가 크다가 (w목적지 − w물체 ≈ −0.55) 진행하면서 목적지 쪽으로 옮겨 간다. 기울기 중앙값은 NEW 0.33, FALLBACK 0.05 다.
    - 예측 방향과 같지만 유의하지 않다 (p = 0.07, q = 0.11).
  - base 비율은 차이가 없다 (NEW 0.70 vs FALLBACK 0.69, p = 0.54).
- 그림 `b/a7_combinatorial.png`.

## A8. Ames: 제안만 (`A8_ames_proposal.md`)
- libero_goal 에서 학습 지시 쌍 10 개 (+ cand ↔ base 6 쌍) 를 쓴다.
- 지시를 잡기 전 / 잡은 직후 / 운반 중에 바꾼다.
- 규모: 1,080 ep, 약 3.6 GPU-h (rich 재추출 포함 약 5 GPU-h).
- 코드 변경: client 쪽 지시 전환 옵션 하나.
- 예측: 잡기 전 전환은 새 task 의 준비 영역을 다시 지나고, 잡은 뒤 전환은 다른 물체 쌍에서 원래 행동을 유지한다.
- 이번 결과 (A3: 준비·움직임 부분공간이 겹침, A6: 결정은 관측이 함) 를 보면, "준비 영역 재방문" 보다 "현재 상태에서 바로 재매핑" 쪽일 가능성도 크다. 실험이 두 가능성을 가를 수 있다.

## 종합 결론
1. **지시문 정보는 막히지 않는다.**
   - PRO 에서 새 목표 차이는 expert 가 읽는 방향에 장면 차이 이상으로 실린다 (A1).
   - 의미가 같은 paraphrase 만 null 쪽으로 밀린다.
2. **shortcut (fallback) 의 위치는 데이터마다 다르다.**
   - PRO 에서는 expert 가 지시 차이를 행동 방향으로 바꾸지 않는 패턴이 탐색적으로 보였다.
   - cand31 확증 재검정 (B1) 에서는 재현되지 않았다. 오히려 지시 차이가 VLM·K·V 단계부터 작은 후보에서 fallback 이 많다 (후보와 섞여 있어 결정적이지 않다).
3. **신경과학 그림 중 맞는 것과 안 맞는 것**:
   - Elsayed 의 직교 재편 (A3), Churchland 의 회전 (A4): 보이지 않는다.
   - Gallego 의 "같은 동역학, 다른 readout": basket far 에서 맞고 opp 에서는 깨진다 (A5).
   - Inagaki 의 discrete attractor: 학습 지시에서는 뚜렷하지만 새 지시에서는 중간 상태가 흔하다. 노이즈가 아니라 관측이 끝점을 고른다 (A6).
   - 조합 지시의 상태는 source task 상태의 조합으로 잘 설명된다 (A7).

## 한계
- outcome 라벨은 원래 실행 (대부분 A100) 의 것이고, activation 은 4090 재생 (teacher-forced) 이다.
  - A100 1,260 ep 은 r ≥ 0.9998 로 일치하고, basket_far 300 은 비트 단위로 일치한다.
  - 모든 비교는 같은 재생 기계 안에서 했다.
- B1 은 두 결과가 다 나온 후보가 3 개뿐이라 검정력이 낮다.
- LIBERO-90 은 짝이 되는 같은 관측이 없어 A2a 의 L90 비교는 하지 않았다 (장면은 같아도 init 이 다르다).
- A3·A5 의 기준선은 공분산 맞춤 무작위다. A5 CCA 는 공통 시간 구조 때문에 어디서나 높다.
- A5 의 near 반분 CCA 는 ctl4090 으로 대신했다.
- A4 의 R² 는 선형 동역학 가정 아래의 값이다.
- A6 는 노이즈 경로 수렴을 잴 수 없다.
