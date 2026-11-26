# T033 sae_popdyn — 사전 등록 (PREREG)

작성 2026-11-26. 결과를 보기 **전에** 커밋한다. 단계 a → b → c → d → e 순서로 진행한다.
본 시점까지 본 것: T032·T031 README 와 캐시 형식 (codes npz shape, feature_labels.csv 열, select.pkl 키), T029 A7 코드, roles.csv, T028 meta (에피소드 수·call 수·라벨 분포).
이번 분석의 어떤 수치도 아직 계산하지 않았다.

## 0. 데이터

- **VLM (이해 담당) code:** `/outputs/VLA-shortcut-exps/sae/vlm/analysis/_cache/codes_pgNN.npz` (T032; 51,100 call × top-100 idx·val). 층 pg00–pg17.
  feature 태그: `feature_labels.csv` (tag, sub, top_task …).
- **동작 담당 code:** `/data0/woosung/datasets/VLA-shortcut-exps/cache/t031_t028_codes/codes_L{ll}_s9.npz` (denoise 단계 9; 51,100 × top-64). 층 L00–L17.
  feature 라벨: `select.pkl["L{l}_s9"]` (live, ph_sel, t_sel, bt, bp).
- **에피소드:** T028 1,560 ep. meta·라벨·fallback pool·사례 = T030 `layer_pop.py` 의 `meta()`, `labels()`, `cases()` (T032 와 같음).
  - U40 = 학습 task 40개 × init 10 (400 ep). fallback pool = cand 116 + L90 lscene 78 = 194 ep (지시 task I, 실행 task E).
  - grasp / release call = manifest 의 grasp_call / release_call (T025 이벤트에서 온 값).
  - cand 결과: FALLBACK = outcome ∈ {trained, fail_trained}, NEW = {new, new_byproduct} (T029 와 같은 정의). 성공 = manifest success.
- 조성 역할: roles.csv 의 base_task, object_source, destination_source. 후보가 여러 개 (`3;10`) 면 T029 처럼 **첫 번째**를 쓴다.
- 같은 관측 쌍: T029 `pairs.json` 중 valid 이고 rich_available 인 쌍 (T032 와 같은 150 쌍).

## 1. 공통 전처리

- **feature 제거:** VLM = T032 tag `rare` (학습 발화율 < 0.1%) 제거. 동작 담당 = T031 `live` 가 아닌 것 제거.
  추가로 T028 51,100 call 에서 한 번도 안 켜진 feature 도 제거 (값이 전부 0 이라 쓸모 없음). 층마다 남은 수를 보고한다.
- **크기 맞추기:** feature 마다 T028 전체 call 에서의 최댓값으로 나눈다 (Williams 2018 처럼 0–1 범위). 모든 단계가 이 공간을 쓴다.
- **시간 축:** call index 0–27. T028 의 모든 에피소드는 call ≥ 28 개 (최소 28) 이므로 **짧은 에피소드는 없고**, 28 번째 이후 call 은 버린다.
  단계에 관한 질문은 grasp 정렬 (call − grasp_call, −6 … +10) 판도 같이 본다 (grasp 없는 에피소드는 그 분석에서 제외).
- **층:** VLM 주 분석 pg00–pg15, pg16 주의, pg17 참고. 동작 담당 주 분석 L05–L17, L00–L04 는 보고하되 표시 (massive 채널).
  비싼 단계 (a, d) 는 고정 부분집합 VLM pg01·04·08·12·15, 동작 L05·08·11·14·17. 싼 단계 (b, c, e) 는 18 층 전부.
- **전체 네트워크 판 (whole):** VLM pg00–pg15 블록 + 동작 L05–L17 블록을 이어 붙이고, 블록마다 자기 RMS (= sqrt(call 평균 ‖x‖²)) 로 나눈다.
  같은 개념이 여러 층에 반복해 나타나므로 whole 판의 "feature 수" 는 독립 신호 수가 아니다. 비교용으로 VLM 만 (vlm_all), 동작만 (exp_all) 판도 b·c·e 에서 계산한다.
- **해석 주의 (README 에 쓴다):** 정책은 call 사이에 내부 상태를 넘기지 않는다. call 에 걸친 궤적은 환경을 통한 closed loop 의 결과이지, 내부 재귀 동역학이 아니다.

## 2. 단계별 정의 · 지표 · 가설

### a. 텐서 분해 (TCA, Williams et al. 2018)
- 텐서 X[trial, feature, call], trial = T028 1,560 ep 전부, call 0–27. 비음수 CP 분해, HALS (직접 구현, 희소 MTTKRP), 무작위 초기화.
  수렴: 상대 오차 변화 < 1e-5 또는 300 iter.
- rank 격자 {2, 4, 6, 8, 10, 12, 15, 20, 25, 30}, rank 마다 init 10 개. 지표: 정규화 재구성 오차 ‖X − X̂‖/‖X‖, 안정성 = 최선 fit 과 나머지 9 개의 유사도
  (Williams 2018 / tensortools: 성분을 헝가리안으로 짝짓고, 짝마다 세 factor cosine 의 곱 × (1 − |λ₁−λ₂|/max), 평균) 의 중앙값.
- **rank 선택 규칙:** 안정성 중앙값 ≥ 0.75 인 rank 중 가장 큰 것 R*. 없으면 가장 작은 rank. 층마다 따로.
- 성분 보고: feature loading 상위 20 개의 태그·최상위 학습 task, 시간 profile 의 정점 call, 그 성분에 trial loading 이 큰 (상위 10%) trial 의 grasp/release call 중앙값,
  trial loading 을 suite / task / 조건 (set·condition) 으로 나눈 η² (분산 중 라벨이 설명하는 비율).
- 태그 정렬: 성분마다 상위 20 feature 중 가장 많은 태그의 비율 ("태그 순도") 과 최상위 학습 task 가 같은 비율 ("task 순도"); 우연 기준 = feature 를 무작위로 20 개 뽑은 1,000 회.
- **fallback 질문:** fallback pool 194 ep 의 trial loading 벡터 a (길이 R*) 를 U40 trial loading 의 task 평균과 cosine 비교.
  지표 = E 쪽이 더 가까운 trial 비율 (E 가 여럿이면 가장 가까운 것). 보조 = U40 의 5-최근접 이웃 중 E task 수 vs I task 수.
- **가설 H-a1:** 모든 주 층에서 E 쪽이 더 가까운 비율 > 0.5 (이항 검정, 층 간 BH). **H-a2 (탐색):** VLM 에서 I 쪽 비율은 깊을수록 커진다 (Spearman, 5 층 + 참고).
  **H-a3:** 성분의 task 순도가 우연보다 높다 (성분 중앙값 vs 무작위 분포).

### b. 분산 나누기 (회귀 기반, dPCA 식; Kobak et al. 2016)
- 행 = (trial, call), 대상 trial = U40 400 + fallback pool 194. 창 W ∈ {0–2, 3–9, 10–19} 마다 따로.
- 요인 (one-hot): **시간** (창 안의 call), **지시 task I** (U40 은 자기 task, pool 은 T032 의 I), **실행 task E** (pool 은 E 목록, 여럿이면 multi-hot), **장면** (meta scene).
  I·E·장면 요인은 주효과 + 창 안 call 과의 상호작용을 포함한다 (dPCA 의 "조건 + 조건×시간" 주변화에 해당).
- 모델: ridge (λ 는 전체 모델의 안쪽 CV 로 {1e-2 … 1e3} 에서 선택 후 고정), 출력 = 그 층 feature 전부 (전처리 공간).
  교차 검증: init 으로 나눈 10-fold (leave-one-init-out). R² = 1 − Σ 오차² / Σ (y − 학습 fold 평균)² (feature 전체 합산 = 분산 가중).
- **고유 R²** (unique) = R²(전체) − R²(그 요인을 뺀 모델). 주 지표는 **fallback trial 행에서만 잰 R²** (U40 에서는 I = E 이라 둘을 가를 수 없음). 모든 행 기준도 보고.
- 보조: 지시 요인을 "문장 자체 (task/variant id)" 로 바꾼 판.
- **dPCA 주변화 (균형 부분집합):** Goal 장면, cand 변형 중 FALLBACK 과 NEW trial 이 모두 있는 것. 조건 3개 = base 학습 trial / cand FALLBACK / cand NEW,
  변형마다 조건 평균 → 시간 주변화, 조건 주변화 (조건 + 조건×시간) 의 분산 비율 (Kobak 2016 식). 완전 요인 설계는 없다 (지시 base 이면서 NEW 인 칸이 없음).
- **가설 H-b1:** 모든 주 층·모든 창에서 unique R²(E) > unique R²(I) (fallback 행). **H-b2:** VLM unique R²(I) 는 깊을수록 커진다 (Spearman pg00–pg15).
  **H-b3 (탐색):** 동작 담당의 시간 unique 비율이 VLM 보다 크다.

### c. 조성 task = 원천 task 의 혼합? (Kim 2024 + NNLS, T029 A7 반복)
- cand 310 ep (31 변형 × 10). 원천 = roles.csv 의 base, object source, destination source (중복 제거, Goal 학습 task 만).
- call c 마다 y_c = cand code(c) − g_c, 원천 행렬 A_c = [μ_t(c) − g_c] (μ_t(c) = Goal 학습 task t 의 U40 trial 평균 at call c, g_c = Goal 학습 10 task 평균 at call c).
  비음수 최소제곱 w_c = argmin ‖A_c w − y_c‖, w ≥ 0. R²_c = 1 − ‖잔차‖²/‖y_c‖². trial R² = call 평균.
  보조: 중심화 없는 판 (비음수 code 를 비음수 centroid 로 직접 맞춤).
- 무작위 기준: 원천이 아닌 Goal 학습 task 를 같은 수만큼 100 회 뽑아 같은 계산 → trial 의 무작위 R² 평균. 차이 = R²(원천) − R²(무작위).
- 가중치 궤적 w(c), 기울기 = polyfit(τ, w_dest − w_obj) (τ = c/27), object ≠ destination source 인 trial 만.
- 검정: (1) 원천 > 무작위: 변형별 평균 차이에 Wilcoxon (T029 와 같음). (2) 기울기 NEW vs FALLBACK: 변형으로 층화한 순열 검정 (10,000 회).
  (3) 성공 vs 실패 (cand 전체, success 플래그) 같은 순열 검정. 층 간 BH (VLM 18 + 동작 18 + whole 3 판 각각 한 가족).
- **가설 H-c1:** 원천 R² > 무작위 (대부분 주 층에서 BH 유의). **H-c2:** 기울기 NEW > FALLBACK (T029 원 벡터: 0.33 vs 0.05, p = 0.07).
- T029 비교: T029 는 동작 담당 층 17 원 벡터, 진행도 20 bin 정렬이었다. 같은 진행도 정렬 판을 동작 L17 에서 보조로 계산한다.

### d. 의사소통 부분공간 (Semedo et al. 2019)
- 입력 X = VLM pg 층 code, 출력 Y = 동작 L 층 code (전처리 공간, 열 중심화). 행 = 51,100 call 전부. CV = leave-one-init-out (10 fold).
- 쌍: (pg_{l−1} → L_l) 와 (pg_l → L_l) for l ∈ {5, 8, 11, 14, 17} (동작 층 l 은 VLM 층 l 의 K/V 를 읽고, 그 K/V 는 층 l 입력 = pg_{l−1} 출력에서 만들어진다).
  추가 쌍: pg04 → L17, pg15 → L05. 기준: VLM 안 pg_l → pg_{l+1} (l ∈ {1,4,8,12}), 동작 안 L_l → L_{l+1} (l ∈ {5,8,11,14}).
  whole 판: VLM pg00–15 블록 결합 (PCA 500) → 동작 L05–17 블록 결합 (PCA 500).
- ridge (λ 안쪽 CV) 후 축소 순위 회귀 (RRR): B_r = B_ridge V_r V_rᵀ (V_r = 학습 fold 적합값의 상위 r 주성분). r = 1 … 60.
  성능 = CV R² (분산 가중). **의사소통 차원** = full ridge 의 평균 R² − 1 SE (fold 간) 이상인 가장 작은 r.
- X 쪽 예측 부분공간 = B_ridge V_r 의 열 공간 (정규직교화, r = 의사소통 차원, 전체 데이터 적합).
  feature 별 loading = 그 부분공간 투영 행 노름; 태그별 (instruction / task-shared-scene / task-confounded / phase / scene·suite / mixed) 평균을 전체 평균 대비 비로 보고, 태그 순열 1,000 회로 p.
- 같은 관측 쌍 call 0: d = cand − base 의 X code 차이. 지표 = ‖P d‖² / ‖d‖² (P = 의사소통 부분공간 투영), 쌍 평균.
  기준 1 = 같은 차원의 무작위 정규직교 부분공간 1,000 개, 기준 2 = X 의 상위 r 주성분 부분공간.
- **가설 H-d1:** VLM → 동작 의사소통 차원 < 같은 깊이의 VLM 안 기준 (pg_l → pg_{l+1}) 의 차원. **H-d2:** 지시문 차이의 의사소통 부분공간 비율 > 무작위 부분공간의 95 백분위.
- README 주의: VLM SAE 입력은 이미지 위주 token 평균이다. 동작 담당이 실제로 읽는 것은 token 별 K/V 다.

### e. 궤적 기하
- 조건 평균 궤적 m_k(c) (조건 k 의 trial 평균, call 0–27). 거리 = cosine 거리 1 − cos.
- **갈라지는 call:** 조건 쌍 (A, B) 마다 call 별 거리 d(c). 바닥 = A 를 무작위로 반씩 나눈 두 평균의 거리 (200 회 평균).
  순열: A·B trial 라벨을 섞어 1,000 회 → call 별 p. 갈라지는 call = d(c) > 바닥 이고 p < 0.05 인 첫 call.
  쌍: cand 변형 × base task (각 변형의 10 trial vs base 학습 10 trial; FALLBACK·NEW 변형별), basket (ctl A100 vs near, opp; ctl4090 vs far20, far25), g10 vs G7·G2.
- **속도:** trial 마다 ‖x(c+1) − x(c)‖ (유클리드, 전처리 공간) 를 grasp 정렬 (−6…+10) · release 정렬로 평균. 정점의 상대 call.
- **유효 차원:** participation ratio PR = (Σλ)² / Σλ², suite 별 (S, O, G, L) U40 call 벡터의 공분산.
- **바구니 O9:** 같은 기계 차이 벡터 Δ_near = m_near − m_ctlA, Δ_far20 = m_far20 − m_ctlV, Δ_far25 = m_far25 − m_ctlV, Δ_opp = m_opp − m_ctlA (call 0–27 평균 벡터와 call 별 둘 다).
  순서: ‖Δ_near‖ < ‖Δ_far20‖ < ‖Δ_far25‖ 인가. 경로 비 = (‖Δ_near‖ + ‖Δ_far20 − Δ_near‖ + ‖Δ_far25 − Δ_far20‖)/‖Δ_far25‖ (1 = 곧은 한 경로).
  opp 이탈 = Δ_opp 에서 선분 [0, Δ_far25] 까지의 거리 / ‖Δ_far25‖. 잡음 바닥 = 각 조건 split-half 차이 노름.
- **fallback:** trial 마다 call 별 d_I = cos 거리 (trial, I 의 U40 평균), d_E (E 의 U40 평균, 여럿이면 최소). d_I − d_E 의 call 별 중앙값과 Wilcoxon p,
  "E 쪽으로 갈라지는 call" = d_I − d_E > 0 이 유의 (p < 0.05) 해지는 첫 call.
- **가설 H-e1:** fallback trial 은 call 0 부터 E 쪽이 더 가깝다 (VLM 얕은 층·동작 전 층). **H-e2:** cand–base 갈라지는 call 은 깊은 VLM 층에서 0 (같은 관측에서 지시문만 다름), 얕은 층에서 > 0.
  **H-e3:** 바구니 거리 순서가 지켜지고 경로 비 < 1.5; opp 이탈 > 0.5. **H-e4:** 속도 정점은 grasp ±2 call 안.

## 3. 실행 · 출력

- 코드 `scripts/shortcut/analysis/sae_popdyn.py`, 브랜치 `exp/sc-sae-popdyn`. CPU 만, 작업자 ≤ 8, `OPENBLAS_CORETYPE=SkylakeX`.
- 출력 `/outputs/VLA-shortcut-exps/sae/population/` (README.md, PREREG.md, numbers.json, fig/). 큰 캐시는 `/data0/woosung/datasets/VLA-shortcut-exps/cache/t033_*/`.
- 단계마다 커밋 → README 갱신 → lead 에 `[STAGE]` 보고 → 다음 단계.
- 사전 등록에 없는 분석은 README 에 "탐색 (사전 등록 아님)" 으로 표시한다.
