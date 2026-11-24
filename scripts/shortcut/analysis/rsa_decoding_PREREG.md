# T027 사전 등록 (PREREG) — RSA · 짝 회귀 · 디코딩

작성: 2026-11-24, 결과를 보기 전. 이 파일은 결과 계산 전에 커밋합니다. 이후 바꾸는 것은 맨 아래 "변경 기록"에 이유와 함께 적습니다.

## 0. 데이터와 특징

- **에피소드:** kubectl 추출 데이터(A100), basket far(4090). 새 추출 없음, CPU만.
  - U40: 학습 task 40개 × init 0–9 = 400 (S/O/L = libero3, G = cand31 train10)
  - U90: LIBERO-90 90 task × 10 = 900
  - cand31: 31 후보 × 10 = 310 (Goal 장면)
  - basket: A100 ctl(=libero3 O)/opp/near, 4090 ctl/far20/far25 (각 100). 짝은 같은 기계 안에서만.
- **특징(call마다, T021 정의 그대로):** expert, gemma, siglip, attention, patch_attention, action + gemma_task(VLM 문장 토큰만, libero90 정의).
  - **주의:** expert·gemma·gemma_task·siglip은 층별 크기 요약(norm, mean |MLP|)이지 hidden vector가 아닙니다. 의미 검정(특히 gemma)은 약할 것으로 예상합니다. 결론은 "크기 요약에서 보이는가"로 한정합니다.
- **층 묶음:** L0–5 / L6–11 / L12–17 (expert, gemma, gemma_task, attention), patch_attention은 추출 때의 3개 층 묶음, siglip은 0–8 / 9–17 / 18–26. action은 층 없음.
- **사건과 phase:** T025 `events_from_steps`(grasp = 두 손가락이 같은 물체 + 닫기 명령, 5 step 이상). 경계(bound) = 첫 grasp call, grasp가 없으면 첫 접촉 call, 둘 다 없으면 에피소드 중간.
  - pre = call ≤ bound, post = call > bound. 각 phase를 5 구간으로 다시 나눔(선형 보간). traj = pre 5 + post 5.
  - 디코딩용 4단계 phase(call 라벨): reach(첫 grasp 전 ~ grasp 3 call 전), grasp(grasp call −2 … +1), carry(grasp +2 … release −1), release(release −1 … +2 이후 = post-release). grasp 없는 에피소드는 제외.
- **정규화:** 특징 차원별 z-score. RSA는 단위(U40, U90) 전체 call로, B는 집합 안 call로, 디코딩은 학습 fold로만 맞춥니다.

## 1. 모델 RDM

과제(task) 단위 정의, 에피소드 단위는 같은 정의를 에피소드에 적용.

- **의미(semantics.csv, 지시문 기준; BDDL은 물체·목적지 해석에만 사용):**
  - (a) verb: 지시문의 동사 집합 {place(put/place/pick-up-and-place/stack), push, open, close, turn_on, turn_off, lift(pick up만)}. 거리 = 1 − Jaccard.
  - (b) object: 다루는 물체 집합(BDDL 첫 인자, 번호 제거·이름 정규화; open/close는 drawer/microwave, turn on/off는 stove). 1 − Jaccard.
  - (c) destination: (관계, 목적지) 집합. 관계 = in / on / on top of / front of / left of / right of / under. 목적지 없는 task(open 등)는 빈 집합(둘 다 비면 0). 1 − Jaccard.
  - (d) spatial qualifier: 물체를 고르는 위치 수식어("between the plate and the ramekin", "next to the ramekin", "at the back", "middle", "left/right" …). 다르면 1, 같거나 둘 다 없으면 0.
  - (e) word: 지시문 단어 집합(the/a/it 제외) 1 − Jaccard.
  - (f) emb: pi05 체크포인트 Gemma 입력 임베딩(lm_head와 공유, paligemma tokenizer)의 문장 평균, 1 − cosine.
  - **sem** = mean(a, b, c, d). B의 회귀와 부분 상관에 쓰는 의미 거리.
  - 애매한 경우는 semantics.csv `ambiguous` 열에 표시(예: L9·t38 BDDL에는 turn on이 있으나 지시문에 없음 → 지시문 기준).
- **동작(실행된 동작, 기록된 step):**
  - motion: eef 위치(3) + gripper 폭(1, ×10 m 단위 맞춤) — phase 별로 bound step 기준 정렬 후 pre / post 각 10점으로 다시 나눔. traj = 둘 이어붙임. task RDM = task 평균 사이 Euclidean.
  - motion_act: 기록된 action(7) 같은 방식.
  - **실행 기준**이므로 cand31·L90의 대체 시행은 실제로 한 동작으로 들어갑니다.
- **장면:** scene_id(같은 장면 0 / 다름 1; S/O/G는 suite 장면, L·L90은 LIBERO scene 이름), scene_pix(call 0 agentview 이미지 64×64 RGB Euclidean, task 평균).

## 2. 분석과 가설

### A. RSA (task 단위; U40, U90 따로)
- 특징 RDM: 교차검증 거리(init 짝/홀로 나눈 두 반쪽 평균의 내적형 Euclidean, 잡음 편향 없음). 특징 × phase(pre, post, traj) × 층 묶음(all + 3).
- Spearman ρ(특징 RDM, 모델 RDM), 위 삼각만. p = task 라벨 순열(Mantel, 1000회).
- 부분 상관(순위 변환 후 선형 잔차): sem | motion + scene_pix, motion | sem + scene_pix, scene_pix | sem + motion.
- 잡음 천장: init 반쪽 RDM끼리 Spearman, Spearman–Brown 보정.
- **가설:**
  - H-A1: gemma, gemma_task — 부분 ρ(sem) > 0, 그리고 부분 ρ(sem) > 부분 ρ(motion).
  - H-A2: expert, action — 부분 ρ(motion) > 0, 그리고 부분 ρ(motion) > 부분 ρ(sem).
  - H-A3: siglip, patch_attention — 세 부분 ρ 중 scene이 가장 큼.
  - 판정은 traj · 층 all 에서 합니다(나머지는 탐색).

### B. 의미·동작이 갈리는 짝 (에피소드 짝 단위)
- 집합: basket_A100(ctl/opp/near), basket_4090(ctl/far20/far25), L90_lscene(LIBERO-10 장면의 LIBERO-90 에피소드 + 같은 장면 L10 학습 에피소드, 짝은 같은 장면 안), cand31(train10 + cand31), spatial(S 100).
- d_feature(짝) ~ β_sem·d_sem + β_motion·d_motion + β_scene·d_scene (모두 z-score, 표준화 β). p = 에피소드 순열(Mantel, 999회). 분산 분할: 고유 = R²(전체) − R²(그 변수 뺌), 공유 = R²(전체) − 고유 합.
- 2×2: 같은 지시 여부 × 같은 실행 동작 라벨 여부 → 평균 특징 거리 / 전체 짝 평균.
  - 실행 라벨: 학습 = task, basket = task×조건, cand31 = 판정의 수행 학습 task(cat ∈ trained/fail_trained) 아니면 후보 자신, L90 = 도달한 L10 목표가 있으면 그 L10 task 아니면 자기 task.
- **가설:**
  - H-B1: gemma, gemma_task — β_sem > β_motion.
  - H-B2: expert, action — β_motion > β_sem.
  - H-B3 (basket): 같은 init ctl까지 거리가 near < far20/far25 < opp 순으로 커지는 것은 expert·action에서 뚜렷하고 gemma_task에서는 작다(지시가 같으므로).

### C. 디코딩
- PCA(학습 fold에서만 맞춤, ≤50) + 표준화 + 다항 로지스틱 회귀.
- CV: task·suite·phase·basket 조건은 leave-one-init-out(10 fold). verb·object·destination은 task 단위 GroupKFold(새 task로 일반화; 대표 라벨 = 첫 동사/첫 물체/첫 목적지, 2개 task 이상 가진 라벨만). success는 U90, task 단위 5-fold, balanced accuracy.
- 시간: (i) 정규화 시간 20 구간, (ii) grasp 기준 상대 call −10…+15, (iii) pre/post.
- 혼동 행렬: 40-task 디코더(pre, post), 대칭화한 혼동률과 모델 RDM의 Spearman(음의 상관 = 가까운 task끼리 헷갈림).
- 교차 일반화(학습 에피소드 전부로 학습, call 위치별):
  - cand31: Goal train10 10-way, 다른 학습 task를 대신 수행한 시행(바탕 ≠ 수행) → 예측 = 수행 task(executed) vs 바탕 task(instructed 대리) 비율.
  - L90_lscene: 40-way, outcome = trained 이고 수행 L10 ≠ 지시문이 가장 가까운 학습 task(tasks.csv nearest_trained) 인 시행 → executed vs instructed-proxy 비율.
  - basket: 40-way, 각 조건에서 자기 O task로 예측하는 비율; 같은 기계 안 조건(ctl/near/opp 또는 ctl/far20/far25) 3-way 디코딩.
- **가설:**
  - H-C1: gemma_task(·gemma)는 instructed 비율이 expert·action보다 높고, expert·action은 executed 비율이 더 높다.
  - H-C2("지시는 스위치"): gemma_task의 instructed 비율은 초반 call이 후반보다 높다 / expert·action의 executed 비율은 초반부터 높다.
  - H-C3: task 디코딩은 attention·patch_attention이 가장 높고, grasp 전에 이미 높다(T025와 일관).

## 3. 다중 비교
- A, B의 모든 p값 수와 BH-FDR q를 numbers.json에 같이 보고합니다. 가설 판정은 위에서 정한 셀(traj, 층 all)만 씁니다.

## 4. 변경 기록
- (없음)
