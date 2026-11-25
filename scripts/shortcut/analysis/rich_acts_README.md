# rich_acts — pi05 내부 활성화 벡터 재추출 (T028, 2026-11-25)

기존 에피소드의 replay 입력(call마다 이미지·state·env step)을 pi05(`lerobot/pi05_libero_finetuned`)에 다시 넣었습니다. 이번에는 norm이 아니라 **방향과 크기를 가진 벡터**를 남겼습니다. 토큰 축·디노이징 축·여러 noise 축도 함께 담았습니다.

신경과학 집단 분석(Kaufman 2014 null space, Elsayed 2016, Churchland 2012 jPCA, Gallego 2018, Inagaki 2019, Kim 2024)을 VLA에 적용하기 위한 데이터입니다. 분석은 T029가 합니다.

- **추출 코드:** repo 브랜치 `exp/sc-rich-acts`
  - `scripts/shortcut/analysis/rich_extract.py`
  - `scripts/shortcut/serve/activations.py`의 `RichHooks`
  - 테스트 `scripts/shortcut/tests/test_rich_extract.py`
- **장비:** VGI-102 RTX 4090, GPU 0 한 장
  - 원래 run은 A100(ferrari) 또는 4090에서 돌았습니다. `manifest.csv`의 `machine` 열에 있습니다.
- **noise 재현:** attn_extract.py와 같습니다.
  - 에피소드 시작에 `torch.manual_seed(seed)`를 걸고, call마다 순서대로 한 번씩 뽑습니다 (single exec).
  - 새로 env를 돌리지 않았습니다.

## 폴더

| 경로 | 내용 |
|---|---|
| `<set>/<job>/<episode>.npz` | 에피소드 하나 (아래 키). `np.savez_compressed`(무손실 zlib)라서 `np.load`로 바로 읽힙니다 |
| `manifest.csv` | 에피소드마다 한 줄 (아래 열) |
| `pca_bases.npz` | (stream, 층)마다 PCA 기저 |
| `pca_report.json` | PCA 설명 분산 비율. fit 비율과 held-out 비율을 k별로 |
| `weights.npz` | 분석용 가중치 (아래) |
| `config.json` | 본 추출의 키 설정 (PCA 차원, MLP 층) |
| `extract_{pilot,main}.jsonl` | 에피소드마다 확인 수치. 이전 추출과의 상관, 기록 action과의 차이, 시간, 크기 |
| `summary.json` | GPU-h, 디스크, 일치 요약 |
| `logs/` | 실행 스크립트(`run_gpu.sh`)와 로그 |
| `_smoke/` | GPU smoke 10 ep (모든 stream raw) |

## set

| set | job | 에피소드 | 원래 장비 | condition |
|---|---|---|---|---|
| `libero3` | ours-{spat,obj,l10}-custom-train10 | 300 | A100 | spatial / object / long |
| `basket_a100` | ours-obj-custom-basket | 200 | A100 | near / opp |
| `goal` | ours-goal-custom-{train10,cand31} | 410 | A100 | train / cand |
| `basket_far` | ours-obj-custom-{train10,basket} | 300 | 4090 | ctl4090 / far20 / far25 |
| `libero90` | ours-l90-custom-ctl10 (L10 장면 task 35개) | 350 | A100 | part / lscene |

call 수(C)는 에피소드마다 고정입니다: Long 52, Goal 30, LIBERO-90 40, 나머지 28. call k의 env step은 `t[k] = 10k`입니다.

## manifest.csv 열

- **식별:** `episode`, `set`, `job`, `path`(이 폴더 기준 상대 경로)
- **장비:** `machine`(원래 run), `replay_machine`(4090)
- **task:** `task`(S1…/O1…/L1…/G1…/g01…/tNN), `variant`, `init`, `condition`, `instruction`
- **결과:** `success`
- **사건 call:** `grasp_call`, `release_call`, `contact_call`
  - S/O/L/Goal은 T025 events_*.csv에서 가져왔습니다.
  - basket은 outcomes.csv의 step ÷ 10입니다.
  - LIBERO-90은 episodes.jsonl의 first_grasp_t ÷ 10이고, release는 비어 있습니다.
- `executed_task`, `outcome`: 대신 실행한 학습 task(Goal의 executed_training_tasks, LIBERO-90의 reached_instead)와 실패 유형
- `n_calls`
- **multi-noise probe:**
  - `probe_calls`는 `;`로 구분한 4개입니다: {0, g−2, g+3, 마지막}, 범위를 넘으면 잘랐습니다.
  - `probe_anchor`는 g의 출처입니다: grasp, 없으면 첫 contact(contact), 둘 다 없으면 중간 call(middle).
- **pilot:** `pilot`, `pca_fit`
  - pilot = 60 ep입니다. set별로 층화했고 task마다 init 하나입니다.
  - `pca_fit=1`(48 ep)로 PCA를 맞췄고, `0`(12 ep)은 held-out 검증용입니다.
- **원본 경로:** `obs_path`(replay 입력), `old_out_path`(T021 추출), `steps_path`(기록된 action)

## npz 키

**기호:**
- C = call 수, N = 그 에피소드의 최대 언어 token 수 (비패딩, 46–60)
- 층 번호는 0–17 (VLM·expert 모두 18층)
- step = 디노이징 step 0–9 (0이 순수 noise 입력)
- 세그먼트 6개 순서: agentview(256 patch), wrist(256), BOS(1), task(지시문), state(숫자), template("Task: … , State: … ;\nAction: "의 나머지)
- 빈 세 번째 카메라와 언어 패딩은 빠져 있습니다.

**dtype:** f16 키는 `astype(np.float32)`로 읽으십시오. 층 17 template token의 massive activation(dim 1577·1959, 최대 ≈2.9e5)이 fp16 범위를 넘습니다. 그런 배열은 그 파일에서만 float32로 저장했습니다 (주로 raw `vlm_tok`, 드물게 `vlm_seg`).

### 기존 키 (T021/T023과 같은 정의, 비교용)

| 키 | shape | 내용 |
|---|---|---|
| `attn_mass` | (C, 10, 18, 7) | action query의 attention이 입력 종류 7개(6 세그먼트 + action)에 가는 비율. head 평균 |
| `attn_head` | (C, 18, 8, 7) | 같은 값의 head별, step 평균 |
| `attn_img` | (C, 3, 512) f16 | 두 카메라 patch 512개에 가는 attention. 층 묶음 0–5/6–11/12–17 |
| `exp_h`, `exp_mlp` | (C, 10, 18) | expert 층 출력의 token L2 norm 평균 / MLP 중간 \|값\| 평균 |
| `vlm_h`, `vlm_mlp` | (C, 18, 6) | VLM prefix pass의 세그먼트별 norm / MLP \|값\| |
| `sig_h` | (C, 3, 27) | SigLIP 층별 token norm (카메라 3개, 세 번째는 빈 카메라) |
| `exp_vec` | (C, 10, 3, 1024) f16 | expert 층 {5,11,17} 출력의 action token 평균, 모든 step (T023) |
| `act_pred` | (C, 10, 7) | 실행된 chunk (후처리 후, env action) |
| `seg_ntok` | (C, 7) | 입력 종류별 key 수 |
| `t` | (C,) | env step |

T023의 `vlm_vec`는 본 추출에서 뺐습니다. `vlm_seg`의 부분집합이기 때문입니다 (pilot 파일에는 남아 있습니다).

### 1. VLM residual 벡터 (prefix pass)

| 키 | shape | 내용 |
|---|---|---|
| `vlm_seg` | (C, 18, 6, 2048) f16 | VLM decoder 층 출력(residual stream, final norm 전)의 세그먼트 평균. 18층 모두, raw |
| `vlm_tok_pc` | (C, 7, N, 256) f16 | 층 {0,2,5,8,11,14,17} 출력의 언어 token 벡터를 층별 PCA-256으로 투영. 패딩 위치는 0 |
| `lang_ids` | (C, N) int32 | token id (패딩 −1) |
| `lang_seg` | (C, N) int8 | 세그먼트 id (2 BOS, 3 task, 4 state, 5 template, 7 패딩) |
| `lang_valid` | (C, N) bool | 비패딩 |

**투영 식:** `vlm_tok_pc[c, j, n] = (x − mean_L) @ components_L.T`
- L = 층 slot j의 층 번호
- 기저는 `pca_bases.npz`의 `vlm_tok/L{L}/mean`, `vlm_tok/L{L}/components`

**원래 공간으로 근사 복원:** `x ≈ pc @ components + mean`

### 2. expert가 읽는 K·V (VLM cache)

VLM self_attn의 `k_proj` / `v_proj` 출력이고, **RoPE 전**입니다. Gemma 2B는 KV head가 1개이고 head_dim은 256입니다.
- K에는 이후 위치에 따른 RoPE가 붙습니다. V는 cache에 있는 그대로입니다.

| 키 | shape | 내용 |
|---|---|---|
| `k_seg`, `v_seg` | (C, 18, 6, 256) f16 | 세그먼트 평균, 18층 |
| `k_tok`, `v_tok` | (C, 7, N, 256) f16 | 언어 token별, 층 {0,2,5,8,11,14,17}, raw |

### 3. action expert

| 키 | shape | 내용 |
|---|---|---|
| `exp_tok_pc` | (C, 6, 3, 50, 128) f16 | 층 {2,5,8,11,14,17} × step {0,4,9} × action token 50개(계획된 50 step), 층별 PCA-128. 기저 `exp_tok/L{L}/…` (step을 합쳐서 맞춤) |
| `exp_mean` | (C, 3, 18, 1024) f16 | step {0,4,9} × 18층, 층 출력의 action token 평균, raw |
| `exp_mlp_mean` | (C, 3, 6, 4096) f16 | step {0,4,9} × 층 {2,5,8,11,14,17}, MLP 중간(down_proj 입력)의 token 평균, raw (pilot 파일은 18층) |

### 4. 디노이징 경로

| 키 | shape | 내용 |
|---|---|---|
| `den_x` | (C, 11, 50, 32) f32 | x_k, k = 0…10. x_0 = noise, x_10 = 출력 chunk (정규화 공간, 32차원 중 앞 7개가 실제 action) |
| `den_v` | (C, 10, 50, 32) f32 | step k의 velocity v_k. `x_{k+1} = x_k − v_k / 10` |

### 5. multi-noise probe

probe call 4개(`mn_calls`, manifest의 `probe_calls`와 같음)에서 prefix를 한 번 계산합니다. 그 KV cache를 고정하고 noise 16개로 디노이징했습니다.

| 키 | shape | 내용 |
|---|---|---|
| `mn_calls` | (4,) | probe call 번호 |
| `mn_seeds` | (4, 16) int64 | noise seed. **row 0은 원래 noise**(−1)이고, row r≥1은 `torch.Generator().manual_seed(seed·10⁵ + call·100 + r)`로 뽑은 (1,50,32) 표준 정규입니다 |
| `mn_chunk` | (4, 16, 50, 32) f32 | 출력 chunk (정규화 공간) |
| `mn_act` | (4, 16, 50, 7) f32 | 같은 chunk의 env action (후처리) |
| `mn_exp_last` | (4, 16, 1024) f16 | 마지막 step의 expert 마지막 층(17) 출력 token 평균 |

row 0은 본 pass와 같은 입력입니다. 다만 batch 구성이 달라 수치가 조금 다릅니다 (차이 median ≈0.03). gripper가 경계에 있으면 드물게 부호가 뒤집힙니다.

### meta

json 문자열입니다. 들어 있는 것:
- 에피소드, seed, 지시문
- 층·step 목록, `pca_dims`, `raw` 여부
- probe anchor
- 확인 수치(`info`)

## pilot 파일

- **held-out 12 ep (`pca_fit=0`):** raw 벡터가 그대로 남아 있습니다. 위 키에 더해 다음이 있습니다.
  - `vlm_tok` (C, 7, N, 2048): 층 17 때문에 float32
  - `exp_tok` (C, 6, 3, 50, 1024)
  - `vlm_vec`, `exp_mlp_mean`(18층)
  - PCA 복원 오차를 직접 잴 수 있습니다.
- **fit 48 ep:** raw `vlm_tok`·`exp_tok`는 지웠습니다 (디스크 절약). 나머지는 held-out과 같습니다.

## pca_bases.npz

키는 `{stream}/L{층}/{mean, components, explained_variance, total_variance, n_samples}`입니다.
- `components`는 (k_max, D)이고, 행이 주성분입니다. 분산이 큰 순서입니다.
- stream별 표본과 k_max:

| stream | 표본 | k_max |
|---|---|---|
| `vlm_tok` | 비패딩 언어 token | 256 |
| `exp_tok` | call × step × token | 128 |
| `k_tok`, `v_tok` | 언어 token | 256 (전체) |
| `vlm_seg` | call × 세그먼트 | 512 |

- 본 추출에서 투영에 쓴 것은 `vlm_tok`(256)과 `exp_tok`(128)뿐입니다. 나머지는 차원 축소가 가능한지 보려고 맞췄습니다.

held-out 설명 비율은 `pca_report.json`의 `heldout_ratio`에 있습니다. 결과는 아래 "PCA" 절에 있습니다.

## weights.npz (float32, PyTorch Linear 배치 (out, in), y = x @ W.T + b)

| 키 | shape | 내용 |
|---|---|---|
| `vlm/L{i}/k_proj`, `vlm/L{i}/v_proj` | (256, 2048) | VLM K/V 투영 (bias 없음) |
| `vlm/L{i}/input_layernorm` | (2048,) | Gemma RMSNorm 가중치. x̂·(1 + w) |
| `exp/L{i}/q_proj` | (2048, 1024) | expert query (8 head × 256) |
| `exp/L{i}/k_proj`, `exp/L{i}/v_proj` | (256, 1024) | expert 자신의 K/V |
| `exp/L{i}/input_layernorm_dense_w`, `_b` | (3072, 1024), (3072,) | adaRMS. `scale, shift, gate = chunk(dense(cond), 3)` (cond = 시간 임베딩), `y = x̂·(1 + scale) + shift`, gate는 residual에 곱함 |
| `action_out_proj_w`, `_b` | (32, 1024), (32,) | expert 최종 출력(final norm 후) → velocity |
| `meta` | json | eps, rope theta, 메모 |

## 결과 요약

(`summary.json`과 아래 수치는 [DONE] 시점입니다)

### 규모
- **에피소드:** 1,560개 (51,100 call). 모두 추출했습니다.
- **디스크:** 87 GB (사람 승인 95 GB 이내).
  - 처음 상한은 60 GB였습니다. spec 키를 그대로 두면 넘기 때문에 A안(80 GB)이 승인됐습니다.
  - 그 뒤 vlm_tok_pc의 fp32 저장 때문에 95 GB로 다시 승인받았습니다.
- **GPU:** ≈1.8 GPU-h (VGI-102 GPU 0)
  - 본 추출 1.44 h, pilot 0.15 h, weights 0.03 h, smoke ≈0.2 h
  - 본 추출 속도: 28-call 에피소드 3.0 s, 40-call 3.8 s
- **fp32로 저장한 배열** (fp16 범위 초과):
  - `vlm_tok_pc`: 1,096/1,560 파일 (층 17 계수가 최대 ≈1e5)
  - raw `vlm_tok`: pilot 20 파일
  - `vlm_seg`는 최대 5.2e4라서 모두 fp16입니다.

### 이전 추출(T021/T022)과의 일치
- **basket_far (원래 4090에서 돌림, 300 ep):** 모든 기존 키가 **bit 단위로 같습니다** (act diff 0, r = 1.0).
- **A100 세트 (1,260 ep):** act_pred를 뺀 기존 키의 Pearson r이 모두 ≥ 0.99982입니다 (중앙값 ≥ 0.99996).
  - 기계가 달라 bit 일치는 기대하지 않았습니다.
- **act_pred:**
  - 이전 추출과의 차이는 중앙값 0.007입니다.
  - 드물게 한 call에서 gripper가 ±1 경계를 넘어 부호가 뒤집힙니다 (r 최소 0.972).
  - 기록된 action과 비교하면, call의 99.9%가 0.05 이내입니다. 이전 추출에도 같은 flip이 있습니다.
- **multi-noise probe:** row 0(원래 noise)과 본 pass의 차이는 중앙값 0.028, p90 0.087입니다.
  - 5%는 > 0.5인데, 모두 gripper flip입니다.
  - probe call이 grasp 전후라서 gripper 경계에 있는 경우가 많습니다.

### PCA 설명 분산 비율

pilot 48 ep로 맞췄고, held-out 12 ep에서 쟀습니다. 모든 층에서 분산의 몇 %를 잡는지입니다.

| stream (k) | 층별 held-out 비율 |
|---|---|
| `vlm_tok` (256) — 저장함 | L0 0.999 · L2 0.987 · L5 0.979 · L8 0.974 · L11 0.957 · L14 0.908 · L17 0.993 |
| `exp_tok` (128) — 저장함 | L2 1.000 · L5 0.999 · L8 0.994 · L11 0.981 · L14 0.921 · L17 0.937 |
| `k_tok` (128) — 참고 | L0 0.995 · L2 0.961 · L5 0.961 · L8 0.941 · L11 0.934 · L14 0.959 · L17 0.921 |
| `v_tok` (128) — 참고 | L0 0.991 · L2 0.915 · L5 0.881 · L8 0.817 · L11 0.793 · L14 0.878 · L17 0.962 |
| `vlm_seg` (512) — 참고 | 전체 0.995–0.9999 |

**주의 1 — vlm_seg는 raw로 저장했습니다.**
- 전체 분산은 k=8에서도 0.996이지만, 이는 세그먼트 간 차이(이미지 / BOS / 문장)와 massive dim이 지배하기 때문입니다.
- 같은 세그먼트 안의 변화(세그먼트별 평균을 뺀 분산)는 L5–L11에서 k=256이어도 0.60–0.83만 잡힙니다.

**주의 2 — vlm_tok PCA도 같은 이유로 조심해야 합니다.**
- vlm_tok의 PCA도 token 종류 간 차이가 분산의 상당 부분입니다.
- 같은 token 위치에서 call 사이의 미세한 변화는 위 비율보다 덜 보존될 수 있습니다.
- 확인하려면 held-out pilot 12 ep(`pca_fit=0`)의 raw `vlm_tok`을 쓰십시오.
