# vlm_attn — VLM 자체 self-attention 기록 (T035, 2026-11-27)

pi05(`lerobot/pi05_libero_finetuned`) 이해 담당(VLM, PaliGemma 18층)의 **prefix 안 self-attention**을 기록했습니다.
prefix 토큰(이미지 patch, 지시문, state, template)이 서로에게 주는 attention입니다.
지금까지는 동작 담당이 VLM을 보는 attention(T028 `attn_mass`, `attn_img`)만 있었습니다.

- **입력:** T028 rich_acts와 같은 에피소드 1,560개와 같은 replay 입력입니다 (`/outputs/VLA-shortcut-exps/rich_acts/manifest.csv`의 `obs_path`).
- **계산:** prefix만 forward 했습니다 (denoising 없음, noise 없음).
  - pi05가 원래 쓰는 eager attention의 softmax 뒤 가중치를 hook으로 읽기만 했습니다. 수치는 바꾸지 않았습니다.
  - 모델이 가중치를 **bf16**으로 계산합니다. 그래서 한 query 행의 합이 1에서 최대 ≈1.6e-3 벗어납니다 (아래 확인).
  - prefix mask는 양방향입니다. 모든 prefix 토큰이 모든 prefix 토큰을 봅니다.
- **장비:** VGI-102 RTX 4090, GPU 1
- **코드:** 브랜치 `exp/sc-vlm-attn`
  - 추출 `scripts/shortcut/analysis/vlm_attn_extract.py`
  - 그림 `scripts/shortcut/analysis/vlm_attn_figs.py`
  - 테스트 `scripts/shortcut/tests/test_vlm_attn_extract.py`
- **큰 데이터:** `/data0/woosung/datasets/VLA-shortcut-exps/vlm_attn/` (이 폴더의 `data` 링크)

## 폴더

| 경로 | 내용 |
|---|---|
| `manifest.csv` | 에피소드마다 한 줄. T028 manifest 열 + `goal_pool`, `full`, `path`, `full_path`, `rich_path` |
| `data/ep/<set>/<job>/<episode>.npz` | ①② (1,560개 모두) |
| `data/full/<set>/<job>/<episode>.npz` | ③ (표준 300개만) |
| `data/seg_headmean_all.npz` | 모든 call의 ① head 평균을 한 파일에 모았습니다 (분석용) |
| `data/extract_main.jsonl` | 에피소드마다 확인 수치와 시간 |
| `agg.npz` | 그림에 쓴 집계 (층 평균, 그룹별, 시간축, 사건 정렬) |
| `headline.json` | 본문 수치 |
| `summary.json` | GPU-h, 디스크, 확인 결과 |
| `figures/` | 그림 (아래 "그림 읽는 법") |
| `logs/` | 실행 스크립트와 로그 |

npz는 무손실 deflate 압축입니다. `np.load`로 바로 읽힙니다.

## 세그먼트

T028 `vlm_seg` / `seg_ntok`와 같은 순서, 같은 토큰 분할입니다.

| id | 이름 | 토큰 수 |
|---|---|---|
| 0 | agentview | 256 patch (16×16) |
| 1 | wrist | 256 patch |
| 2 | BOS | 1 |
| 3 | instr (지시문, task) | 에피소드마다 고정 (5–20) |
| 4 | state | call마다 조금씩 다름 (숫자 토큰) |
| 5 | template | "Task", ":", ", State:", ";\nAction: " 등 나머지 |

- 빈 세 번째 카메라(256)와 언어 padding은 뺐습니다.
- 이 둘은 attention mask에서 막혀 있어서 받는 가중치가 정확히 0입니다 (확인 `pad_mass_max = 0`).
- prefix 위치: 카메라 0 = 0–255, 카메라 1 = 256–511, (빈 카메라 512–767), 언어 768부터.

## 기호

- C = call 수 (Long 52, Goal 30, LIBERO-90 40, 나머지 28). call k의 env step은 `t[k] = 10k`.
- 층 0–17, head 0–7.
- **head 평균:** ②③은 8개 head의 **평균**을 저장했습니다. head 합은 평균 × 8입니다.

## ① 구간 × 구간 attention (`ep/…npz`의 `seg_attn`)

| 키 | shape | dtype | 내용 |
|---|---|---|---|
| `seg_attn` | (C, 18, 8, 6, 6) | f32 | [call, 층, head, query 구간, key 구간] |

- 값 = query 구간의 각 토큰이 key 구간 전체에 준 attention을 더하고, query 구간 토큰들에 대해 평균한 것입니다.
- 그래서 **행 합이 1**입니다 (bf16 오차 안에서).
- head 평균: `seg_attn.mean(2)`. 구간 정의가 선형이라 ③에서 다시 만든 값과 같습니다.

## ② 지시문 토큰 ↔ 이미지 patch (`ep/…npz`)

| 키 | shape | dtype | 내용 |
|---|---|---|---|
| `i2p` | (C, 18, n_i, 512) | f16 | 지시문 토큰(query) → patch 512개(key). head 평균 |
| `p2i` | (C, 18, 512, n_i) | f16 | patch 512개(query) → 지시문 토큰(key). head 평균 |
| `instr_tok` | (n_i,) | str | 지시문 토큰 문자열 (sentencepiece, `▁` = 앞 공백) |
| `instr_ids`, `instr_pos` | (n_i,) | int | token id, prefix 위치 |

- patch 번호 p: 0–255 agentview, 256–511 wrist. 카메라 안에서 `row = p % 256 // 16`, `col = p % 16`.
- 이미지 위쪽이 row 0입니다. replay 입력 이미지(`obs_path`의 `image`, 360×360)를 그대로 16×16으로 나눈 것과 맞습니다.
- 지시문 토큰 수 n_i는 에피소드 안에서 고정입니다.

같이 들어 있는 키:
- `seg_ntok` (C, 6): 구간별 토큰 수
- `lang_ids`, `lang_seg`, `lang_valid`, `lang_pos` (C, N): T028과 같은 언어 토큰 정보 (padding −1 / 7 / False / 0)
- `t` (C,)
- `meta`: json (에피소드, 지시문, 확인 수치 `info`)

## ③ 전체 토큰 × 토큰 attention (`full/…npz`, 표준 300 에피소드)

대상: LIBERO-Goal train10 100, libero3 Object 100, libero3 Spatial 100.

| 키 | shape | dtype | 내용 |
|---|---|---|---|
| `full` | (C, 18, V, V) | f16 | head 평균 attention. 행 = query, 열 = key. 유효 토큰만 |
| `full_valid` | (C, V) | bool | 유효 토큰 (V = 에피소드 안 최대; state 토큰 수 때문에 call마다 548–574) |
| `full_seg` | (C, V) | int8 | 구간 id (무효 −1) |
| `full_pos` | (C, V) | int16 | prefix 위치 |
| `full_row`, `full_col` | (C, V) | int8 | 이미지 토큰의 patch 행·열 (글자 토큰 −1) |
| `full_tok_id`, `full_tok` | (C, V) | int32 / str | 글자 토큰의 id / 문자열 (이미지 토큰은 −1 / "") |

- 토큰 순서는 prefix 순서입니다: agentview 256, wrist 256, BOS, 지시문, state, template.
- 무효 위치는 0으로 채웠습니다.

## 읽는 법 (예)

```python
import numpy as np
z = np.load("data/ep/goal/ours-goal-custom-train10/ctl_t00_train_i03.npz")
A = z["seg_attn"].mean(2)                 # (C, 18, 6, 6) head 평균
img2instr = (A[..., 0, 3] + A[..., 1, 3]) / 2   # (C, 18): patch 하나가 지시문 전체에 주는 attention
instr2img = A[..., 3, 0] + A[..., 3, 1]         # (C, 18): 지시문 토큰 하나가 patch 512개 전체에 주는 attention
toks = list(z["instr_tok"]); k = toks.index("▁drawer")
m = z["i2p"][5, 8, k, :256].astype(np.float32).reshape(16, 16)   # call 5, 층 8: 'drawer' → agentview 지도

f = np.load("data/full/goal/ours-goal-custom-train10/ctl_t00_train_i03.npz")
v = f["full_valid"][5]; F = f["full"][5][:, v][:, :, v].astype(np.float32)   # (18, V, V)
```

모든 call의 ① head 평균은 `data/seg_headmean_all.npz` 하나에 있습니다.
- `seg_attn_headmean` (51,100, 18, 6, 6), `seg_ntok`, `ep_index`, `call`
- `episode`: `ep_index` → `manifest.csv`의 `rich_path`

## Goal pool (`manifest.csv`의 `goal_pool`)

T029/T030/T032와 같은 정의입니다 (`layer_pop.py`).
- `train`: 학습 task 10개 × init 10 (condition = train)
- `cand_fallback`: cand31 중 결과가 `trained`, `fail_trained` (지시와 다른 학습 task를 함)
- `cand_new`: cand31 중 결과가 `new`, `new_byproduct` (새 조합을 해냄)
- `cand_other`: 나머지 (`partial`, `fail_new`, `fail_other`)

## 확인

`summary.json`, `data/extract_main.jsonl` (에피소드마다), `logs/repeat_check.json`.

| 확인 | 결과 |
|---|---|
| hook이 관찰만 하는가: VLM 층 출력의 구간 평균(T028 방식 → f16)을 T028 `vlm_seg`와 비교 | basket_far(4090 원본) **298/300** bit 일치, A100 원본 세트 **1,244/1,260** bit 일치. 나머지 18개도 Pearson r ≥ 0.99999998, 다른 원소는 에피소드당 1–몇 개(전체 ~3백만 중)이고 f16 1 ulp 차이 |
| 위 18개의 원인 | 같은 에피소드를 4번 다시 돌리면 우리 쪽 f32 구간 평균끼리도 최대 6e-5 다릅니다 (CUDA `scatter_add`의 덧셈 순서가 비결정적). attention 출력(②)은 4번 모두 bit 일치. 즉 차이는 구간 평균을 내는 축약에서 생기고, 층 출력 자체나 hook 탓이 아닙니다 (`logs/repeat_check.json`) |
| 막힌 key(빈 카메라·padding)가 받는 attention | 정확히 0 |
| head 평균 행 합 = 1 | 최대 오차 1.9e-3 (모델이 가중치를 bf16으로 계산; bf16 상대 정밀도 ≈4e-3). ③ fp16 저장본 1.9e-3 |
| ① 행 합 = 1 (head별) | 최대 오차 2.9e-3 (같은 이유) |
| ③에서 다시 만든 ① = 저장한 ① (head 평균) | 300 에피소드 모두 최대 차이 2.4e-4 (f16 반올림) |

규모:
- **에피소드:** 1,560개 (51,100 call). ③은 300개 (8,600 call).
- **디스크:** ①② 16.2 GB (0.32 MB/call), ③ 77.7 GB (9.25 MB/call). 합 ≈ 94 GB (+ smoke 1.7 GB, + `seg_headmean_all.npz`). 상한 150 GB 이내.
- **GPU:** 본 추출 0.93 h (그중 forward 0.65 h), smoke 2회 ≈0.08 h, 모델 로드·반복 확인 ≈0.1 h → **≈1.1 GPU-h** (상한 2).
- 속도: 28-call 에피소드 1.6 s, 52-call 3 s, ③ 에피소드 ≈2.9 s (압축 쓰기는 별도 thread).
- smoke 첫 판(`data/_smoke_v0_nonblocking_bug`)은 ③의 GPU→CPU 복사가 non-blocking이라 일부 call이 깨졌습니다. 고친 뒤 다시 돌린 것이 `data/_smoke`와 본 추출입니다. 옛 smoke는 지우지 않았습니다.

## 결과 요약

수치: `headline.json`, `agg.npz`. 모두 head 평균, call 평균입니다.

1. **이미지 → 지시문은 층 0에서만 0이고, 층 1부터 거의 일정합니다.**
   - patch 하나가 지시문 전체에 주는 몫: 층 0 **0.0008** → 층 1 **0.11**, 이후 층 1–17에서 0.05–0.14 (봉우리 층 6 0.12, 층 9 0.14, 층 12 0.14; Spatial 층 12 0.18).
   - 깊어질수록 커지지 않습니다. 따라서 T032의 "깊은 층일수록 지시를 바꾸면 달라지는 feature가 많다"는 이미지→지시문 attention 몫이 깊은 층에서 커져서가 아닙니다. 층 1부터 매 층 비슷한 몫으로 읽은 것이 residual에 쌓인 것과 맞습니다.
   - 반대 방향(지시문 → 이미지 512 patch)은 얕은 층에서 큽니다: 층 1–8에서 0.21–0.43, 층 9–17에서 0.10–0.26. 지시문 토큰은 앞쪽 층에서 이미지를 많이 읽고 뒤쪽 층에서는 BOS로 갑니다.
   - 지시문 → 이미지는 wrist 쪽이 agentview보다 큽니다 (→ wrist 0.09–0.31, → agentview 0.02–0.20). 이미지 → 지시문은 두 카메라가 비슷합니다.
2. **BOS가 attention sink입니다. 층 14–16에서 가장 강합니다.**
   - 전체 유효 토큰 평균 BOS 몫: 층 0 0.19, 층 1–3 0.37–0.43, 층 4–12 0.15–0.43 (층 11 0.43), **층 13–16 0.48–0.69 (층 15 0.69)**, 층 17 0.29.
   - BOS 자신은 층 0–15에서 거의 자기만 봅니다 (0.67–0.93). 층 16–17에서는 이미지를 봅니다 (층 17: agentview 0.47, wrist 0.45).
3. **세트·Goal pool 사이 층 프로필은 거의 같습니다** (`2_layer_profile.png`의 선이 겹침). 차이는 작습니다.
   - Goal train은 깊은 층의 이미지 → 지시문이 낮습니다 (층 16: 0.044 vs Spatial 0.10).
   - Goal cand-NEW / cand-FALLBACK / train 사이 차이는 층마다 ≤0.03 정도이고 일관된 순서가 없습니다.
4. **에피소드 안 시간 변화는 작습니다.**
   - 층 8–14 이미지 → 지시문 평균, 첫 call → 마지막 call: Object 0.108 → 0.131, Spatial 0.127 → 0.137, basket 0.11 → 0.12 (오름); Goal train 0.094 → 0.090, cand-NEW 0.087 → 0.079, cand-FALLBACK 0.110 → 0.103 (그대로 또는 조금 내림).
   - 잡기 정렬(`3_img2instr_aligned_grasp.png`): 층 4·16에서 잡기 직후 몇 call 동안 오릅니다 (층 16 ≈0.06 → 0.10, Object·Spatial·basket). Goal은 거의 평평합니다.
5. **② 예 (`4_i2p_overlay_*`):** "cream cheese" 토큰은 층 8 call 0에서 크림치즈 상자 patch를 봅니다 (max 0.11, agentview 합 0.32). 잡은 뒤(call 8)에는 agentview 합이 0.05로 떨어집니다. 층 0은 화면 모서리의 고정 patch 하나에 몰립니다 (위치 sink로 보임).

해석은 기초 수준입니다. 더 깊은 분석(head별, ③ token 단위, 지시 바꾸기 쌍 비교)은 이후 작업입니다.

## 그림 읽는 법

모든 값은 head 평균입니다. "몫"은 한 query 토큰의 attention(합 1) 중 그 구간에 간 비율입니다.

**`figures/1_seg_heatmap_all.png`, `1_seg_heatmap_set_<set>.png` — 층별 ① 평균**
- 칸 하나가 층 하나입니다 (3 × 6 = 18층).
- 행 = 보내는 쪽(query) 구간, 열 = 받는 쪽(key) 구간입니다. 각 행의 합은 1입니다.
- 빨간 테두리 4칸이 이미지 ↔ 지시문입니다:
  - (agentview, instr), (wrist, instr) = 이미지 patch가 지시문을 보는 몫
  - (instr, agentview), (instr, wrist) = 지시문 토큰이 이미지를 보는 몫
- BOS 열이 진하면 그 층에서 BOS가 attention sink(쓸 곳 없는 attention을 버리는 곳)로 쓰인다는 뜻입니다.

**`figures/2_layer_profile.png` — 층에 따른 변화, 그룹별 선**
- 1번째 칸: 이미지 → 지시문 (patch 하나가 지시문 토큰 전체에 주는 몫)
- 2번째 칸: 지시문 → 이미지 (지시문 토큰 하나가 patch 512개 전체에 주는 몫)
- 3번째 칸: BOS 몫 (모든 유효 토큰의 평균, 토큰 수 가중)
- 4번째 칸: 이미지 → BOS
- 선이 겹치면 그룹(suite, Goal pool, 바구니) 사이 차이가 작다는 뜻입니다.

**`figures/2b_per_head.png` — 층 × head**
- 세로 = 층, 가로 = head. 어느 head가 이미지 ↔ 지시문 교환을 맡는지 봅니다.

**`figures/3_time_img2instr.png`, `3_time_instr2img.png` — 에피소드 안 시간 변화**
- 칸 하나가 그룹 하나입니다. 가로 = call(env step = 10·call), 세로 = 층, 색 = 몫.
- 같은 그룹 에피소드들의 평균입니다. 뒤쪽 call은 에피소드 수가 같습니다 (call 수가 그룹 안에서 고정).

**`figures/3_<양>_aligned_grasp.png`, `…_aligned_release.png` — 잡기·놓기 기준 정렬**
- 가로 = (call − 잡기 call) 또는 (call − 놓기 call). 0이 사건 순간입니다 (점선).
- 칸 = 층 0, 4, 8, 12, 16, 17. 선 = 그룹. 범례 괄호 안은 사건이 있는 에피소드 수입니다.
- 사건 call: T025 events (`grasp_call`, `release_call`). 5개 미만 에피소드인 점은 그리지 않았습니다.

**`figures/4_i2p_overlay_<task>_<명사>.png` — ② 예: 물체 이름 토큰이 보는 곳**
- 행 = 층 0 / 8 / 16, 열 = call (처음, 잡기 직전, 놓기, 마지막).
- agentview 이미지 위에 지시문의 물체 명사 토큰(예: "cream", "cheese")이 각 patch에 준 attention을 겹쳤습니다 (그 토큰들 합).
- 밝을수록 그 patch를 많이 봅니다. 색 범위는 칸마다 다릅니다. 제목의 max / total로 크기를 비교하십시오.
- total = agentview 256 patch 전체에 간 몫입니다.

**`figures/5_full_matrix_example.png` — ③ 예**
- Goal train 에피소드 하나, 잡기 call 하나의 전체 행렬입니다. 칸 = 층.
- 행 = query 토큰, 열 = key 토큰. 색은 log scale (1e-5 – 1)입니다.
- 하늘색 선이 구간 경계입니다 (agentview | wrist | BOS | 지시문 | state | template).
- BOS 열 하나가 밝은 세로줄로 보이면 sink입니다. 이미지 블록 안의 대각선은 자기 자신·이웃 patch를 보는 것입니다.
