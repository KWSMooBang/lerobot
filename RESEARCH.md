# pid4vla — RESEARCH

설계 원문: "PoC 실험 설계: LIBERO에서 π0.5·OpenVLA의 지시문 활용을 순간별 PID로 진단"
(https://claude.ai/artifact/1ajFWWSJzastXNaNaTAwhn, 2026-10-09). 아래는 실행용 요약이며, 세부 정의는 원문을 따른다.

## 브랜치 구조
- `main`: upstream lerobot 를 따라가는 fork 기준선. 연구 코드는 넣지 않는다.
- `pid4vla`: 연구 통합 브랜치 (main 53feaef3 에서 분기, 2026-10-19). L2 는 여기서 worktree 를 따고 `exp/<slug>` 로 push,
  리뷰 후 `pid4vla` 로 병합한다.

## 목표
LIBERO 에서 OpenVLA·π0.5 의 결정 순간마다 장면(V) × 지시문(L) 격자를 만들어 순간별 PID(u_L, u_V, s)를 계산하고,
성공률로는 안 보이는 지시문 shortcut 을 진단할 수 있는지 확인한다. 학습·미세조정·실로봇·층별 분석은 범위 밖.
- 양성 대조: LIBERO-Goal (장면 같고 목표만 다름 → 지시문 정보가 커야 함)
- shortcut 예상: LIBERO-Object (장면만으로 과제가 정해짐, LIBERO-Plus 보고)

## 가설 (기준값은 제안, E1 후 조정)
| ID | 가설 | 성공 기준 |
|---|---|---|
| H1 계산 가능성 | 격자 추정이 수렴, 비용이 예산 안 | K=32 에서 bootstrap SE ≤ 0.05 bit, E2 모델당 ≤ 24 GPU-h |
| H2 양성 대조 | Goal 대상 선택 순간에서 u_L+s 가 oracle 요구량에 가까움 | G ≤ 0.3 |
| H3 shortcut 검출 | Object 교차 격자에서 성공률 높아도 u_L+s 낮고 u_V 큼 | G_Object > G_Goal, 95% CI 비중첩 |
| H4 예측 타당성 | 순간 gap Δ_t 가 교차 칸 rollout 의 오물체 실패를 예측 | AUROC ≥ 0.7 |
| H5 모델 비교 (탐색) | π0.5·OpenVLA 의 synergy 비율 σ 가 다름 | 방향만 |

## 단계와 관문
- E0 환경·재현: LIBERO 환경, π0.5 내장 정책, OpenVLA 플러그인, MuJoCo 상태 저장·복원, 장면 변형, 의미 범주 분류기, Goal·Object 성공률 재현(과제당 10회)
  - 관문 1: 두 모델 성공률이 보고값 5%p 이내 (OpenVLA 79.2/88.4, π0.5 LeRobot 98.0/99.0), 같은 상태 복원 시 같은 이미지
- E1 작은 격자: Goal 과제 2 × 순간 5, K=8~64 수렴, 분류기 사람 검증(50샘플), OpenVLA 1스텝 vs 10스텝, 호출 시간 실측
  - 관문 2: SE ≤ 0.05 bit, 분류 일치율 ≥ 90%, 재계산 비용이 예산 안
- E2 본 실험: Goal·Object × 2 모델 × 과제 10 × rollout 10, H2·H3 판정, H5 탐색
  - 관문 3: H2 가 성립해야 H3 해석
- E3 검증: Object 교차 칸 20 × rollout 5 로 H4, measure 교차 확인(I_ccs, BROJA, I_±), shuffle test

## 백로그 (E0 를 L2 로 나누는 안 — 승인 대기)
1. e0-sim (kind: code): LiberoEnv `_env` MuJoCo 상태 저장·복원 + 재렌더링, 장면 변형(물체 위치 교체·유효성 검사), 의미 범주 분류기. 결정성 테스트.
2. e0-pi05 (kind: train): lerobot/pi05_libero_finetuned 로 Goal·Object 성공률 재현 (코드 변경 없음, eval run 만).
3. e0-openvla (kind: train): OpenVLA 를 lerobot_policy_* 플러그인으로 감싸고 (회전·90% crop·역정규화·그리퍼 부호) Goal·Object 체크포인트로 성공률 재현 (greedy·T=1 둘 다).
- 위 셋은 서로 다른 파일을 고치므로 동시 진행 가능 (L2 최대 3).

## 결정
- 2026-10-19: lerobot fork 의 main 은 건드리지 않고 `pid4vla` 브랜치에서 실험을 진행한다.
- 2026-10-19: PoC 설계 문서를 연구 계획으로 채택 (세부 가정은 아래 열린 질문 확정 후 고정).

## 열린 질문 (설계 문서 10절)
- OpenVLA 통합: LeRobot 플러그인(가정) vs OpenVLA 원래 코드에 상태 저장 코드만 공유
- π0.5 체크포인트: lerobot/pi05_libero_finetuned(가정) vs OpenPI 변환본
- Spatial 묶음 포함 여부 (기본: 제외)
- 성공 기준값 그대로 쓸지, 일정

## GPU 예산
- 장비: 이 컨테이너 RTX 4090 24GB × 4 (설계 가정 "모델당 ≥24GB 1장" 충족, OpenVLA bf16 + 배치 32 는 메모리 빠듯할 수 있음 → E1 에서 확인)
- 설계 추정: E2+H4 = OpenVLA ~29 GPU-h + π0.5 ~25 GPU-h ≈ 54 GPU-h, E0·E1 은 수 GPU-h 수준
- 총 예산: (미정 — 사람 또는 main-control [BUDGET])
