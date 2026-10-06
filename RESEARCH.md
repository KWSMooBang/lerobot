# pid4vla

## 목표
(미정 — 사람과 정한다)

## 가설

## GPU 예산
(미정)

## 백로그
- [code] torch index 를 pytorch-cu128 → cu126 으로 바꾸고 uv.lock 갱신 (서버 드라이버 550 은 cu128 을 못 씀). GPU 학습 전에 먼저 한다.
- [train] 하네스 smoke test: lerobot-train 을 RUN_HOURS=0.5, GPU 1장으로 돌려 $RUN_DIR·W&B·results.tsv 연결 확인.
- 고정 평가 /harness/pid4vla/eval.sh 필요 여부 결정 (사람이 호스트 ~/infra/harness 에 둔다).

## 결정
- 2026-10-06: 연구 브랜치는 `pid4vla`, L2 는 `exp/<slug>` 로 push 한다. upstream 추적용 `main` 은 그대로 둔다.
