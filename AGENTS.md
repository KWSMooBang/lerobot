# pid4vla — L2 실험·작업 세션 규칙  (CLAUDE.md 는 이 파일을 가리킨다)
이 저장소는 LeRobot 포크다. 연구 작업은 아래 규칙을 먼저 따르고, 코드 구조·개발 규칙은 뒤의 "LeRobot 개발 가이드"를 따른다.

- 명세(/outputs/pid4vla/tasks/<id>.md) 한 건만 한다. 범위를 넓히지 않고, 지금 worktree 밖은 건드리지 않는다.
- 데이터: 원본은 /data (읽기 전용). 내려받기·전처리 결과(HF 데이터셋 캐시 포함)는 /data-work/pid4vla/ 에 쓴다.
  다른 사람 데이터는 /data0·/data1·/data2 경로로 읽기만 하고, 협업 폴더는 컨테이너의 SHARE_RW 에 있는 곳에만 쓴다.
- 의존성이 달라야 하면 이 worktree 의 pyproject.toml 을 고치고 `uv sync --locked --extra <...>` (.venv 는 worktree 마다 따로).
  필요한 extra 만 설치한다 (`--extra all` 은 무겁다).
- torch: 서버 GPU 드라이버는 550 이라 CUDA 12.6 까지만 쓸 수 있다. upstream pyproject 의 pytorch-cu128 index(드라이버 570 이상)는
  이 서버에서 GPU 를 못 쓴다. cu126 index 로 바꾸기 전에는 GPU 학습을 제출하지 않는다. PyPI 기본 torch(CUDA 13 빌드)도 쓰지 않는다.
- 시스템 패키지(apt)는 apt-get 을 직접 쓰지 않는다 (기록이 안 남아 컨테이너를 다시 만들면 사라짐).
  사람이 연 lab 세션: sys-install <패키지> (승인 필요). 이 worktree 의 apt.txt 에 기록되니 함께 커밋한다.
  L2: 설치하지 않고 [NEEDS-HUMAN] <id> apt=<패키지,...> wt=<worktree 경로> 로 요청한다. 설치되면 apt.txt 도 함께 커밋한다.
  (비디오 디코딩·테스트에 ffmpeg 가 필요하다.)
- nvcc·CUDA toolkit·Ubuntu 버전이 달라야 하면 멈추고 [NEEDS-HUMAN].

## 학습 (kind: train)
- 학습·평가는 반드시 `RUN_HOURS=<시간> /harness/launch.sh <run_id> <gpu_수> <명령>`.
  예: `RUN_HOURS=6 /harness/launch.sh 1007-pid-base 1 uv run lerobot-train --config_path=<cfg> --output_dir=$RUN_DIR/train`
  `lerobot-train`·`lerobot-eval`·torchrun·accelerate 를 하네스 밖에서 직접 실행하지 않는다.
- 본 학습 전에 RUN_HOURS=0.5, GPU 1장, 짧은 steps 로 smoke test.
- 체크포인트·산출물은 $RUN_DIR (= /outputs/pid4vla/runs/<run_id>, 하네스가 넣어 줌) 아래에 쓴다 (`--output_dir` 를 $RUN_DIR 아래로).
  W&B 는 하네스가 넣는 WANDB_* 를 쓴다 (`--wandb.enable=true`, project 이름을 따로 정하지 않는다).
- 설정값만 다른 run(lr·seed 등)은 config 파일·CLI 인자로 나눠 한 번에 제출한다. run 마다 코드를 복사하지 않는다.
- 본 학습을 제출하기 전에 커밋한다 (run 의 git_sha 가 코드 버전이 된다).
- 제출하면 기다리지 않는다: 본 학습을 모두 제출하고 git push origin HEAD:exp/<slug> 한 뒤 [SUBMITTED] 를 보내고 끝낸다.
  결과 확인과 keep/discard 판단은 L1 이 한다.
- 학습이 끝나기 전에는 이 worktree 를 지우거나 다른 브랜치로 바꾸지 않는다 (실행 중인 학습과 고정 평가가 이 폴더의 코드와 .venv 를 쓴다).
- run_id: <MMDD>-<slug>-<변형> (예: 1007-pid-kp0.5). 공식 지표는 고정 평가의 metrics.json 과 /outputs/pid4vla/results.tsv.

## 분석 (kind: analysis) · 코드 작업 (kind: code)
- analysis: 끝난 run 의 /outputs/pid4vla/runs/<id>/ 와 W&B 를 읽어 그림·표를 만들고 /outputs/pid4vla/analysis/<task_id>/ 에 둔다.
  GPU 가 필요한 추가 평가도 launch.sh 로 제출하고, 끝날 때까지 기다리지 말고 [SUBMITTED] 로 보고한다.
- code: 학습 없이 코드만 고치고 관련 테스트(`uv run pytest tests/<모듈> -x`)와 `pre-commit run --files <바꾼 파일>` 을 통과시킨 뒤 push 하고 [DONE].

## 보고 (명세의 report-to 세션에 한 줄)
- [SUBMITTED] <id> runs=<run_id들> branch=exp/<slug>   학습을 제출하고 끝낼 때
- [DONE] <id> branch=<브랜치> <한 줄 요약>              학습 없이 끝났을 때
- [FAILED] <id> <이유>  /  [NEEDS-HUMAN] <id> <이유>
- 끝나면 git push origin HEAD:exp/<slug> (main·pid4vla 에 push 금지). RESEARCH.md·EXPERIMENTS.md 는 L1 이 쓴다.
- 승인이 필요한 일은 하지 않고 [NEEDS-HUMAN] 으로 멈춘다.
- 사람이 lab 서버에서 직접 연 세션도 같은 규칙을 따르고, 끝나면 lead_pid4vla 에 [SUBMITTED] 또는 [DONE] 으로 알린다.

---

# LeRobot 개발 가이드 (upstream)

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests (GPU → 하네스로만)
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy**: one ruleset (`[tool.mypy]` in `pyproject.toml`) checks the paths in its `files` setting, except `lerobot.rl`, the vendored `molmoact2_hf_model` and the generated `*_pb2` modules. Run it with `pre-commit run mypy --all-files`. Add type annotations when modifying code.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
