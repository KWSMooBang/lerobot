# sc-train: LIBERO training harness (T011)

Design: `/outputs/VLA-shortcut-exps/tasks/TESTBED_DESIGN.md` (§3 checkpoint registry). `sc-train` wraps
`lerobot-train`: a preset gives the policy and recipe, `--data-variant` changes the training data, and the
finished checkpoint is added to `ckpt_registry.tsv` so `sc-bench --ckpt <id>` can evaluate it.

## Train in 3 lines

```bash
scripts/shortcut/train/sc-train presets                                                 # what exists
scripts/shortcut/train/sc-train --preset sc_template_libero_all --run-id tb-tmpl-a     # GPU 0
scripts/shortcut/train/sc-train --preset smolvla_libero_all --run-id tb-smolvla-a --gpus 0,1,2,3   # A100 server
```

Options: `--steps N` (optimizer updates), `--batch B` (max per-GPU micro-batch), `--effective-batch E`,
`--data-variant ...`, `--variant-eval same|none`, `--out-root DIR` (default `$SC_OUT_ROOT` or
`/outputs/VLA-shortcut-exps`), `--resume`, `--dry-run`, `--wandb auto|online|offline|off`, `--ckpt-id`,
`--no-register`, `--notes`. Anything after `--` goes to `lerobot-train` unchanged (lerobot units: micro-batches),
e.g. `-- --policy.lang_cond=none`.

| file                      | what                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| `sc-train`, `sc_train.py` | CLI: `run` (default), `register`, `presets`                                                           |
| `presets/*.yaml`          | lerobot-train args + per-GPU-count batch/steps + time/memory estimates (comments at the top)          |
| `lerobot_train_entry.py`  | the process that is launched: plugins, data variant, PaliGemma tokenizer fallback, then lerobot-train |
| `data_variants.py`        | `--data-variant` parsing and the task-string wrapper                                                  |
| `registry.py`             | `ckpt_registry.tsv` read/append (flock)                                                               |
| `sc_plugins.py`           | imports `scripts/shortcut/policies/lerobot_policy_*` without `pip install`                            |
| `with_plugins.py`         | any lerobot CLI with the plugins loaded, e.g. `with_plugins.py lerobot_eval --policy.path=...`        |
| `new_policy.sh`           | copy `sc_template` into a new policy + preset                                                         |

### What a run writes

`<out-root>/runs/<run_id>/`:
`run_meta.json` (preset, resolved batch/steps, GPUs, data variant, dataset revision, command, git sha, timings, final loss),
`git_sha`, `cmd.sh` (re-runnable), `train.log`, `loss.tsv` (every logged loss), `wandb_url.txt`,
`train/` = lerobot `output_dir` (`checkpoints/<step>/pretrained_model/{config.json, model.safetensors, train_config.json, processors}`,
`checkpoints/last` symlink), `checkpoints -> train/checkpoints`.
At the end one line goes to `<out-root>/results.tsv` (same columns as the eval runs; `metrics` =
`preset;steps;variant;loss_first;loss_last;ckpt_id`) and one to `<out-root>/ckpt_registry.tsv`.

- Data: the local copy `/outputs/VLA-shortcut-exps/data/hf_libero` (or `$SC_LIBERO_DATA_ROOT`, or
  `<out-root>/data/hf_libero`) when it exists, else the Hub at the pinned revision `86958911`.
- GPUs: `CUDA_VISIBLE_DEVICES=--gpus`. One GPU = plain python; several = `accelerate launch --num_processes N`
  (DDP; every setting stays in lerobot's own config, as `docs/source/multi_gpu_training.mdx` asks).
- Batch: each preset fixes `effective_batch` and `optimizer_updates`; for N GPUs sc-train sets
  batch/GPU = min(`micro_batch`, effective / N) and grad accum = effective / (batch/GPU × N), and stops with an
  error if that does not divide (e.g. pi05 on 3 GPUs). So 1, 2, 4 or 8 GPUs train the same recipe. lerobot's
  `--steps` counts micro-batches, so sc-train passes updates × accum (and scales `save_freq`/`log_freq`,
  which presets give in updates, and the LR schedule the same way).
- In this container (`/queue/locks` exists) the GPU's `<uuid>.lock` is flock'ed for the whole run and only GPU 0
  is accepted (`--allow-any-gpu` to override). No `/harness` queue. Without `/queue` nothing is locked.
- W&B: `auto` uses the online project `VLA-shortcut-exps` (`$SC_WANDB_PROJECT`, entity `$SC_WANDB_ENTITY`) when the
  login works, otherwise logs offline under `train/wandb/`; `wandb_url.txt` says which, and how to `wandb sync`.
- `--resume` continues `checkpoints/last` with the run's own config and data variant on the same number of GPUs
  (`--steps` may raise the total number of updates).
- `--dry-run` prints GPUs, effective batch, optimizer updates, epochs, memory/time estimates and the exact command,
  then lets lerobot parse and validate the config on CPU (no weights, no data).

## Presets

| preset                   | policy      | start from                | 1 GPU (GPU 0, RTX 4090)              | 4×A100-80GB                                                        |
| ------------------------ | ----------- | ------------------------- | ------------------------------------ | ------------------------------------------------------------------ |
| `sc_template_libero_all` | sc_template | scratch (~4 M params)     | 256 × 7.5k updates = 64×1×4, ~0.45 h | 64×4×1, ~0.2 h                                                     |
| `smolvla_libero_all`     | smolvla     | SmolVLM2-500M VLM weights | 256 × 25k = 64×1×4, ~16 h            | 64×4×1, ~3 h                                                       |
| `pi05_libero_all`        | pi05        | `lerobot/pi05_libero`     | does not fit 24 GB                   | 256 × 6k = 32×4×2 (official `pi05_libero_finetuned` recipe), ~12 h |
| `pi0_libero_all`         | pi0         | `lerobot/pi0_base`        | does not fit 24 GB                   | 32 × 30k = 8×4×1 (openpi pi0_libero), ~7 h                         |

Measured / estimated hours are in each yaml header and in `sc-train presets`.

## Data variants (`--data-variant`, independent of the preset)

```bash
sc-train --preset sc_template_libero_all --run-id tb-tmpl-goal-blank --data-variant suites=goal --data-variant instr=blank
```

| item                            | effect                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `suites=spatial,object,goal,10` | only episodes of these suites (sets lerobot `dataset.episodes`; goal = 428 episodes)                                                                                                                                |
| `instr=blank`                   | every task string is `""` (design C9: vision-only training)                                                                                                                                                         |
| `instr=paraphrase`              | each frame's task is a paraphrase from `scripts/shortcut/instructions/<suite>.json`, picked by hash(seed, frame index): the same frame always gets the same sentence (any worker count, process, epoch, `--resume`) |
| `para_review=include`           | also use the 6 paraphrases still flagged for human review (excluded by default)                                                                                                                                     |
| `para_keep=0.3`                 | with `instr=paraphrase`, keep the original sentence 30 % of the time                                                                                                                                                |

Nothing is copied: `data_variants.install` patches lerobot-train's dataset factory in the training process and
switches the dataset object to `VariantLeRobotDataset`, which only rewrites `item["task"]` (in every dataloader
worker). Normalization stats stay those of the full dataset. The variant id (e.g. `suites=goal;instr=blank`) is
recorded in `run_meta.json` and in the registry column `train_variant`.

`--variant-eval same|none` (default `same`, recorded in `run_meta.json` as `variant_eval`) concerns lerobot-train's
held-out split only (`-- --dataset.eval_split=0.05 --eval_steps=...`: offline validation loss on held-out
episodes). `same` = validation on the training distribution, `none` = original instructions. Env-rollout evaluation
never sees the variant (that is sc-bench's job).

## Checkpoint registry

`<out-root>/ckpt_registry.tsv`, columns
`ckpt_id path_or_hub policy_type train_data train_variant steps run_id git_sha created_at notes`.
It starts with `pi05lib` (`lerobot/pi05_libero_finetuned`) and `pi05base` (`lerobot/pi05_base`, needs the LIBERO
adapter). A successful `sc-train` run adds `<run_id>` → `runs/<run_id>/train/checkpoints/<last step>/pretrained_model`.

```bash
sc-train register --init                                                      # create the file with pi05lib, pi05base
sc-train register --run-id tb-tmpl-a [--ckpt-id tmpl-a] [--replace]            # (re-)register a run
sc-train register --ckpt-id act-x --path org/act_libero --policy-type act --notes "..."   # external ckpt
```

## A new policy (your idea)

```bash
scripts/shortcut/train/new_policy.sh my_idea      # copies sc_template -> policies/lerobot_policy_my_idea + presets/my_idea_libero_all.yaml
# edit scripts/shortcut/policies/lerobot_policy_my_idea/src/lerobot_policy_my_idea/modeling_my_idea.py
scripts/shortcut/train/sc-train --preset my_idea_libero_all --run-id tb-my-idea-a --steps 2000
```

The plugin follows `docs/source/bring_your_own_policies.mdx` (out-of-tree `lerobot_policy_<name>` package:
config registered as `<name>`, `<Name>Policy`, `make_<name>_pre_post_processors`). sc-train imports it from source.
For tools that rely on lerobot's own plugin discovery, install it once: `uv pip install --no-deps -e scripts/shortcut/policies/lerobot_policy_<name>`
(`setup_server.sh` does this; `uv sync` removes it again), or run them through `with_plugins.py`.
Where to put ideas: `policies/lerobot_policy_sc_template/README.md`.

Smoke-check a plugin checkpoint in lerobot's own LIBERO eval (1 task × 2 episodes):

```bash
uv run python scripts/shortcut/train/with_plugins.py lerobot_eval --policy.path=<run>/checkpoints/last/pretrained_model \
  --env.type=libero --env.task=libero_goal --env.task_ids='[0]' --eval.n_episodes=2 --eval.batch_size=1 --output_dir=<dir>
```

## A100 server

```bash
git clone -b shortcut-learning-exps <repo> && cd <repo>
SC_OUT_ROOT=/data/sc scripts/shortcut/setup_server.sh --gpus 0,1,2,3     # venv + plugins + data (~35 GB) + weights + preset checks
SC_OUT_ROOT=/data/sc scripts/shortcut/train/sc-train --preset pi05_libero_all --run-id tb-pi05-a --gpus 0,1,2,3
```

`setup_server.sh --skip-train-data` skips the dataset/weights. Run long jobs under `tmux`/`nohup`; `--resume` picks
up after an interruption. Results, runs and the registry all live under `$SC_OUT_ROOT`.
