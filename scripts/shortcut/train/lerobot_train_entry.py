"""`lerobot-train` with the shortcut-testbed hooks, launched by sc_train.py (plain python or accelerate launch).

Before handing argv to `lerobot.scripts.lerobot_train.main()`:
  1. imports the local policy plugins in scripts/shortcut/policies/lerobot_policy_*/src (no pip install needed);
  2. installs the data variant from $SC_DATA_VARIANT (+ $SC_VARIANT_EVAL for the held-out split);
  3. if the gated `google/paligemma-3b-pt-224` tokenizer cannot be loaded, falls back to the local
     sentencepiece copy that common.py uses (pi0 / pi05 processors).
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_variants import DataVariant, install  # noqa: E402
from sc_plugins import register_local_plugins  # noqa: E402

PALIGEMMA = "google/paligemma-3b-pt-224"


def patch_paligemma_tokenizer() -> None:
    import transformers

    orig = transformers.AutoTokenizer.from_pretrained

    def from_pretrained(name: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return orig(name, *args, **kwargs)
        except Exception:
            if str(name) != PALIGEMMA:
                raise
            sys.path.insert(0, str(HERE.parent))
            from common import TOKENIZER_DIR, USER_TOKENIZER_DIR
            from transformers import GemmaTokenizer

            for d in (Path(TOKENIZER_DIR), USER_TOKENIZER_DIR):
                if (d / "tokenizer.model").exists():
                    print(f"[sc-train] {PALIGEMMA} is gated here; using {d}", flush=True)
                    return GemmaTokenizer.from_pretrained(str(d), add_bos_token=True, add_eos_token=False)
            raise

    transformers.AutoTokenizer.from_pretrained = from_pretrained


def check_config() -> None:
    """`sc-train --dry-run`: parse + validate the lerobot-train arguments, print the resolved config, no training."""
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig

    @parser.wrap()
    def _check(cfg: TrainPipelineConfig) -> None:
        cfg.validate()
        d = cfg.to_dict()
        pol = d.get("policy") or {}
        keep = (
            "type",
            "pretrained_path",
            "dtype",
            "chunk_size",
            "n_action_steps",
            "empty_cameras",
            "normalization_mapping",
            "gradient_checkpointing",
            "compile_model",
            "optimizer_lr",
            "scheduler_warmup_steps",
            "scheduler_decay_steps",
            "scheduler_decay_lr",
        )
        print(
            "[sc-train] lerobot config OK:",
            json.dumps(
                {
                    "policy": {k: pol.get(k) for k in keep if k in pol},
                    "batch_size": d["batch_size"],
                    "steps": d["steps"],
                    "seed": d["seed"],
                    "accelerator": {
                        "mixed_precision": d["accelerator"]["mixed_precision"],
                        "gradient_accumulation": d["accelerator"]["gradient_accumulation"],
                    },
                    "parallelism": d["parallelism"],
                    "optimizer": d["optimizer"],
                    "scheduler": d["scheduler"],
                    "dataset": {k: d["dataset"][k] for k in ("repo_id", "root", "revision", "episodes")},
                },
                default=str,
                indent=1,
            ),
            flush=True,
        )

    _check()  # type: ignore[call-arg,unused-ignore]  # parser.wrap fills cfg from argv


def main() -> None:
    register_local_plugins()
    if os.environ.get("SC_CHECK_CONFIG") == "1":
        check_config()
        return
    install(
        DataVariant.from_json(os.environ.get("SC_DATA_VARIANT")), os.environ.get("SC_VARIANT_EVAL", "same")
    )
    patch_paligemma_tokenizer()
    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
