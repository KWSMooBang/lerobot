"""Run a lerobot CLI with the local policy plugins registered (no pip install of the plugin needed).

uv run python scripts/shortcut/train/with_plugins.py lerobot_eval --policy.path=<ckpt>/pretrained_model ...
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sc_plugins import register_local_plugins  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = sys.argv[1].replace("-", "_")
    register_local_plugins()
    sys.argv = [script, *sys.argv[2:]]
    importlib.import_module(f"lerobot.scripts.{script}").main()


if __name__ == "__main__":
    main()
