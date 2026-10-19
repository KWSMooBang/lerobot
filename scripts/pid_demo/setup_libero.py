"""Write a LIBERO config.yaml (non-interactive) pointing at HF-cached assets.

LIBERO's __init__ calls input() when the config is missing, so run this once before any
`import libero` and export LIBERO_CONFIG_PATH=<dir> for later processes.

    uv run python scripts/pid_demo/setup_libero.py <config_dir>
"""

import importlib.util
import os
import sys

from huggingface_hub import snapshot_download


def main() -> None:
    cfg_dir = sys.argv[1]
    os.makedirs(cfg_dir, exist_ok=True)
    spec = importlib.util.find_spec("libero")
    assert spec is not None and spec.origin is not None
    libero_dir = os.path.join(os.path.dirname(spec.origin), "libero")
    assets = snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset")
    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        f.write(
            f"assets: {assets}\n"
            f"bddl_files: {libero_dir}/bddl_files\n"
            f"benchmark_root: {libero_dir}\n"
            f"datasets: {libero_dir}/../datasets\n"
            f"init_states: {libero_dir}/init_files\n"
        )
    print("wrote", os.path.join(cfg_dir, "config.yaml"), "assets:", assets)


if __name__ == "__main__":
    main()
