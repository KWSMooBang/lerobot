"""Generic benchmark worker loop (runs in the benchmark's venv; see adapters.py for the protocol).

    run_worker(run_unit)    # run_unit(unit: dict, client: PolicyClient) -> int (episodes written)

The policy-server URL comes from $SC_POLICY_URL. stdout is reserved for `@@result` lines: everything the
adapter prints goes to stderr (the worker log).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from collections.abc import Callable
from typing import Any

from .client import PolicyClient
from .protocol import ORIENT_UPRIGHT

RESULT_PREFIX = "@@result "


def run_worker(
    run_unit: Callable[[dict[str, Any], PolicyClient], int], orientation: str | None = None
) -> None:
    proto = sys.stdout
    sys.stdout = sys.stderr  # library prints → log
    url = os.environ["SC_POLICY_URL"]
    client: PolicyClient | None = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        unit = json.loads(line)
        if unit.get("quit"):
            break
        res: dict[str, Any] = {"unit_id": unit["unit_id"]}
        try:
            if client is None:
                client = PolicyClient(url, orientation=orientation or ORIENT_UPRIGHT)
            res.update(status="done", episodes=run_unit(unit, client))
        except Exception as e:
            traceback.print_exc()
            res.update(status="failed", error=f"{type(e).__name__}: {e}"[:2000])
            if client is not None:  # reconnect next time (the server may have dropped us)
                with contextlib.suppress(Exception):
                    client.close()
                client = None
        proto.write(RESULT_PREFIX + json.dumps(res) + "\n")
        proto.flush()
    if client is not None:
        client.close()
