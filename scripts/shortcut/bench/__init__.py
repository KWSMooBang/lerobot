"""Shortcut-robustness benchmark testbed (T010): client library, battery runner, report, benchmark adapters.

`bench.core.protocol`, `bench.core.client` and `bench.core.registry` import only numpy, msgpack and
websockets, so benchmark venvs (LIBERO-Plus, LIBERO-PRO) can put `scripts/shortcut` on sys.path and use them.
"""
