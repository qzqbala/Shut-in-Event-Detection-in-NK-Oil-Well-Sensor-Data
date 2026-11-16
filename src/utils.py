"""Small helpers: timing / memory measurement and JSON state."""
import json
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime

from .config import STATE_PATH


@contextmanager
def measure(results: dict, name: str):
    """Measure wall time (s) and peak Python memory (MB) of a code block."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[name] = {"time_s": round(elapsed, 4), "peak_mem_mb": round(peak / 1e6, 2)}


def df_memory_mb(df) -> float:
    return round(df.memory_usage(deep=True).sum() / 1e6, 2)


def save_state(state: dict) -> None:
    """Persist program state (last run, chosen options) to JSON."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state, saved_at=datetime.now().isoformat(timespec="seconds"))
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str))


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print("State file is corrupted - starting with an empty state.")
        return {}
