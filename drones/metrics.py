"""Utilidades de medición: memoria pico (tracemalloc), agregados y escritura de CSV."""
from __future__ import annotations

import csv
import os
import platform
import sys
import time
import tracemalloc

import numpy as np


def peak_memory(fn, *args, **kwargs):
    """Ejecuta fn y devuelve (resultado, memoria pico en KiB asignada por Python).

    tracemalloc hace más lento el código, por eso el tiempo se mide en una corrida
    aparte sin instrumentar.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        out = fn(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return out, peak / 1024.0


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def mean_std(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def write_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def environment() -> dict:
    import scipy
    import sklearn
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpus": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info
