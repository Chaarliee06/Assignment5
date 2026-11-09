"""
comun.py - Utilidades compartidas por todas las partes del Problema 2.

  * rutas de datos, resultados y figuras
  * medicion de tiempo y memoria
  * guardado de figuras y tablas
"""
import json
import os
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, "data")
RES = os.path.join(AQUI, "results")
FIG = os.path.join(AQUI, "..", "reporte", "figuras")
for _d in (RES, FIG):
    os.makedirs(_d, exist_ok=True)

SEMILLAS = list(range(10))          # el enunciado pide al menos 10 semillas

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
COLORES = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def cronometro(f, *args, **kw):
    """Ejecuta f y regresa (resultado, segundos)."""
    t0 = time.perf_counter()
    r = f(*args, **kw)
    return r, time.perf_counter() - t0


def memoria_pico(f, *args, **kw):
    """Ejecuta f con tracemalloc y regresa (resultado, MB pico).
    Se usa en una corrida aparte de la de tiempo, porque tracemalloc hace mas
    lento el codigo."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    r = f(*args, **kw)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return r, pico / 2 ** 20


def guardar_fig(fig, nombre):
    fig.savefig(os.path.join(FIG, nombre + ".pdf"))
    plt.close(fig)


def guardar_csv(df, nombre):
    df.to_csv(os.path.join(RES, nombre + ".csv"), index=False)


def guardar_json(d, nombre):
    with open(os.path.join(RES, nombre + ".json"), "w") as f:
        json.dump(d, f, indent=1, default=_a_json)


def leer_json(nombre):
    with open(os.path.join(RES, nombre + ".json")) as f:
        return json.load(f)


def _a_json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def media_std(serie):
    """Texto 'media ± std' para tablas."""
    return f"{np.mean(serie):.3g} ± {np.std(serie, ddof=1) if len(serie) > 1 else 0:.2g}"


def banda(ax, x, Y, etiqueta, color, log=False):
    """Dibuja media ± desviacion estandar de las filas de Y."""
    Y = np.asarray(Y, float)
    m, s = Y.mean(0), Y.std(0)
    ax.plot(x, m, color=color, label=etiqueta, lw=1.4)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, lw=0)
    if log:
        ax.set_yscale("log")
