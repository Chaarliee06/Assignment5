"""Rutas y utilidades compartidas por los experimentos."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / "datos"
RES = ROOT / "resultados"
FIGS = ROOT / "figuras"
MAPAS_KAGGLE = DATOS / "mapas_kaggle"   # .pt originales del 2D Path Planning Dataset (Kaggle)
MAPAS_REPRO = DATOS / "mapas_2d"        # .npz reproducidos con el generador del autor
MAPAS = MAPAS_KAGGLE if any(MAPAS_KAGGLE.rglob("*.pt")) else MAPAS_REPRO
HOSPITALES = DATOS / "hospitales_covidcaremap.csv"
KNAPSACK = DATOS / "knapsack_5_items.csv"

for d in (RES, FIGS):
    os.makedirs(d, exist_ok=True)


def hospitals_csv() -> str:
    """Usa el CSV original de Kaggle si el usuario lo dejó en datos/, si no el espejo."""
    for name in ("us_hospital_locations.csv", "Hospitals.csv", "hospitals.csv"):
        p = DATOS / name
        if p.exists():
            return str(p)
    return str(HOSPITALES)


def snap(comp: np.ndarray, W: int, r: int, c: int) -> int:
    """Celda permitida más cercana (euclidiana) a (r, c)."""
    cand = np.flatnonzero(comp)
    rr, cc = np.divmod(cand, W)
    k = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))
    return int(cand[k])
