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
HOSPITALES = DATOS / "us_hospital_locations.csv"   # US Hospital Locations (Kaggle), archivo original
KNAPSACK = DATOS / "knapsack_5_items.csv"          # Knapsack Problem Data (Kaggle), archivo original
KNAPSACK_NEW = DATOS / "knapsack_5_items_new.csv"  # segundo archivo del mismo dataset

for d in (RES, FIGS):
    os.makedirs(d, exist_ok=True)


def hospitals_csv() -> str:
    """CSV de hospitales de Kaggle (US Hospital Locations)."""
    return str(HOSPITALES)


def snap(comp: np.ndarray, W: int, r: int, c: int) -> int:
    """Celda permitida más cercana (euclidiana) a (r, c)."""
    cand = np.flatnonzero(comp)
    rr, cc = np.divmod(cand, W)
    k = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))
    return int(cand[k])
