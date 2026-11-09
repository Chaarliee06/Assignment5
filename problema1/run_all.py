"""Corre todos los experimentos del Problema 1 y regenera figuras, tablas y panel.

    python run_all.py            # todo (≈ 1.5 h en 2 núcleos)
    python run_all.py --rapido   # sólo figuras, tablas y panel a partir de resultados/
"""
from __future__ import annotations

import argparse
import subprocess
import sys

PASOS = [
    ["-m", "experiments.exp_search", "--parte", "validacion"],
    ["-m", "experiments.exp_search", "--parte", "entrenar"],
    ["-m", "experiments.exp_search", "--parte", "prueba"],
    ["-m", "experiments.exp_search", "--parte", "rejillas"],
    ["-m", "experiments.exp_search", "--parte", "ida_pequenas"],
    ["-m", "experiments.exp_knapsack", "--parte", "todo"],
    ["-m", "experiments.exp_csp"],
    ["-m", "experiments.exp_atsp", "--parte", "exacto"],
    ["-m", "experiments.exp_atsp", "--parte", "meta"],
    ["-m", "experiments.exp_mision"],
    ["-m", "experiments.exp_memoria"],   # memoria pico que faltaba (completa los CSV)
]
SALIDAS = [
    ["-m", "experiments.make_figures"],
    ["-m", "experiments.make_tables"],
    ["-m", "experiments.make_dashboard"],
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args()
    for paso in ([] if a.rapido else PASOS) + SALIDAS:
        print(">>", " ".join(paso), flush=True)
        subprocess.run([sys.executable, *paso], check=True)
