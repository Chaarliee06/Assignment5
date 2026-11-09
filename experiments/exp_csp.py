"""Experimentos del CSP de plataformas y turnos.

transicion  hub de 3×4 plataformas (adyacencia de 8), T = 8 turnos, D creciente: muestra
            la región difícil cerca de la capacidad (4 drones por turno).
escala      hub de 4×5 plataformas (adyacencia de 4), T = 12, D ∈ {20,…,100}.
Variantes: BT, BT+MRV, FC, FC+MRV, FC+MRV+LCV, AC-3+FC+MRV+LCV y MAC+MRV+LCV;
min-conflicts (10 semillas) como búsqueda local incompleta.
"""
from __future__ import annotations

import argparse

import numpy as np

from drones.csp import make_csp, min_conflicts, solve
from drones.metrics import peak_memory, write_csv

from .common import RES

VARIANTES = [
    ("BT", dict(inference="none", var_order="static", val_order="natural")),
    ("BT+MRV", dict(inference="none", var_order="mrv", val_order="natural")),
    ("FC", dict(inference="fc", var_order="static", val_order="natural")),
    ("FC+MRV", dict(inference="fc", var_order="mrv", val_order="natural")),
    ("FC+MRV+LCV", dict(inference="fc", var_order="mrv", val_order="lcv")),
    ("AC-3+FC+MRV+LCV", dict(inference="fc", var_order="mrv", val_order="lcv", preprocess_ac3=True)),
    ("MAC+MRV+LCV", dict(inference="mac", var_order="mrv", val_order="lcv")),
]


def run_family(name, Ds, rows_cols, T, window, heavy, adjacency, variants, n_inst=10,
               node_limit=100_000, time_limit=5.0, mc_seeds=10, mc_steps=1000):
    rows = []
    for D in Ds:
        for i in range(n_inst):
            csp = make_csp(D, rows_cols[0], rows_cols[1], T, np.random.default_rng(7919 * D + i),
                           window=window, heavy_frac=heavy, adjacency=adjacency)
            statuses = {}
            for vname, kw in variants:
                r = solve(csp, node_limit=node_limit, time_limit=time_limit, **kw)
                mem = None
                if i == 0:
                    _, mem = peak_memory(lambda: solve(csp, node_limit=node_limit, time_limit=time_limit, **kw))
                statuses[vname] = r["status"]
                rows.append({"familia": name, "D": D, "inst": i, "metodo": vname, "estado": r["status"],
                             "valida": r.get("valid", None), "nodos": r["nodes"],
                             "retrocesos": r["backtracks"], "revisiones": r["revisions"],
                             "podados": r["pruned"], "t_s": r["time_s"], "mem_kib": mem,
                             "capacidad": None})
            for s in range(mc_seeds):
                m = min_conflicts(csp, np.random.default_rng(s), mc_steps)
                mem = None
                if i == 0 and s == 0:
                    _, mem = peak_memory(lambda: min_conflicts(csp, np.random.default_rng(s), mc_steps))
                rows.append({"familia": name, "D": D, "inst": i, "semilla": s, "metodo": "Min-conflicts",
                             "estado": "sat" if m["status"] == "sat" else "fallo",
                             "valida": m.get("valid", None), "nodos": m["steps"], "t_s": m["time_s"],
                             "mem_kib": mem})
            # coherencia: los métodos completos que terminaron deben coincidir
            done = {v for v in statuses.values() if v in ("sat", "unsat")}
            assert len(done) <= 1, (D, i, statuses)
        print(name, D, flush=True)
    return rows


def main():
    rows = run_family("transicion", (12, 16, 20, 22, 24, 26, 28, 30, 32), (3, 4), 8, (1, 3), 0.4, "8",
                      VARIANTES)
    write_csv(str(RES / "csp_transicion.csv"), rows)
    var_b = [v for v in VARIANTES if v[0] in ("BT+MRV", "FC+MRV", "FC+MRV+LCV", "MAC+MRV+LCV")]
    rows_b = run_family("escala", (20, 40, 60, 80, 100), (4, 5), 12, (2, 4), 0.3, "4", var_b,
                        mc_seeds=10, mc_steps=1500)
    write_csv(str(RES / "csp_escala.csv"), rows_b)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
