"""Misión de ejemplo de punta a punta (N = 100, K = 25) para el dashboard y el reporte.

1. Ciudad con viento y zonas de exclusión; depósito y clínicas reales de Los Ángeles.
2. Una consulta depósito → clínica más lejana con UCS, A* y A*-θ (nodos expandidos).
3. Carga de un dron con B&B (kits del dataset de Kaggle, pares incompatibles).
4. Turnos de aterrizaje en el hub con MAC+MRV+LCV.
5. Orden de visita: óptimo exacto y la mejor solución de cada metaheurística (semilla 0),
   con los tramos dibujados como caminos de A* sobre la rejilla.
Salida: resultados/mision.json
"""
from __future__ import annotations

import json

import joblib
import numpy as np

from drones.atsp import atsp_exact_milp
from drones.csp import make_csp, solve
from drones.datasets import kit_pool, load_knapsack_kaggle
from drones.knapsack import knap_bnb, make_instance
from drones.scenario import build_scenario, cost_matrix
from drones.search import astar

from .common import KNAPSACK, RES, hospitals_csv
from .exp_atsp import BUDGET, METODOS

N, K = 100, 25


def main():
    city, term, info = build_scenario(N, K, hospitals_csv(), seed=0, city_seed=N)
    succ = city.successors()
    C = cost_matrix(city, term)
    ex = atsp_exact_milp(C)
    tours = {"Óptimo (HiGHS)": [0] + ex["perm"].tolist() + [0]}
    costs = {"Óptimo (HiGHS)": ex["cost"]}
    for name, fn in METODOS.items():
        ev = fn(C, np.random.default_rng(0), BUDGET)
        tours[name] = [0] + ev.best_perm.tolist() + [0]
        costs[name] = ev.best

    def legs(order):
        cells = []
        for a, b in zip(order[:-1], order[1:]):
            s, g = int(term[a]), int(term[b])
            r = astar(succ, s, g, city.h_admissible(g).tolist())
            cells.append(r.path)
        return cells

    routes = {k: legs(v) for k, v in tours.items()}
    # consulta de ejemplo: depósito → clínica con mayor costo
    far = int(np.argmax(C[0]))
    s, g = int(term[0]), int(term[far])
    hl = city.h_admissible(g).tolist()
    model = joblib.load(RES / "modelos" / "heuristica_mse.joblib")
    hm = model.heuristic_map(city, g)
    searches = {}
    for name, fn in (("UCS", lambda: astar(succ, s, g, None, keep_closed=True)),
                     ("A*", lambda: astar(succ, s, g, hl, keep_closed=True)),
                     ("A*-θ (MSE)", lambda: astar(succ, s, g, hm.tolist(), reopen=True, keep_closed=True))):
        r = fn()
        searches[name] = {"cost": r.cost, "expanded": r.expanded, "path": r.path,
                          "closed": np.flatnonzero(r.extra["closed"]).tolist()}
    # carga útil
    df = load_knapsack_kaggle(str(KNAPSACK))
    pw, pv = kit_pool(df)
    inst = make_instance(pw, pv, 30, np.random.default_rng(42))
    kb = knap_bnb(inst)
    payload = {"w": inst.w.tolist(), "v": inst.v.tolist(), "W": inst.W, "pairs": inst.pairs,
               "sel": kb["sel"], "value": kb["value"], "nodes": kb["nodes"]}
    # turnos en el hub
    csp = make_csp(24, 3, 4, 8, np.random.default_rng(5), window=(1, 3), heavy_frac=0.4, adjacency="8")
    sol = solve(csp, inference="mac", var_order="mrv", val_order="lcv")
    sched = None
    if sol["assignment"] is not None:
        sched = [{"d": d, "pad": csp.decode(a)[0], "slot": csp.decode(a)[1],
                  "win": csp.windows[d], "heavy": csp.heavy[d]} for d, a in enumerate(sol["assignment"])]
    out = {
        "N": N, "K": K,
        "blocked": city.blocked.astype(int).ravel().tolist(),
        "wind": np.round(city.wind[::4, ::4], 3).tolist(),
        "m_min": city.m_min, "m_max": city.m_max,
        "hospitals": [{"name": r.name.title(), "city": r.city.title(), "node": int(r.node),
                       "beds": None if np.isnan(r.beds) else float(r.beds)} for r in info.itertuples()],
        "C": np.round(C, 3).tolist(), "tours": tours, "tour_costs": costs, "routes": routes,
        "query": {"s": s, "g": g, "far": far}, "searches": searches, "payload": payload,
        "csp": {"rows": csp.rows, "cols": csp.cols, "T": csp.T, "status": sol["status"],
                "nodes": sol["nodes"], "schedule": sched},
    }
    with open(RES / "mision.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("misión:", {k: round(v, 2) for k, v in costs.items()})


if __name__ == "__main__":
    main()
