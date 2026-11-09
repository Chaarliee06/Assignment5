"""Experimentos del ATSP (orden de visita a las clínicas).

exacto  K ∈ {6, 8, 10, 12}, 10 instancias: fuerza bruta (K ≤ 8), Held–Karp, B&B con cota
        MST sin cota inicial, con vecino más cercano y con la cota de ACO+LS (híbrido).
meta    rejillas N ∈ {50, 100, 150, 200} × K ∈ {12, 25, 50, 100}: SA, Tabu, GA-OX, ACO,
        hiperheurística UCB, GA memético y ACO+LS con 10 semillas y el mismo presupuesto
        de evaluaciones; el óptimo exacto se obtiene con HiGHS (cortes DFJ) para la brecha.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np

from drones.atsp import (aco_mmas, assignment_bound, atsp_bnb, atsp_exact_milp, brute_force,
                         ga_ox, held_karp, memetic_ga, nearest_neighbor, simulated_annealing,
                         tabu_search, tour_cost, ucb_hyperheuristic)
from drones.metrics import peak_memory, write_csv
from drones.scenario import build_scenario, cost_matrix

from .common import RES, hospitals_csv

BUDGET = 30_000
BUDGET_MEM = 5_000      # la memoria pico casi no depende del presupuesto (población/feromona)
SEEDS = 10
METODOS = {
    "SA": simulated_annealing,
    "Tabu": tabu_search,
    "GA-OX": ga_ox,
    "ACO (MMAS)": aco_mmas,
    "UCB-HH": ucb_hyperheuristic,
    "GA memético": memetic_ga,
    "ACO+LS": lambda C, rng, b: aco_mmas(C, rng, b, local_search=True),
}


def instance(N, K, seed=0):
    city, term, info = build_scenario(N, K, hospitals_csv(), seed=seed, city_seed=N)
    return cost_matrix(city, term), info


def parte_exacto(Ks=(6, 8, 10, 12), n_inst=10, N=100):
    rows = []
    for K in Ks:
        for i in range(n_inst):
            C, _ = instance(N, K, seed=i)
            hk, hp, t_hk = held_karp(C)
            base = {"K": K, "inst": i, "opt": hk, "AP": assignment_bound(C)}
            rows.append({**base, "metodo": "Held–Karp (DP)", "costo": hk, "nodos": (1 << K) * K,
                         "t_s": t_hk, "ok": True})
            if K <= 8:
                t0 = time.perf_counter()
                bf, _ = brute_force(C)
                rows.append({**base, "metodo": "Fuerza bruta", "costo": bf, "nodos": math.factorial(K),
                             "t_s": time.perf_counter() - t0, "ok": abs(bf - hk) < 1e-6})
            # B&B sin cota inicial, con vecino más cercano y con ACO+LS (híbrido)
            nn = nearest_neighbor(C)
            t0 = time.perf_counter()
            ev = aco_mmas(C, np.random.default_rng(i), 2000, local_search=True)
            t_aco = time.perf_counter() - t0
            for name, ub, extra in (("B&B (MST)", None, 0.0),
                                    ("B&B (MST) + VMC", tour_cost(C, nn), 0.0),
                                    ("B&B (MST) + ACO+LS", ev.best, t_aco)):
                r = atsp_bnb(C, ub=ub, time_limit=120)
                cost = r["cost"] if r["perm"] is not None else ub
                mem = None
                if i < 2 and K <= 10:
                    _, mem = peak_memory(lambda: atsp_bnb(C, ub=ub, time_limit=120))
                rows.append({**base, "metodo": name, "costo": cost, "nodos": r["nodes"],
                             "podas": r["pruned"], "t_s": r["time_s"] + extra, "mem_kib": mem,
                             "ok": abs(cost - hk) < 1e-6, "estado": r["status"],
                             "cota_inicial": ub if ub is not None else math.nan})
        print("exacto", K, flush=True)
    write_csv(str(RES / "atsp_exacto.csv"), rows)


def _run(task):
    N, K, name, seed, C, measure_mem = task
    fn = METODOS[name]
    t0 = time.perf_counter()
    ev = fn(C, np.random.default_rng(seed), BUDGET)
    dt = time.perf_counter() - t0
    mem = None
    if measure_mem:
        _, mem = peak_memory(lambda: fn(C, np.random.default_rng(seed), BUDGET_MEM))
    assert sorted(ev.best_perm.tolist()) == list(range(1, K + 1))
    out = {"N": N, "K": K, "metodo": name, "semilla": seed, "costo": ev.best, "t_s": dt,
           "evals": ev.evals, "mem_kib": mem, "curva": ev.curve}
    if hasattr(ev, "ucb_history"):
        out["ucb_hist"] = ev.ucb_history
        out["ucb_counts"] = ev.ucb_counts.tolist()
    return out


def parte_meta(Ns=(50, 100, 150, 200), Ks=(12, 25, 50, 100), workers=2, tag=""):
    inst_rows, tasks = [], []
    mats = {}
    for N in Ns:
        for K in Ks:
            C, info = instance(N, K)
            mats[(N, K)] = C
            ex = atsp_exact_milp(C, time_limit=900)
            row = {"N": N, "K": K, "opt": ex["cost"], "estado_opt": ex["status"],
                   "rondas_cortes": ex["rounds"], "t_milp_s": ex["time_s"], "AP": assignment_bound(C),
                   "asimetria": float(np.abs(C - C.T).sum() / C.sum())}
            if K <= 12:
                row["opt_hk"] = held_karp(C)[0]
            inst_rows.append(row)
            print("opt", N, K, ex["status"], round(ex["cost"], 3), round(ex["time_s"], 1), flush=True)
            for name in METODOS:
                for s in range(SEEDS):
                    tasks.append((N, K, name, s, C, s == 0))
    write_csv(str(RES / f"atsp_instancias{tag}.csv"), inst_rows)
    np.savez_compressed(RES / f"atsp_matrices{tag}.npz", **{f"N{N}_K{K}": C for (N, K), C in mats.items()})
    opt = {(r["N"], r["K"]): r["opt"] for r in inst_rows}
    results = []
    if workers <= 1:
        for k, task in enumerate(tasks):
            results.append(_run(task))
            if k % 50 == 0:
                print("meta", k, "/", len(tasks), flush=True)
    else:
        with Pool(workers) as pool:
            for k, out in enumerate(pool.imap_unordered(_run, tasks, chunksize=1)):
                results.append(out)
                if k % 50 == 0:
                    print("meta", k, "/", len(tasks), flush=True)
    rows, curves = [], {}
    for o in results:
        o_opt = opt[(o["N"], o["K"])]
        rows.append({k: v for k, v in o.items() if k not in ("curva", "ucb_hist", "ucb_counts")} |
                    {"opt": o_opt, "brecha_pct": 100.0 * (o["costo"] / o_opt - 1.0)})
        curves.setdefault(f'{o["N"]}|{o["K"]}|{o["metodo"]}', {})[o["semilla"]] = o["curva"]
        if "ucb_hist" in o:
            curves.setdefault(f'UCB|{o["N"]}|{o["K"]}', {})[o["semilla"]] = {
                "hist": o["ucb_hist"], "counts": o["ucb_counts"]}
    rows.sort(key=lambda r: (r["N"], r["K"], r["metodo"], r["semilla"]))
    write_csv(str(RES / f"atsp_meta{tag}.csv"), rows)
    with open(RES / f"atsp_curvas{tag}.json", "w") as fh:
        json.dump(curves, fh)


def unir(tags=("_a", "_b")):
    """Une los resultados de corridas parciales (por ejemplo, por tamaño de rejilla)."""
    import pandas as pd
    for base in ("atsp_instancias", "atsp_meta"):
        parts = [pd.read_csv(RES / f"{base}{t}.csv") for t in tags if (RES / f"{base}{t}.csv").exists()]
        df = pd.concat(parts, ignore_index=True)
        df = df.sort_values([c for c in ("N", "K", "metodo", "semilla") if c in df.columns])
        df.to_csv(RES / f"{base}.csv", index=False)
    curves = {}
    for t in tags:
        p = RES / f"atsp_curvas{t}.json"
        if p.exists():
            curves.update(json.load(open(p)))
    with open(RES / "atsp_curvas.json", "w") as fh:
        json.dump(curves, fh)
    mats = {}
    for t in tags:
        p = RES / f"atsp_matrices{t}.npz"
        if p.exists():
            z = np.load(p)
            mats.update({k: z[k] for k in z.files})
    np.savez_compressed(RES / "atsp_matrices.npz", **mats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", default="todo", choices=["exacto", "meta", "unir", "todo"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--N", type=int, nargs="*", default=[50, 100, 150, 200])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.parte in ("exacto", "todo"):
        parte_exacto()
    if a.parte in ("meta", "todo"):
        parte_meta(Ns=tuple(a.N), workers=a.workers, tag=a.tag)
    if a.parte == "unir":
        unir()
