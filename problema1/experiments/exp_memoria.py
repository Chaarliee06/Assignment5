"""Completa las mediciones de memoria pico (tracemalloc) que faltaban en los CSV.

La memoria se mide en una corrida aparte del mismo método sobre la misma instancia (los
tiempos de los CSV no cambian). Para los métodos que llegaron al límite se repite la misma
búsqueda hasta el mismo número de nodos (sin límite de tiempo), así que la memoria
corresponde al punto en que se detuvo la corrida cronometrada.

HiGHS (MILP) corre en C++ y tracemalloc no ve su memoria: se reporta como no medible.

Uso:  python -m experiments.exp_memoria [prueba] [pequenas] [mochila] [kaggle] [atsp] [rejillas]
(sin argumentos corre todo; va después de los demás experimentos porque completa sus CSV)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from drones.atsp import aco_mmas, atsp_bnb, brute_force, held_karp, nearest_neighbor, tour_cost
from drones.datasets import kit_pool, load_knapsack_kaggle, load_map_sample
from drones.knapsack import KnapInstance, knap_backtracking, knap_bnb, knap_dp, knap_ga, knap_mitm, make_instance
from drones.metrics import peak_memory
from drones.scenario import build_city, build_scenario
from drones.search import astar, ida_star, ucs

from .common import KNAPSACK, RES, hospitals_csv
from .exp_atsp import instance
from .exp_search import dataset_city, load_models, map_splits


def _pairs(term, N, n_queries=20):
    rng = np.random.default_rng(N)
    pairs = []
    while len(pairs) < n_queries:
        a, b = rng.choice(len(term), size=2, replace=False)
        pairs.append((int(term[a]), int(term[b])))
    return pairs


def busqueda_rejillas():
    """IDA* clásico (con TT) en las 3 primeras consultas de cada N."""
    path = RES / "busqueda_rejillas.csv"
    d = pd.read_csv(path)
    for N in sorted(d.N.unique()):
        city, term, _ = build_scenario(int(N), 25, hospitals_csv(), seed=0, city_seed=int(N))
        succ = city.successors()
        pairs = _pairs(term, int(N))
        for q in range(3):
            sel = (d.N == N) & (d.consulta == q) & (d.metodo == "IDA*")
            if not sel.any() or d.loc[sel, "mem_kib"].notna().all():
                continue
            s, g = pairs[q]
            hl = city.h_admissible(g).tolist()
            lim = int(d.loc[sel, "exp"].iloc[0])
            _, mem = peak_memory(lambda: ida_star(succ, s, g, hl, mode="classic", tt=True,
                                                  node_limit=lim + 1, time_limit=None))
            d.loc[sel, "mem_kib"] = mem
            print("IDA*", N, q, round(mem, 1), flush=True)
        d.to_csv(path, index=False)


def busqueda_ida_pequenas():
    """Memoria de A* y las 4 variantes de IDA* en rejillas pequeñas (3 consultas por N)."""
    path = RES / "busqueda_ida_pequenas.csv"
    d = pd.read_csv(path)
    if "mem_kib" not in d:
        d["mem_kib"] = np.nan
    for N in sorted(d.N.unique()):
        city = build_city(int(N), seed=7 * int(N))
        comp = city.largest_component()
        free = np.flatnonzero(comp)
        succ = city.successors()
        rng = np.random.default_rng(int(N))
        queries = [tuple(int(x) for x in rng.choice(free, size=2, replace=False)) for _ in range(10)]
        for q in range(3):
            s, g = queries[q]
            hl = city.h_admissible(g).tolist()
            for m in d[(d.N == N) & (d.consulta == q)].metodo.unique():
                sel = (d.N == N) & (d.consulta == q) & (d.metodo == m)
                lim = int(d.loc[sel, "exp"].iloc[0])
                if m == "A*":
                    fn = lambda: astar(succ, s, g, hl)  # noqa: E731
                else:
                    mode = "cr" if "CR" in m else "classic"
                    tt = "con TT" in m
                    # sin TT la memoria es la pila (O(d)); con TT la tabla satura en ≤ N² nodos
                    cap = min(lim + 1, 300_000 if tt else 100_000)
                    fn = lambda mode=mode, tt=tt, cap=cap: ida_star(  # noqa: E731
                        succ, s, g, hl, mode=mode, tt=tt, node_limit=cap, time_limit=None)
                _, mem = peak_memory(fn)
                d.loc[sel, "mem_kib"] = mem
        print("ida pequeñas", N, flush=True)
    d.to_csv(path, index=False)


def busqueda_prueba():
    """UCS, A*, WA* y A*-θ en los 3 primeros mapas de prueba del dataset."""
    path = RES / "busqueda_heuristica_prueba.csv"
    d = pd.read_csv(path)
    if "mem_kib" not in d:
        d["mem_kib"] = np.nan
    models = load_models()
    files = map_splits()[2]
    for k, f in enumerate(files[:3]):
        name = f.split("/")[-1]
        smp = load_map_sample(f)
        city, s, g = dataset_city(smp, wind_seed=90_000 + k)
        succ = city.successors()
        hl = city.h_admissible(g).tolist()
        fns = {"UCS": lambda: ucs(succ, s, g), "A*": lambda: astar(succ, s, g, hl),
               "WA* (w=1.5)": lambda: astar(succ, s, g, hl, weight=1.5)}
        for key, model in models.items():
            fns[f"A*-θ ({model.name})"] = (lambda model=model: astar(
                succ, s, g, model.heuristic_map(city, g).tolist(), reopen=True))
        for m, fn in fns.items():
            sel = (d.mapa == name) & (d.metodo == m)
            if sel.any():
                _, mem = peak_memory(fn)
                d.loc[sel, "mem_kib"] = mem
    d.to_csv(path, index=False)
    print("prueba: memoria en 3 mapas", flush=True)


def mochila_kaggle():
    """Kaggle: máximo, sobre 200 instancias, de la memoria pico de resolver cada una."""
    df = load_knapsack_kaggle(str(KNAPSACK))
    kpath = RES / "mochila_kaggle.csv"
    k = pd.read_csv(kpath)
    insts = [KnapInstance(r.weights.astype(np.int64), r.prices.astype(np.int64), int(r.capacity), [])
             for r in df.head(200).itertuples()]
    fns = {"bt": knap_backtracking, "bnb": knap_bnb, "bnb_best": lambda i: knap_bnb(i, "best"),
           "mitm": knap_mitm, "dp": lambda i: knap_dp(i.w, i.v, i.W)}
    k["mem_kib"] = np.nan
    for m, fn in fns.items():
        k.loc[k.metodo == m, "mem_kib"] = max(peak_memory(fn, i)[1] for i in insts)
    k.to_csv(kpath, index=False)
    print("kaggle memoria lista", flush=True)


def mochila():
    df = load_knapsack_kaggle(str(KNAPSACK))
    pw, pv = kit_pool(df)
    mochila_kaggle()
    # B&B: GA (semilla 0) y backtracking (n = 30) en las 3 primeras instancias
    bpath = RES / "mochila_bnb.csv"
    b = pd.read_csv(bpath)
    for n in sorted(b.n.unique()):
        for i in range(3):
            inst = make_instance(pw, pv, int(n), np.random.default_rng(2000 * int(n) + i))
            sel = (b.n == n) & (b.inst == i) & (b.metodo == "GA (base bioinspirada)") & (b.semilla == 0)
            if sel.any():
                _, mem = peak_memory(lambda: knap_ga(inst, np.random.default_rng(0), budget=60 * int(n)))
                b.loc[sel, "mem_kib"] = mem
            sel = (b.n == n) & (b.inst == i) & (b.metodo == "BT con cota")
            if sel.any():
                _, mem = peak_memory(lambda: knap_backtracking(inst))
                b.loc[sel, "mem_kib"] = mem
        print("mochila memoria", n, flush=True)
    b.to_csv(bpath, index=False)
    # MITM: memoria del B&B que se usa como comparación (3 primeras instancias)
    mpath = RES / "mochila_mitm.csv"
    mm = pd.read_csv(mpath)
    if "mem_bnb_kib" not in mm:
        mm["mem_bnb_kib"] = np.nan
    for n in sorted(mm.n.unique()):
        for i in range(3):
            inst = make_instance(pw, pv, int(n), np.random.default_rng(3000 * int(n) + i), cap_frac=0.35)
            inst = KnapInstance(1000 * inst.w + inst.v, inst.v, int(0.35 * (1000 * inst.w + inst.v).sum()),
                                inst.pairs)
            sel = (mm.n == n) & (mm.inst == i)
            _, mem = peak_memory(lambda: knap_bnb(KnapInstance(inst.w, inst.w, inst.W, inst.pairs), time_limit=5))
            mm.loc[sel, "mem_bnb_kib"] = mem
    mm.to_csv(mpath, index=False)
    print("mitm memoria lista", flush=True)


def atsp_exacto():
    path = RES / "atsp_exacto.csv"
    d = pd.read_csv(path)
    for K in sorted(d.K.unique()):
        for i in range(2):
            C, _ = instance(100, int(K), seed=i)
            base = (d.K == K) & (d.inst == i)
            sel = base & (d.metodo == "Held–Karp (DP)")
            if sel.any() and d.loc[sel, "mem_kib"].isna().all():
                _, mem = peak_memory(lambda: held_karp(C))
                d.loc[sel, "mem_kib"] = mem
            sel = base & (d.metodo == "Fuerza bruta")
            if sel.any() and d.loc[sel, "mem_kib"].isna().all():
                _, mem = peak_memory(lambda: brute_force(C))
                d.loc[sel, "mem_kib"] = mem
            if K > 10 and i == 0:
                ev = aco_mmas(C, np.random.default_rng(i), 2000, local_search=True)
                for name, ub in (("B&B (MST)", None), ("B&B (MST) + VMC", tour_cost(C, nearest_neighbor(C))),
                                 ("B&B (MST) + ACO+LS", ev.best)):
                    sel = base & (d.metodo == name)
                    if sel.any() and d.loc[sel, "mem_kib"].isna().all():
                        _, mem = peak_memory(lambda: atsp_bnb(C, ub=ub, time_limit=600))
                        d.loc[sel, "mem_kib"] = mem
        print("atsp exacto memoria", K, flush=True)
    d.to_csv(path, index=False)


if __name__ == "__main__":
    import sys
    partes = sys.argv[1:] or ["prueba", "pequenas", "mochila", "atsp", "rejillas"]
    if "prueba" in partes:
        busqueda_prueba()
    if "pequenas" in partes:
        busqueda_ida_pequenas()
    if "mochila" in partes:
        mochila()
    if "kaggle" in partes:
        mochila_kaggle()
    if "atsp" in partes:
        atsp_exacto()
    if "rejillas" in partes:
        busqueda_rejillas()
