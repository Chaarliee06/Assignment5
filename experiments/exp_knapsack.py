"""Experimentos de carga útil (mochila con kits incompatibles).

kaggle      las 10 000 instancias de Knapsack Problem Data: BT, B&B (DFS y mejor-primero),
            MITM y DP deben reproducir "Best price".
backtrack   backtracking con máscaras de bits, n ∈ {10,…,30}, con y sin cota.
bnb         B&B con cota fraccional, n ∈ {30, 60, 90, 120}; GA (10 semillas) como base
            bioinspirada; verificación con HiGHS.
mitm        meet-in-the-middle para cargar lo más cerca posible de W (n = 40) y su
            escalamiento 2^{n/2}.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from drones.datasets import kit_pool, load_knapsack_kaggle
from drones.knapsack import (KnapInstance, knap_backtracking, knap_bnb, knap_dp, knap_ga,
                             knap_milp, knap_mitm, make_instance)
from drones.metrics import peak_memory, write_csv

from .common import KNAPSACK, RES

N_INST = 10


def parte_kaggle():
    df = load_knapsack_kaggle(str(KNAPSACK))
    rows = []
    totals = {k: 0.0 for k in ("bt", "bnb", "bnb_best", "mitm", "dp")}
    mism = {k: 0 for k in totals}
    picks_ok = 0
    for r in df.itertuples():
        inst = KnapInstance(r.weights.astype(np.int64), r.prices.astype(np.int64), int(r.capacity), [])
        sel_gt = np.flatnonzero(r.best_picks)
        if inst.feasible(sel_gt) and inst.value(sel_gt) == int(r.best_price):
            picks_ok += 1
        for key, fn in (("bt", lambda: knap_backtracking(inst)["value"]),
                        ("bnb", lambda: knap_bnb(inst)["value"]),
                        ("bnb_best", lambda: knap_bnb(inst, "best")["value"]),
                        ("mitm", lambda: knap_mitm(inst)["value"]),
                        ("dp", lambda: knap_dp(inst.w, inst.v, inst.W))):
            t0 = time.perf_counter()
            val = fn()
            totals[key] += time.perf_counter() - t0
            mism[key] += int(val != int(r.best_price))
    for key in totals:
        rows.append({"metodo": key, "instancias": len(df), "discrepancias": mism[key],
                     "tiempo_total_s": totals[key], "tiempo_medio_us": 1e6 * totals[key] / len(df)})
    rows.append({"metodo": "best_picks_consistentes", "instancias": len(df), "discrepancias": len(df) - picks_ok})
    write_csv(str(RES / "mochila_kaggle.csv"), rows)
    print(rows)


def _pool():
    df = load_knapsack_kaggle(str(KNAPSACK))
    return kit_pool(df)


def parte_backtrack(sizes=(10, 14, 18, 22, 26, 30)):
    pw, pv = _pool()
    rows = []
    for n in sizes:
        for i in range(N_INST):
            inst = make_instance(pw, pv, n, np.random.default_rng(1000 * n + i))
            ref = knap_milp(inst)
            for name, kw in (("BT con cota", {"use_bound": True}), ("BT sin cota", {"use_bound": False})):
                if not kw["use_bound"] and n > 22:
                    continue
                r = knap_backtracking(inst, **kw)
                _, mem = peak_memory(lambda: knap_backtracking(inst, **kw)) if i < 3 else (None, None)
                rows.append({"n": n, "inst": i, "metodo": name, "valor": r["value"], "opt": ref,
                             "ok": r["value"] == ref and inst.feasible(r["sel"]), "nodos": r["nodes"],
                             "t_s": r["time_s"], "mem_kib": mem, "conflictos": len(inst.pairs)})
        print("bt", n, flush=True)
    write_csv(str(RES / "mochila_backtracking.csv"), rows)


def parte_bnb(sizes=(30, 60, 90, 120), seeds=10):
    pw, pv = _pool()
    rows = []
    for n in sizes:
        for i in range(N_INST):
            inst = make_instance(pw, pv, n, np.random.default_rng(2000 * n + i))
            t0 = time.perf_counter()
            ref = knap_milp(inst)
            t_milp = time.perf_counter() - t0
            rows.append({"n": n, "inst": i, "metodo": "HiGHS (MILP)", "valor": ref, "opt": ref,
                         "ok": True, "t_s": t_milp, "conflictos": len(inst.pairs)})
            for name, strat in (("B&B DFS", "dfs"), ("B&B mejor-primero", "best")):
                r = knap_bnb(inst, strat, time_limit=60)
                _, mem = peak_memory(lambda: knap_bnb(inst, strat, time_limit=60)) if i < 3 and r["time_s"] < 20 else (None, None)
                rows.append({"n": n, "inst": i, "metodo": name, "valor": r["value"], "opt": ref,
                             "ok": r["value"] == ref and inst.feasible(r["sel"]), "nodos": r["nodes"],
                             "max_abiertos": r["max_open"], "t_s": r["time_s"], "mem_kib": mem,
                             "estado": r["status"], "conflictos": len(inst.pairs)})
            if n <= 30:
                r = knap_backtracking(inst)
                rows.append({"n": n, "inst": i, "metodo": "BT con cota", "valor": r["value"], "opt": ref,
                             "ok": r["value"] == ref, "nodos": r["nodes"], "t_s": r["time_s"]})
            for s in range(seeds):
                g = knap_ga(inst, np.random.default_rng(s), budget=60 * n)
                rows.append({"n": n, "inst": i, "semilla": s, "metodo": "GA (base bioinspirada)",
                             "valor": g["value"], "opt": ref, "ok": g["value"] == ref and inst.feasible(g["sel"]),
                             "evals": g["evals"], "t_s": g["time_s"],
                             "brecha_pct": 100.0 * (ref - g["value"]) / ref})
        print("bnb", n, flush=True)
    for r in rows:
        r.setdefault("brecha_pct", 100.0 * (r["opt"] - r["valor"]) / r["opt"])
    write_csv(str(RES / "mochila_bnb.csv"), rows)


def parte_mitm(sizes=(20, 24, 28, 32, 36, 40)):
    pw, pv = _pool()
    rows = []
    for n in sizes:
        for i in range(N_INST):
            inst = make_instance(pw, pv, n, np.random.default_rng(3000 * n + i), cap_frac=0.35)
            # pesos "grandes" para que alcanzar W exacto no sea trivial: peso real del kit
            # = 1000·w + valor (gramos), como en un manifiesto de carga
            inst = KnapInstance(1000 * inst.w + inst.v, inst.v, int(0.35 * (1000 * inst.w + inst.v).sum()),
                                inst.pairs)
            t0 = time.perf_counter()
            ref, ref_ok = knap_milp(inst, "weight", time_limit=20, return_status=True)
            t_milp = time.perf_counter() - t0
            r = knap_mitm(inst, "weight")
            _, mem = peak_memory(lambda: knap_mitm(inst, "weight")) if i < 3 else (None, None)
            b = knap_bnb(KnapInstance(inst.w, inst.w, inst.W, inst.pairs), time_limit=5)
            certified = [ref] if ref_ok else []
            if b["status"] == "ok":
                certified.append(b["value"])
            rows.append({"n": n, "inst": i, "W": inst.W, "carga_mitm": r["value"], "carga_milp": ref,
                         "milp_certifica": ref_ok, "t_milp_s": t_milp,
                         "holgura": inst.W - r["value"],
                         "ok": (all(r["value"] == c for c in certified) and inst.feasible(r["sel"])
                                and (ref is None or r["value"] >= ref)),
                         "verificado_por": len(certified),
                         "t_mitm_s": r["time_s"], "mem_mitm_kib": mem, "frontera": r["boundary"],
                         "subconjuntos": r["subsets"], "carga_bnb": b["value"], "t_bnb_s": b["time_s"],
                         "nodos_bnb": b["nodes"], "estado_bnb": b["status"], "conflictos": len(inst.pairs)})
        print("mitm", n, flush=True)
    write_csv(str(RES / "mochila_mitm.csv"), rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", default="todo", choices=["kaggle", "backtrack", "bnb", "mitm", "todo"])
    a = ap.parse_args()
    if a.parte in ("kaggle", "todo"):
        parte_kaggle()
    if a.parte in ("backtrack", "todo"):
        parte_backtrack()
    if a.parte in ("bnb", "todo"):
        parte_bnb()
    if a.parte in ("mitm", "todo"):
        parte_mitm()
