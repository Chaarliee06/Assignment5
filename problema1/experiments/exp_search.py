"""Experimentos de búsqueda de caminos (inciso 1 del Problema 1).

Partes
------
validacion    UCS/A* contra los caminos de referencia (D* Lite) del 2D Path Planning
              Dataset, sin viento y con el modelo de costos del dataset.
entrenar      entrena la heurística aprendida (MSE y cuantil 0.2) en los mapas del
              dataset con viento.
prueba        mapas de prueba del dataset: A*, A*-θ, A* ponderado; verificación de w·C*.
rejillas      ciudades N ∈ {50, 100, 150, 200} con hospitales reales: UCS, A*, IDA*,
              IDA*-CR, A*-θ y A* ponderado (costo, nodos, b*, brecha, tiempo, memoria).
ida_pequenas  IDA* sin tabla de transposición en rejillas pequeñas (explosión de caminos).

Uso:  python -m experiments.exp_search --parte todo
"""
from __future__ import annotations

import argparse
import json
import math
import time

import joblib
import numpy as np
from scipy.sparse.csgraph import dijkstra

from drones.datasets import list_map_files, load_map_sample
from drones.grid import CityGrid, make_wind
from drones.learned import (FEATURES, LearnedHeuristic, cost_to_go, overestimation_factors,
                            training_rows)
from drones.metrics import peak_memory, write_csv
from drones.scenario import build_city, build_scenario
from drones.search import astar, ebf_of, effective_branching_factor, ida_star, ucs

from .common import MAPAS, RES, hospitals_csv, snap

MODELOS = RES / "modelos"
N_TRAIN = 300
N_TEST = 100
N_VALID = 5000
ALPHA = 0.5


def map_splits():
    """(validación, entrenamiento, prueba) a partir de datos/mapas_2d.

    Con los .pt originales de Kaggle se respetan sus carpetas train/validation/test si
    existen: se entrena con N_TRAIN mapas de train y se prueba con N_TEST de test (muestras
    fijas con semilla 0), y se validan contra D* Lite hasta N_VALID mapas. Con los .npz
    reproducidos (sin carpetas) se usan los primeros N_TRAIN para entrenar y el resto para
    probar."""
    files = list_map_files(str(MAPAS))
    train = [f for f in files if "/train/" in f.replace("\\", "/")]
    test = [f for f in files if "/test/" in f.replace("\\", "/")]
    rng = np.random.default_rng(0)

    def sample(lst, k):
        if len(lst) <= k:
            return list(lst)
        return sorted(rng.choice(lst, size=k, replace=False).tolist())

    if train and test:
        return sample(files, N_VALID), sample(train, N_TRAIN), sample(test, N_TEST)
    if len(files) > N_VALID + 0:
        files_v = sample(files, N_VALID)
    else:
        files_v = files
    return files_v, files[:N_TRAIN], files[N_TRAIN:N_TRAIN + N_TEST]


def dataset_city(sample: dict, wind_seed: int | None):
    """Ciudad a partir de un mapa del dataset. Sin viento (wind_seed=None) se usa el modelo
    de costos del dataset (1 y √2, se permite cortar esquinas, la meta siempre es
    alcanzable); con viento, el modelo del dron."""
    blocked = sample["map"].copy()
    path = sample["path"]
    s = (int(path[0][0]), int(path[0][1]))
    g = (int(path[-1][0]), int(path[-1][1]))
    if wind_seed is None:
        blocked[g] = False
        city = CityGrid(blocked, None, alpha=0.0, corner_cutting=True)
        return city, city.idx(*s), city.idx(*g)
    rng = np.random.default_rng(wind_seed)
    city = CityGrid(blocked, make_wind(blocked.shape[0], rng), alpha=ALPHA)
    comp = city.largest_component()
    W = city.W
    return city, snap(comp, W, *s), snap(comp, W, *g)


def dataset_city_margin(sample: dict, margin: int = 1):
    """Modelo de costos con el que el autor generó el dataset de Kaggle (D* Lite con
    obst_margin = 1, ver dataset/ds_generator.py): además de los obstáculos, se prohíben
    las celdas a distancia de Chebyshev ≤ margin de un obstáculo, salvo las que están a
    distancia euclidiana ≤ margin del inicio o de la meta; la meta siempre es alcanzable."""
    from scipy import ndimage
    m = sample["map"]
    path = sample["path"]
    s = (int(path[0][0]), int(path[0][1]))
    g = (int(path[-1][0]), int(path[-1][1]))
    k = 2 * margin + 1
    near = ndimage.binary_dilation(m, structure=np.ones((k, k), dtype=bool))
    rr, cc = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    exempt = (np.hypot(rr - s[0], cc - s[1]) <= margin) | (np.hypot(rr - g[0], cc - g[1]) <= margin)
    blocked = m | (near & ~exempt)
    blocked[g] = False
    city = CityGrid(blocked, None, alpha=0.0, corner_cutting=True)
    return city, city.idx(*s), city.idx(*g)


def gt_cost(path) -> tuple[float, bool]:
    total, ok = 0.0, True
    for a, b in zip(path[:-1], path[1:]):
        dr, dc = abs(int(b[0]) - int(a[0])), abs(int(b[1]) - int(a[1]))
        if max(dr, dc) != 1:
            ok = False
        total += math.sqrt(dr * dr + dc * dc)
    return total, ok


# ------------------------------------------------------------------- validación

def parte_validacion():
    files = map_splits()[0]
    rows = []
    for f in files:
        smp = load_map_sample(f)
        city, s, g = dataset_city(smp, None)
        succ = city.successors()
        h = city.h_admissible(g).tolist()
        r_ucs = ucs(succ, s, g)
        r_a = astar(succ, s, g, h)
        c_gt, ok = gt_cost(smp["path"])
        # mismo mapa con el modelo de holgura de 1 celda con que se generó el dataset
        cm, sm, gm = dataset_city_margin(smp, 1)
        r_m = astar(cm.successors(), sm, gm, cm.h_admissible(gm).tolist())
        rows.append({
            "archivo": f.split("/")[-1], "carpeta": f.replace("\\", "/").split("/")[-2],
            "obstaculos": float(smp["map"].mean()),
            "pasos_gt": len(smp["path"]) - 1, "camino_valido": ok, "costo_gt": c_gt,
            "costo_ucs": r_ucs.cost, "costo_astar": r_a.cost,
            "igual": abs(c_gt - r_a.cost) < 1e-6, "gt_sobre_opt": c_gt / r_a.cost,
            "costo_holgura": r_m.cost, "igual_holgura": abs(c_gt - r_m.cost) < 1e-6,
            "gt_sobre_opt_holgura": c_gt / r_m.cost if np.isfinite(r_m.cost) else np.nan,
            "exp_ucs": r_ucs.expanded, "exp_astar": r_a.expanded,
            "t_ucs": r_ucs.time_s, "t_astar": r_a.time_s,
        })
    write_csv(str(RES / "busqueda_validacion_dataset.csv"), rows)
    eq = sum(r["igual"] for r in rows)
    print(f"validación: {len(rows)} mapas, iguales {eq}, UCS==A* "
          f"{sum(abs(r['costo_ucs'] - r['costo_astar']) < 1e-6 for r in rows)}")


# --------------------------------------------------------------- entrenamiento

def parte_entrenar(per_map: int = 2000):
    files = map_splits()[1]
    rng = np.random.default_rng(123)
    Xs, ys = [], []
    t0 = time.perf_counter()
    for k, f in enumerate(files):
        smp = load_map_sample(f)
        city, s, g = dataset_city(smp, wind_seed=50_000 + k)
        X, y = training_rows(city, g, rng, per_map)
        if X is not None:
            Xs.append(X)
            ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    t_data = time.perf_counter() - t0
    MODELOS.mkdir(parents=True, exist_ok=True)
    info = {"mapas": len(files), "filas": int(len(y)), "rasgos": FEATURES, "t_datos_s": t_data}
    # validación interna: 10 % de las filas
    perm = rng.permutation(len(y))
    cut = int(0.9 * len(y))
    tr, va = perm[:cut], perm[cut:]
    for q in (None, 0.2):
        model = LearnedHeuristic(quantile=q)
        t1 = time.perf_counter()
        model.fit(X[tr], y[tr])
        t_fit = time.perf_counter() - t1
        pred = model.model.predict(X[va])
        key = "mse" if q is None else "q02"
        info[key] = {"t_fit_s": t_fit,
                     "mae_ratio": float(np.mean(np.abs(pred - y[va]))),
                     "frac_sobreestima": float(np.mean(pred > y[va])),
                     "max_sobre": float(np.max(pred / y[va]))}
        model.fit(X, y)          # modelo final con todas las filas
        joblib.dump(model, MODELOS / f"heuristica_{key}.joblib")
        print(key, info[key], flush=True)
    with open(RES / "heuristica_entrenamiento.json", "w") as fh:
        json.dump(info, fh, indent=2)


def load_models():
    return {k: joblib.load(MODELOS / f"heuristica_{k}.joblib") for k in ("mse", "q02")}


def run_learned(city, succ, s, g, model, hstar, opt_path, cstar):
    t0 = time.perf_counter()
    hmap = model.heuristic_map(city, g)
    t_inf = time.perf_counter() - t0
    r = astar(succ, s, g, hmap.tolist(), reopen=True)
    w_p, w_all, frac = overestimation_factors(hmap, hstar, opt_path)
    bound = max(1.0, w_p) * cstar
    return r, t_inf, w_p, w_all, frac, r.cost <= bound + 1e-9


# ------------------------------------------------------------------------ prueba

def parte_prueba():
    files = map_splits()[2]
    models = load_models()
    rows = []
    for k, f in enumerate(files):
        smp = load_map_sample(f)
        city, s, g = dataset_city(smp, wind_seed=90_000 + k)
        succ = city.successors()
        hstar = cost_to_go(city, g)
        cstar = float(hstar[s])
        h = city.h_admissible(g)
        r_a = astar(succ, s, g, h.tolist())
        base = {"mapa": f.split("/")[-1], "c_opt": cstar}
        rows.append({**base, "metodo": "A*", "costo": r_a.cost, "exp": r_a.expanded,
                     "t_s": r_a.time_s, "t_inf_s": 0.0, "w_P": float(np.max(
                         np.where(hstar[r_a.path] > 0, h[r_a.path] / np.maximum(hstar[r_a.path], 1e-12), 0))),
                     "cota_ok": abs(r_a.cost - cstar) < 1e-6})
        r_u = ucs(succ, s, g)
        rows.append({**base, "metodo": "UCS", "costo": r_u.cost, "exp": r_u.expanded,
                     "t_s": r_u.time_s, "t_inf_s": 0.0, "cota_ok": abs(r_u.cost - cstar) < 1e-6})
        for w in (1.5,):
            r_w = astar(succ, s, g, h.tolist(), weight=w)
            rows.append({**base, "metodo": f"WA* (w={w})", "costo": r_w.cost, "exp": r_w.expanded,
                         "t_s": r_w.time_s, "t_inf_s": 0.0, "w_P": w,
                         "cota_ok": r_w.cost <= w * cstar + 1e-9})
        for key, model in models.items():
            r, t_inf, w_p, w_all, frac, ok = run_learned(city, succ, s, g, model, hstar, r_a.path, cstar)
            rows.append({**base, "metodo": f"A*-θ ({model.name})", "costo": r.cost,
                         "exp": r.expanded, "reabiertos": r.reopened, "t_s": r.time_s + t_inf,
                         "t_inf_s": t_inf, "w_P": w_p, "w_all": w_all, "frac_sobre": frac,
                         "cota_ok": ok})
    for r in rows:
        r["brecha_pct"] = 100.0 * (r["costo"] / r["c_opt"] - 1.0)
    write_csv(str(RES / "busqueda_heuristica_prueba.csv"), rows)
    print("prueba:", len(files), "mapas")


# --------------------------------------------------------------------- rejillas

def parte_rejillas(n_queries: int = 20, n_ida: int = 10, ida_limit: float = 15.0,
                   sizes=(50, 100, 150, 200)):
    models = load_models()
    rows = []
    for N in sizes:
        city, term, info = build_scenario(N, 25, hospitals_csv(), seed=0, city_seed=N)
        succ = city.successors()
        rng = np.random.default_rng(N)
        pairs = []
        while len(pairs) < n_queries:
            a, b = rng.choice(len(term), size=2, replace=False)
            pairs.append((int(term[a]), int(term[b])))
        for q, (s, g) in enumerate(pairs):
            hstar = cost_to_go(city, g)
            cstar = float(hstar[s])
            h = city.h_admissible(g)
            hl = h.tolist()
            base = {"N": N, "consulta": q, "c_opt": cstar, "m_min": city.m_min}
            r_a = astar(succ, s, g, hl)
            methods = [("UCS", lambda: ucs(succ, s, g)), ("A*", lambda: astar(succ, s, g, hl)),
                       ("WA* (w=1.5)", lambda: astar(succ, s, g, hl, weight=1.5))]
            if q < n_ida:
                methods += [("IDA*-CR", lambda: ida_star(succ, s, g, hl, mode="cr", tt=True,
                                                         time_limit=ida_limit, node_limit=4_000_000)),
                            ("IDA*", lambda: ida_star(succ, s, g, hl, mode="classic", tt=True,
                                                      time_limit=ida_limit, node_limit=4_000_000))]
            for name, fn in methods:
                r = fn()
                row = {**base, "metodo": name, "estado": r.status, "costo": r.cost,
                       "exp": r.expanded, "gen": r.generated, "prof": r.depth,
                       "ebf": ebf_of(r) if r.status == "ok" else math.nan,
                       "iter": r.iterations, "t_s": r.time_s, "t_inf_s": 0.0,
                       "max_frontera": r.max_frontier}
                if q < 3 and name != "IDA*":
                    _, mem = peak_memory(fn)
                    row["mem_kib"] = mem
                rows.append(row)
            for key, model in models.items():
                r, t_inf, w_p, w_all, frac, ok = run_learned(city, succ, s, g, model, hstar, r_a.path, cstar)
                row = {**base, "metodo": f"A*-θ ({model.name})", "estado": r.status, "costo": r.cost,
                       "exp": r.expanded, "gen": r.generated, "prof": r.depth,
                       "ebf": ebf_of(r), "iter": 1,
                       "t_s": r.time_s + t_inf, "t_inf_s": t_inf, "max_frontera": r.max_frontier,
                       "reabiertos": r.reopened, "w_P": w_p, "w_all": w_all, "frac_sobre": frac,
                       "cota_ok": ok}
                if q < 3:
                    _, mem = peak_memory(lambda: astar(succ, s, g, model.heuristic_map(city, g).tolist(), reopen=True))
                    row["mem_kib"] = mem
                rows.append(row)
            print(N, q, f"C*={cstar:.2f}", flush=True)
        for r in rows:
            if r["N"] == N:
                r["brecha_pct"] = 100.0 * (r["costo"] / r["c_opt"] - 1.0) if np.isfinite(r["costo"]) else math.nan
        write_csv(str(RES / "busqueda_rejillas.csv"), rows)


def parte_rejillas_aprendida(n_queries: int = 20, sizes=(50, 100, 150, 200)):
    """Vuelve a medir sólo A*-θ en las mismas consultas de `parte_rejillas` (con un hilo de
    OpenMP) y reemplaza esas filas en busqueda_rejillas.csv."""
    import pandas as pd
    models = load_models()
    path = RES / "busqueda_rejillas.csv"
    old = pd.read_csv(path)
    keep = old[~old.metodo.str.startswith("A*-θ")]
    rows = []
    for N in sizes:
        city, term, info = build_scenario(N, 25, hospitals_csv(), seed=0, city_seed=N)
        succ = city.successors()
        rng = np.random.default_rng(N)
        pairs = []
        while len(pairs) < n_queries:
            a, b = rng.choice(len(term), size=2, replace=False)
            pairs.append((int(term[a]), int(term[b])))
        for q, (s, g) in enumerate(pairs):
            hstar = cost_to_go(city, g)
            cstar = float(hstar[s])
            r_a = astar(succ, s, g, city.h_admissible(g).tolist())
            base = {"N": N, "consulta": q, "c_opt": cstar, "m_min": city.m_min}
            for key, model in models.items():
                r, t_inf, w_p, w_all, frac, ok = run_learned(city, succ, s, g, model, hstar, r_a.path, cstar)
                row = {**base, "metodo": f"A*-θ ({model.name})", "estado": r.status, "costo": r.cost,
                       "exp": r.expanded, "gen": r.generated, "prof": r.depth,
                       "ebf": ebf_of(r), "iter": 1,
                       "t_s": r.time_s + t_inf, "t_inf_s": t_inf, "max_frontera": r.max_frontier,
                       "reabiertos": r.reopened, "w_P": w_p, "w_all": w_all, "frac_sobre": frac,
                       "cota_ok": ok, "brecha_pct": 100.0 * (r.cost / cstar - 1.0)}
                if q < 3:
                    _, mem = peak_memory(lambda: astar(succ, s, g, model.heuristic_map(city, g).tolist(), reopen=True))
                    row["mem_kib"] = mem
                rows.append(row)
        print("aprendida", N, flush=True)
    new = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    new = new.sort_values(["N", "consulta"], kind="stable")
    new.to_csv(path, index=False)


def parte_recalcular_ebf():
    """Recalcula b* en busqueda_rejillas.csv con N = generados − 1 (sin la raíz), a partir de
    las columnas `gen` y `prof` ya medidas (los caminos no cambian)."""
    import pandas as pd
    path = RES / "busqueda_rejillas.csv"
    d = pd.read_csv(path)
    ok = (d.estado == "ok") & d.gen.notna() & d.prof.notna()
    d.loc[ok, "ebf"] = [effective_branching_factor(int(g) - 1, int(p)) for g, p in zip(d.gen[ok], d.prof[ok])]
    d.loc[~ok, "ebf"] = math.nan
    d.to_csv(path, index=False)
    print("b* recalculado en", int(ok.sum()), "filas")


# ------------------------------------------------------------ IDA* sin TT

def parte_ida_pequenas(sizes=(8, 12, 16, 20, 24, 28), n_queries: int = 10, limit: float = 10.0):
    rows = []
    for N in sizes:
        city = build_city(N, seed=7 * N)
        comp = city.largest_component()
        free = np.flatnonzero(comp)
        succ = city.successors()
        rng = np.random.default_rng(N)
        for q in range(n_queries):
            s, g = (int(x) for x in rng.choice(free, size=2, replace=False))
            hl = city.h_admissible(g).tolist()
            cstar = float(dijkstra(city.csr(), directed=True, indices=s)[g])
            for name, fn in [("A*", lambda: astar(succ, s, g, hl)),
                             ("IDA* sin TT", lambda: ida_star(succ, s, g, hl, mode="classic", tt=False,
                                                              time_limit=limit, node_limit=3_000_000)),
                             ("IDA*-CR sin TT", lambda: ida_star(succ, s, g, hl, mode="cr", tt=False,
                                                                 time_limit=limit, node_limit=3_000_000)),
                             ("IDA* con TT", lambda: ida_star(succ, s, g, hl, mode="classic", tt=True,
                                                              time_limit=limit, node_limit=3_000_000)),
                             ("IDA*-CR con TT", lambda: ida_star(succ, s, g, hl, mode="cr", tt=True,
                                                                 time_limit=limit, node_limit=3_000_000))]:
                r = fn()
                rows.append({"N": N, "consulta": q, "metodo": name, "estado": r.status,
                             "costo": r.cost, "c_opt": cstar, "exp": r.expanded, "iter": r.iterations,
                             "t_s": r.time_s, "memoria_nodos": r.max_frontier})
        print("ida", N, flush=True)
        write_csv(str(RES / "busqueda_ida_pequenas.csv"), rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", default="todo",
                    choices=["validacion", "entrenar", "prueba", "rejillas", "rejillas_aprendida",
                             "recalcular_ebf", "ida_pequenas", "todo"])
    a = ap.parse_args()
    if a.parte in ("validacion", "todo"):
        parte_validacion()
    if a.parte in ("entrenar", "todo"):
        parte_entrenar()
    if a.parte in ("prueba", "todo"):
        parte_prueba()
    if a.parte in ("rejillas", "todo"):
        parte_rejillas()
    if a.parte == "rejillas_aprendida":
        parte_rejillas_aprendida()
    if a.parte == "recalcular_ebf":
        parte_recalcular_ebf()
    if a.parte in ("ida_pequenas", "todo"):
        parte_ida_pequenas()
