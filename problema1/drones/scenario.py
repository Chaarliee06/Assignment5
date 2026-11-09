"""Escenario integrado: ciudad (rejilla con viento y zonas de exclusión), depósito y
clínicas tomadas de hospitales reales de Los Ángeles proyectados a la rejilla."""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .datasets import LA_BBOX, load_hospitals, project_to_grid, region
from .grid import CityGrid, make_wind
from .mapgen import scaled_maze

DEPOT_NAME = "LAC+USC MEDICAL CENTER"


def build_city(n: int, seed: int, alpha: float = 0.5) -> CityGrid:
    """Ciudad n×n: zonas de exclusión con el generador del dataset y viento suave."""
    rng = np.random.default_rng(seed)
    blocked = scaled_maze(n, rng)
    return CityGrid(blocked, make_wind(n, rng), alpha=alpha)


def build_scenario(n: int, k: int, hospitals_csv: str, seed: int = 0, alpha: float = 0.5,
                   city_seed: int | None = None):
    """Devuelve (city, terminals, info). terminals[0] es el depósito y terminals[1:] las
    K clínicas (nodos de la rejilla). Las clínicas se eligen al azar (semilla) entre los
    hospitales de la región, sin repetir celdas."""
    city = build_city(n, seed if city_seed is None else city_seed, alpha)
    allowed = city.largest_component()
    hosp = region(load_hospitals(hospitals_csv), LA_BBOX)
    rng = np.random.default_rng(10_000 + seed)
    is_depot = hosp.name.str.upper().eq(DEPOT_NAME)
    depot_row = int(np.flatnonzero(is_depot)[0]) if is_depot.any() else int(hosp.beds.fillna(0).idxmax())
    others = [i for i in range(len(hosp)) if i != depot_row]
    chosen = list(rng.choice(others, size=k, replace=False))
    sub = hosp.iloc[[depot_row] + chosen].reset_index(drop=True)
    nodes = project_to_grid(sub, n, allowed, LA_BBOX)
    info = sub.assign(node=nodes)
    return city, nodes, info


def cost_matrix(city: CityGrid, terminals) -> np.ndarray:
    """Matriz C (K+1)×(K+1) de costos de camino mínimo (Dijkstra dirigido)."""
    D = dijkstra(city.csr(), directed=True, indices=np.asarray(terminals))
    C = D[:, np.asarray(terminals)]
    np.fill_diagonal(C, 0.0)
    return C
