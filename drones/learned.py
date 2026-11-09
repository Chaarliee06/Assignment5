"""Heurística aprendida para A*.

Se aprende el cociente r(n) = h*(n) / octil(n, g) entre el costo exacto restante y la
distancia octil, con un modelo de gradient boosting sobre rasgos que se calculan para
todas las celdas a la vez (vectorizado). La heurística es h_θ(n) = octil(n, g)·r̂(n), con
h_θ(g) = 0. Los objetivos h*(n) salen de un Dijkstra sobre el grafo inverso desde la meta.

h_θ puede sobreestimar, así que A* pierde la garantía de optimalidad. Si en un camino
óptimo P* se cumple h_θ(n) ≤ w·h*(n), A* con reapertura devuelve un costo ≤ w·C*
(se verifica empíricamente en experiments/exp_search.py).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse.csgraph import dijkstra
from sklearn.ensemble import HistGradientBoostingRegressor

from .grid import CityGrid, octile

FEATURES = ["octil", "euclid", "frac_bloq_linea", "cruces_linea", "mult_viento_linea",
            "octil_x_viento", "dens_local_n", "dens_local_meta", "abs_dr", "abs_dc",
            "dens_global", "m_min"]


def cost_to_go(city: CityGrid, goal: int) -> np.ndarray:
    """h*(n) para todo n: Dijkstra desde la meta en el grafo inverso."""
    return dijkstra(city.csr(reverse=True), directed=True, indices=goal)


def feature_matrix(city: CityGrid, goal: int, n_samples: int = 16) -> np.ndarray:
    """Rasgos (H·W, len(FEATURES)) para la meta dada."""
    H, W = city.H, city.W
    blocked = city.blocked
    gr, gc = divmod(goal, W)
    rr, cc = np.mgrid[0:H, 0:W]
    dr = (gr - rr).astype(float)
    dc = (gc - cc).astype(float)
    octv = octile(dr, dc)
    euc = np.hypot(dr, dc)
    ts = (np.arange(n_samples) + 0.5) / n_samples
    pr = np.clip(np.rint(rr[..., None] + dr[..., None] * ts), 0, H - 1).astype(np.int32)
    pc = np.clip(np.rint(cc[..., None] + dc[..., None] * ts), 0, W - 1).astype(np.int32)
    occ = blocked[pr, pc]
    frac = occ.mean(axis=-1)
    cross = (np.diff(occ.astype(np.int8), axis=-1) == 1).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ur = np.where(euc > 0, dr / euc, 0.0)
        uc = np.where(euc > 0, dc / euc, 0.0)
    proj = (city.wind[pr, pc, 0] * ur[..., None] + city.wind[pr, pc, 1] * uc[..., None]).mean(axis=-1)
    mult = 1.0 - city.alpha * proj
    dens = ndimage.uniform_filter(blocked.astype(float), size=7, mode="constant", cval=1.0)
    X = np.stack([
        octv, euc, frac, cross, mult, octv * mult, dens,
        np.full_like(octv, dens[gr, gc]), np.abs(dr), np.abs(dc),
        np.full_like(octv, blocked.mean()), np.full_like(octv, city.m_min),
    ], axis=-1)
    return X.reshape(H * W, -1)


def training_rows(city: CityGrid, goal: int, rng: np.random.Generator, per_map: int = 2000):
    """Filas (X, y) con y = h*/octil para celdas alcanzables distintas de la meta."""
    hstar = cost_to_go(city, goal)
    octv = city.octile_map(goal)
    ok = np.isfinite(hstar) & (octv > 0)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return None, None
    pick = rng.choice(idx, size=min(per_map, len(idx)), replace=False)
    X = feature_matrix(city, goal)[pick]
    y = hstar[pick] / octv[pick]
    return X, y


class LearnedHeuristic:
    """Envuelve el modelo; `quantile` < 0.5 produce una heurística más conservadora."""

    def __init__(self, quantile: float | None = None, max_iter: int = 300, seed: int = 0):
        kw = dict(max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=63,
                  min_samples_leaf=40, random_state=seed)
        if quantile is None:
            self.model = HistGradientBoostingRegressor(loss="squared_error", **kw)
            self.name = "MSE"
        else:
            self.model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)
            self.name = f"q={quantile:g}"
        self.quantile = quantile

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def heuristic_map(self, city: CityGrid, goal: int) -> np.ndarray:
        """h_θ para todas las celdas (incluye el tiempo de inferencia del modelo)."""
        X = feature_matrix(city, goal)
        r = self.model.predict(X)
        r = np.maximum(r, 0.0)
        h = city.octile_map(goal) * r
        h[goal] = 0.0
        return h


def overestimation_factors(h: np.ndarray, hstar: np.ndarray, path: list[int]):
    """w_P = máx_{n∈P*} h(n)/h*(n) y w_all = máx sobre todas las celdas alcanzables."""
    ok = np.isfinite(hstar) & (hstar > 0)
    ratio = np.zeros_like(h)
    ratio[ok] = h[ok] / hstar[ok]
    w_all = float(ratio[ok].max()) if ok.any() else 1.0
    p = np.array([n for n in path if ok[n]], dtype=int)
    w_path = float(ratio[p].max()) if len(p) else 1.0
    frac_over = float((ratio[ok] > 1.0 + 1e-9).mean()) if ok.any() else 0.0
    return w_path, w_all, frac_over
