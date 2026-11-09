"""Generador de mapas de ocupación.

Reproduce el procedimiento del 2D Path Planning Dataset de Kaggle
(https://www.kaggle.com/datasets/dcaffo/2dpathplanningdataset), cuyo código fuente
está en https://github.com/dcaffo98/path-planning-cnn (dataset/map_factory.py):

    maze = rand(h, w) > alpha          # 1 con probabilidad 1 − alpha
    maze = MORPH_OPEN(maze, ones(k, k))  # erosión y luego dilatación
    obstáculo ⇔ maze == 1

con (k, alpha) elegido al azar de MAZE_COMBINATIONS. Aquí la apertura morfológica se
hace con SciPy usando el mismo manejo de bordes que OpenCV (el borde no erosiona ni
dilata), y se verificó que coincide celda por celda con cv2.morphologyEx.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

MAZE_COMBINATIONS = ((7, 0.06), (7, 0.07), (7, 0.08), (7, 0.09),
                     (5, 0.10), (5, 0.11), (5, 0.12), (5, 0.13), (5, 0.14))


def morph_open(img: np.ndarray, k: int) -> np.ndarray:
    """Apertura binaria con kernel k×k, bordes como en OpenCV."""
    st = np.ones((k, k), dtype=bool)
    er = ndimage.binary_erosion(img.astype(bool), structure=st, border_value=1)
    return ndimage.binary_dilation(er, structure=st, border_value=0)


def random_maze(h: int, w: int, rng: np.random.Generator, k: int | None = None,
                alpha: float | None = None) -> np.ndarray:
    """Mapa de ocupación (True = obstáculo) con el procedimiento del dataset."""
    if k is None or alpha is None:
        k, alpha = MAZE_COMBINATIONS[rng.integers(len(MAZE_COMBINATIONS))]
    maze = rng.random((h, w)) > alpha
    return morph_open(maze, k)


def scaled_maze(n: int, rng: np.random.Generator, base: int = 100) -> np.ndarray:
    """Mapa N×N con obstáculos del mismo tamaño relativo que en mapas de 100×100.

    Se genera a la resolución del dataset y se reescala por vecino más cercano, de modo
    que la fracción de área bloqueada y la forma de las zonas no dependen de N.
    """
    m = random_maze(base, base, rng)
    if n == base:
        return m
    idx = np.minimum((np.arange(n) * base / n).astype(int), base - 1)
    return m[np.ix_(idx, idx)]


def random_start_goal(blocked: np.ndarray, min_dist: float, rng: np.random.Generator,
                      max_it: int | None = None):
    """Inicio libre y meta a distancia euclidiana ≥ min_dist (como el dataset)."""
    h, w = blocked.shape
    max_it = max_it or h * w
    for _ in range(max_it):
        s = (int(rng.integers(h)), int(rng.integers(w)))
        g = (int(rng.integers(h)), int(rng.integers(w)))
        if blocked[s] or blocked[g]:
            continue
        if math.dist(s, g) >= min_dist:
            return s, g
    raise ValueError("no se encontró un par inicio/meta suficientemente lejano")
