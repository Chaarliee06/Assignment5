"""Modelo de la ciudad como grafo dirigido ponderado.

La ciudad es una rejilla N×N. Cada celda libre es un nodo; las celdas bloqueadas son
zonas de exclusión aérea (no-fly). Un dron se mueve a sus 8 vecinas. El costo de la
arista u→v depende de la dirección por el viento:

    c(u, v) = ℓ(u, v) · (1 − α ⟨w̄(u, v), d̂(u, v)⟩),

con ℓ ∈ {1, √2} la longitud del movimiento, d̂ su dirección unitaria, w̄ el promedio del
viento en u y v (|w| ≤ 1) y α ∈ [0, 1). Con viento a favor el costo baja y con viento en
contra sube, así que en general c(u, v) ≠ c(v, u).

La heurística admisible y consistente es h(n) = m_min · octil(n, g), donde
m_min = min_{(u,v)} c(u, v) / ℓ(u, v) es el costo mínimo por unidad de longitud.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse

SQRT2 = math.sqrt(2.0)
# (dr, dc): primero los 4 ortogonales y luego los 4 diagonales
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
MOVE_LEN = tuple(SQRT2 if dr and dc else 1.0 for dr, dc in MOVES)


def make_wind(n: int, rng: np.random.Generator, prevailing: float = 0.7,
              sigma_frac: float = 0.12) -> np.ndarray:
    """Campo de viento suave de forma (n, n, 2) con |w| ≤ 1.

    Es la suma de un viento dominante (dirección aleatoria) y una turbulencia suave
    (ruido blanco filtrado con una gaussiana de σ = sigma_frac·n). Las componentes
    están en ejes (fila, columna).
    """
    theta = rng.uniform(0.0, 2.0 * math.pi)
    base = prevailing * np.array([math.sin(theta), math.cos(theta)])
    sigma = max(1.0, sigma_frac * n)
    noise = np.stack([ndimage.gaussian_filter(rng.normal(size=(n, n)), sigma, mode="wrap")
                      for _ in range(2)], axis=-1)
    noise /= np.max(np.linalg.norm(noise, axis=-1)) + 1e-12
    w = base + (1.0 - prevailing) * noise
    mag = np.linalg.norm(w, axis=-1, keepdims=True)
    return np.where(mag > 1.0, w / np.maximum(mag, 1e-12), w)


def octile(dr, dc):
    """Distancia octil para desplazamientos (dr, dc) (escalares o arreglos)."""
    dr = np.abs(dr)
    dc = np.abs(dc)
    return (dr + dc) + (SQRT2 - 2.0) * np.minimum(dr, dc)


@dataclass
class CityGrid:
    """Rejilla con zonas de exclusión y viento; construye las aristas dirigidas.

    Parámetros
    ----------
    blocked : (H, W) bool, True = zona de exclusión.
    wind : (H, W, 2) float o None (sin viento).
    alpha : intensidad del efecto del viento sobre el costo.
    corner_cutting : si es False no se permiten diagonales que "corten esquina"
        junto a una celda bloqueada (más seguro para un dron).
    """

    blocked: np.ndarray
    wind: np.ndarray | None = None
    alpha: float = 0.5
    corner_cutting: bool = False
    cost: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.blocked = np.asarray(self.blocked, dtype=bool)
        self.H, self.W = self.blocked.shape
        self.n_nodes = self.H * self.W
        if self.wind is None:
            self.wind = np.zeros((self.H, self.W, 2))
        self._build_costs()
        self._succ = None
        self._pred = None
        self._csr = None

    # ------------------------------------------------------------------ aristas
    def _build_costs(self):
        H, W = self.H, self.W
        free = ~self.blocked
        cost = np.full((8, H, W), np.inf)
        for k, (dr, dc) in enumerate(MOVES):
            r0, r1 = max(0, -dr), H - max(0, dr)
            c0, c1 = max(0, -dc), W - max(0, dc)
            src = np.zeros((H, W), dtype=bool)
            dst = np.zeros((H, W), dtype=bool)
            src[r0:r1, c0:c1] = free[r0:r1, c0:c1]
            dst[r0:r1, c0:c1] = free[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            ok = src & dst
            if dr and dc and not self.corner_cutting:
                side1 = np.zeros((H, W), dtype=bool)
                side2 = np.zeros((H, W), dtype=bool)
                side1[r0:r1, c0:c1] = free[r0 + dr:r1 + dr, c0:c1]
                side2[r0:r1, c0:c1] = free[r0:r1, c0 + dc:c1 + dc]
                ok &= side1 & side2
            length = MOVE_LEN[k]
            wbar = np.zeros((H, W, 2))
            wbar[r0:r1, c0:c1] = 0.5 * (self.wind[r0:r1, c0:c1] + self.wind[r0 + dr:r1 + dr, c0 + dc:c1 + dc])
            proj = (wbar[..., 0] * dr + wbar[..., 1] * dc) / length  # ⟨w̄, d̂⟩
            mult = 1.0 - self.alpha * proj
            cost[k][ok] = (length * mult)[ok]
        self.cost = cost
        finite = np.isfinite(cost)
        lengths = np.array(MOVE_LEN)[:, None, None]
        per_unit = np.where(finite, cost / lengths, np.inf)
        self.m_min = float(per_unit.min()) if finite.any() else 1.0
        self.m_max = float(np.where(finite, per_unit, -np.inf).max()) if finite.any() else 1.0
        self.n_edges = int(finite.sum())

    # ------------------------------------------------------------------ índices
    def idx(self, r: int, c: int) -> int:
        return r * self.W + c

    def rc(self, u: int) -> tuple[int, int]:
        return divmod(u, self.W)

    # --------------------------------------------------------- listas de vecinos
    def successors(self):
        """Lista de adyacencia hacia adelante: succ[u] = [(v, c(u,v)), ...]."""
        if self._succ is None:
            self._succ = self._adjacency(reverse=False)
        return self._succ

    def predecessors(self):
        """Lista de adyacencia inversa: pred[v] = [(u, c(u,v)), ...]."""
        if self._pred is None:
            self._pred = self._adjacency(reverse=True)
        return self._pred

    def _edge_arrays(self):
        H, W = self.H, self.W
        us, vs, cs = [], [], []
        rr, cc = np.mgrid[0:H, 0:W]
        for k, (dr, dc) in enumerate(MOVES):
            ok = np.isfinite(self.cost[k])
            u = (rr[ok] * W + cc[ok])
            v = ((rr[ok] + dr) * W + (cc[ok] + dc))
            us.append(u)
            vs.append(v)
            cs.append(self.cost[k][ok])
        return np.concatenate(us), np.concatenate(vs), np.concatenate(cs)

    def _adjacency(self, reverse: bool):
        u, v, c = self._edge_arrays()
        if reverse:
            u, v = v, u
        order = np.lexsort((v, u))
        u, v, c = u[order], v[order], c[order]
        adj = [[] for _ in range(self.n_nodes)]
        for a, b, w in zip(u.tolist(), v.tolist(), c.tolist()):
            adj[a].append((b, w))
        return adj

    def csr(self, reverse: bool = False) -> sparse.csr_matrix:
        """Matriz de adyacencia dispersa (para Dijkstra de SciPy)."""
        u, v, c = self._edge_arrays()
        if reverse:
            u, v = v, u
        return sparse.csr_matrix((c, (u, v)), shape=(self.n_nodes, self.n_nodes))

    # ------------------------------------------------------------- heurísticas
    def octile_map(self, goal: int) -> np.ndarray:
        gr, gc = self.rc(goal)
        rr, cc = np.mgrid[0:self.H, 0:self.W]
        return octile(rr - gr, cc - gc).ravel()

    def h_admissible(self, goal: int) -> np.ndarray:
        """h(n) = m_min · octil(n, g): admisible y consistente (ver reporte)."""
        return self.m_min * self.octile_map(goal)

    def path_cost(self, path: list[int]) -> float:
        """Costo de un camino (lista de nodos) usando las aristas del grafo."""
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            ar, ac = self.rc(a)
            br, bc = self.rc(b)
            k = MOVES.index((br - ar, bc - ac))
            total += float(self.cost[k, ar, ac])
        return total

    def largest_component(self) -> np.ndarray:
        """Máscara de la componente conexa (8-vecindad válida) más grande."""
        from scipy.sparse.csgraph import connected_components
        ncomp, labels = connected_components(self.csr(), directed=True, connection="strong")
        free = ~self.blocked.ravel()
        counts = np.bincount(labels[free], minlength=ncomp)
        best = int(np.argmax(counts))
        return (labels == best) & free

    def asymmetry(self) -> float:
        """Promedio de |c(u,v) − c(v,u)| / ((c(u,v)+c(v,u))/2) sobre las aristas."""
        u, v, c = self._edge_arrays()
        m = sparse.csr_matrix((c, (u, v)), shape=(self.n_nodes, self.n_nodes))
        back = np.asarray(m[v, u]).ravel()
        return float(np.mean(np.abs(c - back) / (0.5 * (c + back))))
