"""Búsqueda de caminos en el grafo dirigido: UCS, A*, A* ponderado, IDA* e IDA*-CR.

Todas las funciones reciben `succ` (lista de adyacencia: succ[u] = [(v, c), ...]) y
nodos enteros. `h` es una secuencia indexable por nodo (lista de Python, por velocidad)
o None (h ≡ 0, es decir UCS).

Contadores
----------
expanded   nodos expandidos (se generan sus sucesores); IDA* cuenta re-expansiones
generated  nodos generados (insertados en la frontera o visitados por la DFS)
reopened   nodos cerrados que se vuelven a abrir (sólo con heurística inconsistente)
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

INF = math.inf


@dataclass
class SearchResult:
    cost: float
    path: list
    expanded: int
    generated: int
    reopened: int = 0
    max_frontier: int = 0
    iterations: int = 1
    time_s: float = 0.0
    status: str = "ok"          # ok | limit | unreachable
    extra: dict = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return max(0, len(self.path) - 1)


def _reconstruct(parent, goal):
    path = []
    u = goal
    while u != -1:
        path.append(u)
        u = parent[u]
    path.reverse()
    return path


def astar(succ, start: int, goal: int, h=None, weight: float = 1.0, reopen: bool = False,
          node_limit: int | None = None, keep_closed: bool = False) -> SearchResult:
    """A* (UCS si h es None; A* ponderado si weight > 1).

    La prueba de meta se hace al *expandir*, lo que garantiza optimalidad con h
    admisible. Desempate: a igual f se prefiere el nodo con mayor g.
    Con reopen=True los nodos cerrados se reabren si se mejora su g (necesario para
    acotar el costo con heurísticas inconsistentes, como la aprendida).
    """
    t0 = time.perf_counter()
    n = len(succ)
    g = [INF] * n
    parent = [-1] * n
    closed = bytearray(n)
    g[start] = 0.0
    hs = 0.0 if h is None else weight * h[start]
    heap = [(hs, -0.0, start)]
    push, pop = heapq.heappush, heapq.heappop
    expanded = reopened = 0
    generated = 1
    max_front = 1
    use_h = h is not None
    while heap:
        f, neg_g, u = pop(heap)
        gu = -neg_g
        if gu > g[u] or closed[u]:
            continue
        if u == goal:
            res = SearchResult(gu, _reconstruct(parent, goal), expanded, generated, reopened,
                               max_front, 1, time.perf_counter() - t0)
            if keep_closed:
                res.extra["closed"] = np.frombuffer(bytes(closed), dtype=np.uint8).copy()
            return res
        closed[u] = 1
        expanded += 1
        if node_limit is not None and expanded >= node_limit:
            return SearchResult(INF, [], expanded, generated, reopened, max_front, 1,
                                time.perf_counter() - t0, "limit")
        for v, c in succ[u]:
            nv = gu + c
            if nv < g[v]:
                if closed[v]:
                    if not reopen:
                        continue
                    closed[v] = 0
                    reopened += 1
                g[v] = nv
                parent[v] = u
                push(heap, (nv + weight * h[v] if use_h else nv, -nv, v))
                generated += 1
        if len(heap) > max_front:
            max_front = len(heap)
    return SearchResult(INF, [], expanded, generated, reopened, max_front, 1,
                        time.perf_counter() - t0, "unreachable")


def ucs(succ, start, goal, node_limit=None) -> SearchResult:
    """Búsqueda de costo uniforme (Dijkstra con prueba de meta al expandir)."""
    return astar(succ, start, goal, h=None, node_limit=node_limit)


def ida_star(succ, start: int, goal: int, h, mode: str = "cr", tt: bool = True,
             node_limit: int | None = 5_000_000, time_limit: float | None = 60.0,
             growth: float = 1.0) -> SearchResult:
    """IDA* iterativo (DFS con pila explícita).

    mode="classic": el nuevo umbral es el mínimo f que excedió el umbral anterior
        (IDA* de Korf). La primera solución encontrada es óptima.
    mode="cr": IDA*-CR (Sarkar et al., 1991). El umbral se elige entre los f excedidos
        para que la siguiente iteración incluya ≈ (1+growth)·(nodos de la iteración)
        nodos. Si en la última iteración aparece una solución, la DFS continúa como
        ramificación y acotamiento (poda f ≥ mejor costo) hasta agotar la iteración,
        de modo que la solución devuelta también es óptima.
    tt=True: tabla de transposición por iteración (poda si se llega a un nodo con
        g ≥ al mejor g visto en esa iteración). Sin ella sólo se evitan ciclos sobre
        el camino actual y el número de caminos crece exponencialmente en rejillas.
    """
    t0 = time.perf_counter()
    n = len(succ)
    bound = h[start]
    total_exp = 0
    total_gen = 1
    iterations = 0
    best_cost = INF
    best_path: list = []
    max_tt = 0
    max_depth = 0
    while True:
        iterations += 1
        exceeded_min = INF
        exceeded: list = []
        it_exp = 0
        ttg = {} if tt else None
        on_path = None if tt else bytearray(n)
        if tt:
            ttg[start] = 0.0
        else:
            on_path[start] = 1
        if start == goal:
            return SearchResult(0.0, [start], 0, 1, 0, 1, iterations, time.perf_counter() - t0)
        # pila de marcos: nodo, g, hijos ordenados, índice del siguiente hijo
        st_u = [start]
        st_g = [0.0]
        st_ch = [sorted(succ[start], key=lambda vc: vc[1] + h[vc[0]])]
        st_i = [0]
        it_exp += 1
        while st_u:
            k = st_i[-1]
            ch = st_ch[-1]
            if k >= len(ch):
                u = st_u.pop()
                st_g.pop()
                st_ch.pop()
                st_i.pop()
                if not tt:
                    on_path[u] = 0
                continue
            st_i[-1] = k + 1
            v, c = ch[k]
            nv = st_g[-1] + c
            f = nv + h[v]
            total_gen += 1
            if f > bound:
                if mode == "classic":
                    if f < exceeded_min:
                        exceeded_min = f
                else:
                    exceeded.append(f)
                continue
            if f >= best_cost:          # sólo ocurre en la fase de B&B de IDA*-CR
                continue
            if tt:
                if nv >= ttg.get(v, INF):
                    continue
                ttg[v] = nv
            elif on_path[v]:
                continue
            if v == goal:
                if nv < best_cost:
                    best_cost = nv
                    best_path = st_u + [v]
                if mode == "classic":
                    break
                continue
            # expandir v
            it_exp += 1
            st_u.append(v)
            st_g.append(nv)
            st_ch.append(sorted(succ[v], key=lambda vc: vc[1] + h[vc[0]]))
            st_i.append(0)
            if not tt:
                on_path[v] = 1
            if len(st_u) > max_depth:
                max_depth = len(st_u)
            if (it_exp & 4095) == 0:
                if node_limit is not None and total_exp + it_exp >= node_limit:
                    return SearchResult(INF, [], total_exp + it_exp, total_gen, 0, max_depth,
                                        iterations, time.perf_counter() - t0, "limit",
                                        {"bound": bound, "tt_max": max_tt})
                if time_limit is not None and time.perf_counter() - t0 > time_limit:
                    return SearchResult(INF, [], total_exp + it_exp, total_gen, 0, max_depth,
                                        iterations, time.perf_counter() - t0, "limit",
                                        {"bound": bound, "tt_max": max_tt})
        total_exp += it_exp
        if tt and len(ttg) > max_tt:
            max_tt = len(ttg)
        if best_cost < INF:
            mem = max_tt if tt else max_depth
            return SearchResult(best_cost, best_path, total_exp, total_gen, 0, mem, iterations,
                                time.perf_counter() - t0, "ok",
                                {"bound": bound, "tt_max": max_tt, "max_depth": max_depth})
        # siguiente umbral
        if mode == "classic":
            if exceeded_min == INF:
                return SearchResult(INF, [], total_exp, total_gen, 0, max_tt, iterations,
                                    time.perf_counter() - t0, "unreachable")
            bound = exceeded_min
        else:
            if not exceeded:
                return SearchResult(INF, [], total_exp, total_gen, 0, max_tt, iterations,
                                    time.perf_counter() - t0, "unreachable")
            arr = np.asarray(exceeded)
            kth = int(min(len(arr) - 1, max(0, math.ceil(growth * it_exp) - 1)))
            bound = float(np.partition(arr, kth)[kth])
        if node_limit is not None and total_exp >= node_limit:
            return SearchResult(INF, [], total_exp, total_gen, 0, max_tt, iterations,
                                time.perf_counter() - t0, "limit", {"bound": bound})
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            return SearchResult(INF, [], total_exp, total_gen, 0, max_tt, iterations,
                                time.perf_counter() - t0, "limit", {"bound": bound})


def effective_branching_factor(n_nodes: int, depth: int) -> float:
    """b* tal que N + 1 = 1 + b* + b*² + … + b*^d (definición de Russell y Norvig).

    N es el número de nodos generados SIN contar la raíz (el «+1» de la izquierda ya es la
    raíz) y d la profundidad (aristas) de la solución. Los contadores `generated` de este
    módulo sí incluyen la raíz, así que se debe llamar con `generated - 1`
    (ver `ebf_of`).
    """
    if depth <= 0 or n_nodes <= 0:
        return float("nan")
    target = n_nodes + 1.0

    def total(b):
        if abs(b - 1.0) < 1e-12:
            return depth + 1.0
        # suma geométrica en forma estable
        return (b ** (depth + 1) - 1.0) / (b - 1.0)

    if total(1.0) >= target:
        return 1.0
    lo, hi = 1.0, 2.0
    while total(hi) < target:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ebf_of(result: "SearchResult") -> float:
    """b* de un resultado: los nodos generados incluyen la raíz, que no cuenta en N."""
    return effective_branching_factor(result.generated - 1, result.depth)
