"""Orden de visita a las clínicas: TSP asimétrico (ATSP) con depósito fijo.

Un recorrido es una permutación `p` de las clínicas 1..K; el costo es
c(0, p_1) + Σ c(p_i, p_{i+1}) + c(p_K, 0). La matriz C se obtiene con Dijkstra sobre la
rejilla con viento, por lo que C ≠ Cᵀ.

Exactos:        held_karp (DP O(2^K K²)), atsp_bnb (B&B DFS con cota MST), brute_force.
Cota inferior:  assignment_bound (relajación de asignación, para medir brechas con K grande).
Metaheurísticas (presupuesto igual de evaluaciones de la función objetivo):
    simulated_annealing, tabu_search, ga_ox, aco_mmas, ucb_hyperheuristic,
    memetic_ga (GA + búsqueda local) y aco_mmas(local_search=True)  → híbridos.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

MOVES = ("swap", "insert", "2opt")


# ------------------------------------------------------------------ utilidades

class Evaluator:
    """Cuenta evaluaciones y guarda la curva de convergencia (mejor vs evaluaciones)."""

    def __init__(self, C: np.ndarray, budget: int, n_points: int = 100):
        self.C = np.asarray(C, dtype=float)
        self.K = self.C.shape[0] - 1
        self.budget = budget
        self.evals = 0
        self.best = math.inf
        self.best_perm = None
        self.curve = []
        self.step = max(1, budget // n_points)
        self._a = np.zeros(self.K + 1, dtype=np.int64)
        self._b = np.zeros(self.K + 1, dtype=np.int64)

    def cost(self, perm) -> float:
        self._a[1:] = perm
        self._b[:-1] = perm
        c = float(self.C[self._a, self._b].sum())
        self.evals += 1
        if c < self.best:
            self.best = c
            self.best_perm = np.array(perm, copy=True)
        if self.evals % self.step == 0:
            self.curve.append((self.evals, self.best))
        return c

    def costs(self, perms: np.ndarray) -> np.ndarray:
        """Evalúa varias permutaciones (filas) a la vez."""
        m = perms.shape[0]
        a = np.concatenate([np.zeros((m, 1), dtype=np.int64), perms], axis=1)
        b = np.concatenate([perms, np.zeros((m, 1), dtype=np.int64)], axis=1)
        cs = self.C[a, b].sum(axis=1)
        for k in range(m):
            self.evals += 1
            if cs[k] < self.best:
                self.best = float(cs[k])
                self.best_perm = perms[k].copy()
            if self.evals % self.step == 0:
                self.curve.append((self.evals, self.best))
        return cs

    @property
    def done(self) -> bool:
        return self.evals >= self.budget


def tour_cost(C, perm) -> float:
    t = np.concatenate(([0], np.asarray(perm, dtype=int), [0]))
    return float(np.asarray(C)[t[:-1], t[1:]].sum())


def apply_move(p: np.ndarray, kind: str, i: int, j: int) -> np.ndarray:
    q = p.copy()
    if kind == "swap":
        q[i], q[j] = q[j], q[i]
    elif kind == "insert":
        x = q[i]
        q = np.delete(q, i)
        q = np.insert(q, j, x)
    else:  # 2-opt: invierte el segmento [i, j] (en ATSP cambia el sentido de esas aristas)
        if i > j:
            i, j = j, i
        q[i:j + 1] = q[i:j + 1][::-1]
    return q


def random_move(rng, K: int, kind: str | None = None):
    kind = kind or MOVES[int(rng.integers(3))]
    i, j = rng.choice(K, size=2, replace=False)
    return kind, int(i), int(j)


def nn_lists(C, L: int = 10) -> np.ndarray:
    """nn_in[x] = las L ciudades y con menor c(y, x) (candidatas a preceder a x)."""
    C = np.asarray(C, dtype=float)
    L = min(L, C.shape[0] - 1)          # la diagonal (∞) queda al final y nunca se elige
    M = C.copy()
    np.fill_diagonal(M, np.inf)
    return np.argsort(M, axis=0)[:L].T


def granular_insert(p: np.ndarray, rng, nn_in: np.ndarray):
    """Mueve una ciudad x justo después de uno de sus vecinos cercanos y (lista de
    candidatos granular, como en la búsqueda tabú granular de Toth y Vigo).
    Devuelve (nueva permutación, ciudad movida)."""
    K = len(p)
    i = int(rng.integers(K))
    x = int(p[i])
    y = int(nn_in[x][int(rng.integers(nn_in.shape[1]))])
    q = np.delete(p, i)
    j = 0 if y == 0 else int(np.flatnonzero(q == y)[0]) + 1
    return np.insert(q, j, x), x


def nearest_neighbor(C) -> np.ndarray:
    K = C.shape[0] - 1
    left = set(range(1, K + 1))
    u, perm = 0, []
    while left:
        v = min(left, key=lambda x: C[u, x])
        perm.append(v)
        left.remove(v)
        u = v
    return np.array(perm)


# ----------------------------------------------------------------------- exactos

def brute_force(C):
    K = C.shape[0] - 1
    best, bp = math.inf, None
    for perm in itertools.permutations(range(1, K + 1)):
        c = tour_cost(C, perm)
        if c < best:
            best, bp = c, perm
    return best, np.array(bp)


def held_karp(C):
    """DP de Held–Karp. dp[S, j] = costo mínimo de 0 → … → j visitando S (j ∈ S)."""
    t0 = time.perf_counter()
    C = np.asarray(C, dtype=float)
    K = C.shape[0] - 1
    full = (1 << K) - 1
    dp = np.full((1 << K, K), np.inf)
    parent = np.full((1 << K, K), -1, dtype=np.int64)
    for j in range(K):
        dp[1 << j, j] = C[0, j + 1]
    Cc = C[1:, 1:]
    by_size = sorted(range(1, 1 << K), key=lambda s: bin(s).count("1"))
    for S in by_size:
        row = dp[S]
        if not np.isfinite(row).any():
            continue
        members = [j for j in range(K) if S >> j & 1]
        for k in range(K):
            if S >> k & 1:
                continue
            S2 = S | (1 << k)
            cand = row[members] + Cc[members, k]
            a = int(np.argmin(cand))
            if cand[a] < dp[S2, k]:
                dp[S2, k] = cand[a]
                parent[S2, k] = members[a]
    final = dp[full] + C[1:, 0]
    j = int(np.argmin(final))
    best = float(final[j])
    perm, S = [], full
    while j != -1:
        perm.append(j + 1)
        pj = int(parent[S, j])
        S &= ~(1 << j)
        j = pj
    return best, np.array(perm[::-1]), time.perf_counter() - t0


def _prim(S: np.ndarray, nodes: list) -> float:
    """Peso del árbol de expansión mínima sobre `nodes` con la matriz simétrica S."""
    m = len(nodes)
    if m <= 1:
        return 0.0
    sub = S[np.ix_(nodes, nodes)]
    in_tree = np.zeros(m, dtype=bool)
    in_tree[0] = True
    dist = sub[0].copy()
    total = 0.0
    for _ in range(m - 1):
        d = np.where(in_tree, np.inf, dist)
        k = int(np.argmin(d))
        total += d[k]
        in_tree[k] = True
        dist = np.minimum(dist, sub[k])
    return float(total)


def _prim_list(S: list, nodes: list) -> float:
    """Prim O(m²) en Python puro sobre la matriz simétrica S (lista de listas); para los
    tamaños del B&B (m ≤ 13) es varias veces más rápido que la versión con NumPy."""
    m = len(nodes)
    if m <= 1:
        return 0.0
    first = nodes[0]
    rest = list(nodes[1:])
    S0 = S[first]
    dist = [S0[v] for v in rest]
    total = 0.0
    while rest:
        k = min(range(len(rest)), key=dist.__getitem__)
        total += dist[k]
        v = rest[k]
        last = len(rest) - 1
        rest[k], dist[k] = rest[last], dist[last]
        rest.pop()
        dist.pop()
        Sv = S[v]
        for j in range(len(rest)):
            d = Sv[rest[j]]
            if d < dist[j]:
                dist[j] = d
    return total


def atsp_bnb(C, ub: float | None = None, node_limit: int | None = 2_000_000,
             time_limit: float | None = 120.0):
    """B&B en profundidad con cota MST.

    Para un prefijo 0 → … → u con costo g y conjunto restante R, el resto del recorrido
    es un camino hamiltoniano u → R → 0. Ignorando direcciones y usando
    s_ij = min(c_ij, c_ji), ese camino es un árbol de expansión de R ∪ {u, 0}, así que
        LB₁ = g + MST_s(R ∪ {u, 0}),
        LB₂ = g + MST_s(R) + min_{r∈R} c(u, r) + min_{r∈R} c(r, 0),
    y LB = max(LB₁, LB₂) ≤ costo de cualquier completación (cota válida).
    """
    t0 = time.perf_counter()
    C = np.asarray(C, dtype=float)
    K = C.shape[0] - 1
    S = np.minimum(C, C.T).tolist()
    if ub is None:
        ub = math.inf
        best_perm = None
    else:
        best_perm = None
    best = [ub, best_perm]
    nodes = [0]
    pruned = [0]
    status = ["ok"]
    Cl = C.tolist()

    def lb(u, rem, g):
        if not rem:
            return g + Cl[u][0]
        lb1 = g + _prim_list(S, [u, 0] + rem)
        lb2 = g + _prim_list(S, rem) + min(Cl[u][r] for r in rem) + min(Cl[r][0] for r in rem)
        return max(lb1, lb2)

    def dfs(u, rem, g, path):
        nodes[0] += 1
        if node_limit is not None and nodes[0] >= node_limit:
            status[0] = "limit"
            return
        if time_limit is not None and (nodes[0] & 1023) == 0 and time.perf_counter() - t0 > time_limit:
            status[0] = "limit"
            return
        if not rem:
            total = g + Cl[u][0]
            if total < best[0]:
                best[0], best[1] = total, list(path)
            return
        for r in sorted(rem, key=lambda x: Cl[u][x]):
            ng = g + Cl[u][r]
            if ng >= best[0]:
                pruned[0] += 1
                continue
            rem2 = [x for x in rem if x != r]
            if lb(r, rem2, ng) >= best[0]:
                pruned[0] += 1
                continue
            path.append(r)
            dfs(r, rem2, ng, path)
            path.pop()
            if status[0] != "ok":
                return

    dfs(0, list(range(1, K + 1)), 0.0, [])
    perm = np.array(best[1]) if best[1] is not None else None
    return {"cost": best[0], "perm": perm, "nodes": nodes[0], "pruned": pruned[0],
            "time_s": time.perf_counter() - t0, "status": status[0]}


def assignment_bound(C) -> float:
    """Cota inferior por relajación de asignación (cada nodo con un sucesor y un predecesor)."""
    M = np.array(C, dtype=float)
    np.fill_diagonal(M, 1e18)
    r, c = linear_sum_assignment(M)
    return float(M[r, c].sum())


# --------------------------------------------------------------- metaheurísticas

def _calibrate_T0(ev: Evaluator, p, cur, rng, samples=60, accept=0.5):
    deltas = []
    for _ in range(samples):
        kind, i, j = random_move(rng, ev.K)
        c = ev.cost(apply_move(p, kind, i, j))
        if c > cur:
            deltas.append(c - cur)
    d = np.mean(deltas) if deltas else 1.0
    return -d / math.log(accept)


def simulated_annealing(C, rng, budget: int, t_end_ratio: float = 1e-3, operator=None):
    ev = Evaluator(C, budget)
    K = ev.K
    p = rng.permutation(K) + 1
    cur = ev.cost(p)
    T0 = _calibrate_T0(ev, p, cur, rng)
    T = T0
    remaining = max(1, budget - ev.evals)
    decay = t_end_ratio ** (1.0 / remaining)
    while not ev.done:
        kind, i, j = random_move(rng, K, operator)
        q = apply_move(p, kind, i, j)
        c = ev.cost(q)
        d = c - cur
        if d <= 0 or rng.random() < math.exp(-d / T):
            p, cur = q, c
        T *= decay
    return ev


def tabu_search(C, rng, budget: int, n_cand: int = 30, tenure: int | None = None,
                granular: float = 0.7):
    """Búsqueda tabú con lista de candidatos: en cada iteración se evalúan n_cand
    vecinos (70 % inserciones granulares hacia vecinos cercanos y 30 % swap/2-opt al
    azar) y se pasa al mejor no tabú, aunque empeore. Atributo tabú: las ciudades movidas
    (no pueden volver a moverse durante `tenure` iteraciones); aspiración: mejorar al
    mejor global."""
    ev = Evaluator(C, budget)
    K = ev.K
    nn_in = nn_lists(C)
    tenure = tenure or max(5, K // 4)
    p = rng.permutation(K) + 1
    ev.cost(p)
    tabu_until = np.zeros(K + 1, dtype=np.int64)
    it = 0
    while not ev.done:
        it += 1
        best_q, best_c, best_attr = None, math.inf, None
        for _ in range(n_cand):
            if ev.done:
                break
            if rng.random() < granular:
                q, x = granular_insert(p, rng, nn_in)
                attr = (x,)
            else:
                kind, i, j = random_move(rng, K, "swap" if rng.random() < 0.5 else "2opt")
                q = apply_move(p, kind, i, j)
                attr = (int(p[i]), int(p[j]))
            prev_best = ev.best
            c = ev.cost(q)
            is_tabu = any(tabu_until[x] > it for x in attr)
            if (not is_tabu or c < prev_best) and c < best_c:
                best_q, best_c, best_attr = q, c, attr
        if best_q is None:
            continue
        p = best_q
        for x in best_attr:
            tabu_until[x] = it + tenure + int(rng.integers(0, 3))
    return ev


def order_crossover(p1: np.ndarray, p2: np.ndarray, rng) -> np.ndarray:
    """OX (Davis): copia un segmento de p1 y completa con el orden relativo de p2."""
    K = len(p1)
    i, j = sorted(rng.choice(K, size=2, replace=False))
    child = np.empty(K, dtype=p1.dtype)
    seg = p1[i:j + 1]
    child[i:j + 1] = seg
    order = np.roll(p2, -(j + 1))
    fill = order[~np.isin(order, seg)]
    pos = np.roll(np.arange(K), -(j + 1))
    pos = pos[(pos < i) | (pos > j)]
    child[pos] = fill
    return child


def _local_search(ev: Evaluator, p, cur, rng, max_evals: int, nn_in=None):
    """Búsqueda local de primera mejora: inserciones granulares (Or-opt de una ciudad
    hacia un vecino cercano) y algunos 2-opt al azar; acepta sólo si mejora."""
    if nn_in is None:
        nn_in = nn_lists(ev.C)
    tries = 0
    while tries < max_evals and not ev.done:
        if rng.random() < 0.8:
            q, _ = granular_insert(p, rng, nn_in)
        else:
            kind, i, j = random_move(rng, ev.K, "2opt")
            q = apply_move(p, kind, i, j)
        c = ev.cost(q)
        tries += 1
        if c < cur:
            p, cur = q, c
    return p, cur


def ga_ox(C, rng, budget: int, pop_size: int = 60, pc: float = 0.9, pm: float = 0.5,
          k_tour: int = 3, elite: int = 2, ls_prob: float = 0.0, ls_evals: int = 0):
    """GA de permutaciones con OX, mutación por inversión, torneo y elitismo.
    Con ls_prob > 0 es un algoritmo memético (híbrido GA + búsqueda local)."""
    ev = Evaluator(C, budget)
    K = ev.K
    nn_in = nn_lists(C) if ls_prob > 0 else None
    popu = np.array([rng.permutation(K) + 1 for _ in range(pop_size)])
    fit = ev.costs(popu)
    while not ev.done:
        order = np.argsort(fit)
        new = [popu[k].copy() for k in order[:elite]]
        new_fit = [fit[k] for k in order[:elite]]
        while len(new) < pop_size and not ev.done:
            a = min(rng.integers(pop_size, size=k_tour), key=lambda k: fit[k])
            b = min(rng.integers(pop_size, size=k_tour), key=lambda k: fit[k])
            child = order_crossover(popu[a], popu[b], rng) if rng.random() < pc else popu[a].copy()
            if rng.random() < pm:
                i, j = sorted(rng.choice(K, size=2, replace=False))
                child[i:j + 1] = child[i:j + 1][::-1]
            c = ev.cost(child)
            if ls_prob > 0 and rng.random() < ls_prob:
                child, c = _local_search(ev, child, c, rng, ls_evals, nn_in)
            new.append(child)
            new_fit.append(c)
        popu = np.array(new)
        fit = np.array(new_fit)
    return ev


def memetic_ga(C, rng, budget: int, **kw):
    """GA + búsqueda local (híbrido). Población más pequeña y búsqueda local ocasional
    pero más larga: con presupuesto igual de evaluaciones fue el mejor balance."""
    kw.setdefault("pop_size", 30)
    kw.setdefault("ls_prob", 0.02)
    kw.setdefault("ls_evals", 2 * (np.asarray(C).shape[0] - 1))
    return ga_ox(C, rng, budget, **kw)


def atsp_exact_milp(C, time_limit: float = 900.0, max_rounds: int = 200, mip_rel_gap: float = 0.0):
    """Óptimo exacto del ATSP con HiGHS: formulación de asignación + cortes de
    eliminación de subciclos (Dantzig–Fulkerson–Johnson) agregados de forma perezosa.
    Sólo se usa como referencia independiente para medir la brecha de optimalidad."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    t0 = time.perf_counter()
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    aid = {a: k for k, a in enumerate(arcs)}
    m = len(arcs)
    cvec = np.array([C[i, j] for i, j in arcs])
    rows, cols = [], []
    for k, (i, j) in enumerate(arcs):
        rows += [i, n + j]
        cols += [k, k]
    A_deg = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * n, m)).tocsr()
    cut_rows = []
    for i in range(n):            # cortes de 2-ciclos desde el inicio
        for j in range(i + 1, n):
            cut_rows.append(([aid[(i, j)], aid[(j, i)]], 1.0))
    rounds = 0
    while True:
        rounds += 1
        r_idx, c_idx, ub = [], [], []
        for r, (idxs, rhs) in enumerate(cut_rows):
            r_idx += [r] * len(idxs)
            c_idx += idxs
            ub.append(rhs)
        A_cut = coo_matrix((np.ones(len(c_idx)), (r_idx, c_idx)), shape=(len(cut_rows), m)).tocsr()
        cons = [LinearConstraint(A_deg, 1, 1), LinearConstraint(A_cut, -np.inf, np.array(ub))]
        left = time_limit - (time.perf_counter() - t0)
        res = milp(cvec, constraints=cons, integrality=np.ones(m), bounds=Bounds(0, 1),
                   options={"time_limit": max(1.0, left), "mip_rel_gap": mip_rel_gap})
        if res.x is None:
            return {"cost": math.inf, "perm": None, "status": "limit", "rounds": rounds,
                    "time_s": time.perf_counter() - t0}
        x = res.x > 0.5
        succ = {}
        for k in np.flatnonzero(x):
            i, j = arcs[k]
            succ[i] = j
        seen, cycles = set(), []
        for s in range(n):
            if s in seen:
                continue
            cyc, u = [], s
            while u not in seen:
                seen.add(u)
                cyc.append(u)
                u = succ[u]
            cycles.append(cyc)
        if len(cycles) == 1:
            perm, u = [], succ[0]
            while u != 0:
                perm.append(u)
                u = succ[u]
            status = "ok" if res.status == 0 else "limit"
            return {"cost": float(cvec[x].sum()), "perm": np.array(perm), "status": status,
                    "rounds": rounds, "time_s": time.perf_counter() - t0}
        for cyc in cycles:
            idxs = [aid[(i, j)] for i in cyc for j in cyc if i != j]
            cut_rows.append((idxs, float(len(cyc) - 1)))
        if rounds >= max_rounds or time.perf_counter() - t0 > time_limit:
            return {"cost": math.inf, "perm": None, "status": "limit", "rounds": rounds,
                    "time_s": time.perf_counter() - t0}


def aco_mmas(C, rng, budget: int, n_ants: int = 20, alpha: float = 1.0, beta: float = 3.0,
             rho: float = 0.1, p_best: float = 0.05, local_search: bool = False,
             ls_evals: int | None = None):
    """MAX–MIN Ant System (Stützle y Hoos) con construcción vectorizada por hormigas.
    Con local_search=True la mejor hormiga de cada iteración se mejora con búsqueda
    local (híbrido ACO + LS)."""
    ev = Evaluator(C, budget)
    Cm = np.asarray(C, dtype=float)
    n = Cm.shape[0]
    K = n - 1
    eta = 1.0 / np.where(Cm > 0, Cm, np.inf)
    eta_b = eta ** beta
    nn = nearest_neighbor(Cm)
    L_nn = tour_cost(Cm, nn)
    tau_max = 1.0 / (rho * L_nn)
    pr = p_best ** (1.0 / n)
    tau_min_ratio = (1 - pr) / ((n / 2 - 1) * pr)
    tau = np.full((n, n), tau_max)
    ls_evals = ls_evals or 4 * K
    nn_in = nn_lists(Cm) if local_search else None
    it = 0
    while not ev.done:
        it += 1
        m = min(n_ants, ev.budget - ev.evals)
        visited = np.zeros((m, n), dtype=bool)
        visited[:, 0] = True
        cur = np.zeros(m, dtype=np.int64)
        tours = np.zeros((m, K), dtype=np.int64)
        attract = (tau ** alpha) * eta_b
        for step in range(K):
            w = attract[cur].copy()
            w[visited] = 0.0
            cs = np.cumsum(w, axis=1)
            r = rng.random(m) * cs[:, -1]
            nxt = (cs < r[:, None]).sum(axis=1)
            nxt = np.minimum(nxt, n - 1)
            # por seguridad numérica: si cae en visitado, toma el primer no visitado
            bad = visited[np.arange(m), nxt]
            if bad.any():
                nxt[bad] = np.argmax(~visited[bad], axis=1)
            tours[:, step] = nxt
            visited[np.arange(m), nxt] = True
            cur = nxt
        costs = ev.costs(tours)
        ib = int(np.argmin(costs))
        best_tour, best_cost = tours[ib].copy(), float(costs[ib])
        if local_search and not ev.done:
            best_tour, best_cost = _local_search(ev, best_tour, best_cost, rng, ls_evals, nn_in)
        # depósito: alterna mejor de la iteración y mejor global
        dep_tour, dep_cost = (best_tour, best_cost) if it % 2 else (ev.best_perm, ev.best)
        tau *= (1.0 - rho)
        path = np.concatenate(([0], dep_tour, [0]))
        tau[path[:-1], path[1:]] += 1.0 / dep_cost
        tau_max = 1.0 / (rho * ev.best)
        tau = np.clip(tau, tau_max * tau_min_ratio, tau_max)
    return ev


def ucb_hyperheuristic(C, rng, budget: int, c_ucb: float = math.sqrt(2.0),
                       t_end_ratio: float = 1e-3):
    """Hiperheurística de selección: UCB1 elige el operador (swap, insert, 2-opt) y la
    aceptación es de recocido simulado (misma temperatura que SA). Recompensa = 1 si el
    movimiento mejora la solución actual."""
    ev = Evaluator(C, budget)
    K = ev.K
    p = rng.permutation(K) + 1
    cur = ev.cost(p)
    T0 = _calibrate_T0(ev, p, cur, rng)
    T = T0
    decay = t_end_ratio ** (1.0 / max(1, budget - ev.evals))
    counts = np.zeros(3)
    rewards = np.zeros(3)
    history = []
    t = 0
    while not ev.done:
        t += 1
        if t <= 3:
            a = t - 1
        else:
            ucb = rewards / counts + c_ucb * np.sqrt(np.log(t) / counts)
            a = int(np.argmax(ucb))
        kind, i, j = random_move(rng, K, MOVES[a])
        q = apply_move(p, kind, i, j)
        c = ev.cost(q)
        d = c - cur
        counts[a] += 1
        rewards[a] += 1.0 if d < 0 else 0.0
        if d <= 0 or rng.random() < math.exp(-d / T):
            p, cur = q, c
        T *= decay
        if ev.evals % ev.step == 0:
            history.append((ev.evals, *(counts / counts.sum())))
    ev.ucb_counts = counts
    ev.ucb_history = history
    return ev
