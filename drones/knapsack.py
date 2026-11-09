"""Selección de la carga útil de un dron: mochila 0/1 con pares de kits incompatibles.

    max  Σ v_i x_i   s.a.  Σ w_i x_i ≤ W,   x_i + x_j ≤ 1  ∀ (i, j) ∈ E,   x ∈ {0,1}^n

Algoritmos
----------
knap_backtracking  backtracking con máscaras de bits (n ≤ 30): `banned` acumula los
                   conflictos de los kits elegidos y se consulta en O(1).
knap_bnb           ramificación y acotamiento con la cota fraccional (relajación lineal
                   de Dantzig sobre los kits no prohibidos): DFS o mejor-primero.
knap_mitm          meet-in-the-middle (Horowitz–Sahni) con conflictos; con
                   objective="weight" carga lo más cerca posible de W (n = 40).
knap_ga            algoritmo genético binario con reparación (referencia bioinspirada).
knap_milp          verificación exacta independiente con HiGHS (scipy.optimize.milp).
"""
from __future__ import annotations

import heapq
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.setrecursionlimit(10000)


@dataclass
class KnapInstance:
    w: np.ndarray            # pesos (enteros)
    v: np.ndarray            # valores (enteros)
    W: int                   # capacidad
    pairs: list              # pares incompatibles (i, j), i < j

    @property
    def n(self):
        return len(self.w)

    def conf_masks(self, order=None):
        """conf[i] = máscara de los kits incompatibles con i (en el orden dado)."""
        n = self.n
        pos = np.arange(n) if order is None else np.argsort(order)
        conf = [0] * n
        for i, j in self.pairs:
            a, b = int(pos[i]), int(pos[j])
            conf[a] |= 1 << b
            conf[b] |= 1 << a
        return conf

    def value(self, sel):
        return int(self.v[list(sel)].sum()) if len(sel) else 0

    def weight(self, sel):
        return int(self.w[list(sel)].sum()) if len(sel) else 0

    def feasible(self, sel) -> bool:
        s = set(int(i) for i in sel)
        if self.weight(s) > self.W:
            return False
        return not any(i in s and j in s for i, j in self.pairs)


def make_instance(pool_w, pool_v, n: int, rng: np.random.Generator, cap_frac: float = 0.4,
                  avg_conflicts: float = 1.5) -> KnapInstance:
    """Instancia con kits muestreados del pool de Kaggle y conflictos Erdős–Rényi."""
    idx = rng.choice(len(pool_w), size=n, replace=False)
    w = np.asarray(pool_w)[idx].astype(np.int64)
    v = np.asarray(pool_v)[idx].astype(np.int64)
    W = int(cap_frac * w.sum())
    p = 0.0 if n < 2 else min(1.0, avg_conflicts / (n - 1))
    iu = np.triu_indices(n, 1)
    mask = rng.random(len(iu[0])) < p
    pairs = list(zip(iu[0][mask].tolist(), iu[1][mask].tolist()))
    return KnapInstance(w, v, W, pairs)


def _density_order(inst: KnapInstance):
    return np.lexsort((-inst.v, -(inst.v / np.maximum(inst.w, 1e-9))))


# ------------------------------------------------------------------ backtracking

def knap_backtracking(inst: KnapInstance, use_bound: bool = True, node_limit: int | None = None):
    """Backtracking con máscaras de bits. Con use_bound=True poda si
    valor_actual + Σ valores restantes ≤ mejor (cota trivial, no fraccional)."""
    t0 = time.perf_counter()
    order = _density_order(inst)
    w = inst.w[order].tolist()
    v = inst.v[order].tolist()
    conf = inst.conf_masks(order)
    n, W = inst.n, inst.W
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + v[i]
    best = [0, 0]
    nodes = [0]

    def rec(i, cw, cv, chosen, banned):
        nodes[0] += 1
        if cv > best[0]:
            best[0], best[1] = cv, chosen
        if i == n:
            return
        if use_bound and cv + suffix[i] <= best[0]:
            return
        if node_limit is not None and nodes[0] >= node_limit:
            return
        bit = 1 << i
        if not (banned & bit) and cw + w[i] <= W:
            rec(i + 1, cw + w[i], cv + v[i], chosen | bit, banned | conf[i])
        rec(i + 1, cw, cv, chosen, banned)

    rec(0, 0, 0, 0, 0)
    sel = sorted(int(order[i]) for i in range(n) if best[1] >> i & 1)
    status = "limit" if node_limit is not None and nodes[0] >= node_limit else "ok"
    return {"value": best[0], "sel": sel, "nodes": nodes[0], "time_s": time.perf_counter() - t0,
            "status": status}


# ------------------------------------------------------------- branch and bound

def _greedy(w, v, conf, W):
    cw = cv = chosen = banned = 0
    for i in range(len(w)):
        if not (banned >> i & 1) and cw + w[i] <= W:
            cw += w[i]
            cv += v[i]
            chosen |= 1 << i
            banned |= conf[i]
    return cv, chosen


def knap_bnb(inst: KnapInstance, strategy: str = "dfs", node_limit: int | None = None,
             time_limit: float | None = None, max_heap: int = 1_500_000):
    """B&B con cota superior fraccional.

    UB(nodo) = valor actual + óptimo de la mochila fraccional sobre los kits
    i ≥ nivel no prohibidos (se relajan los conflictos entre ellos y la integralidad),
    así que UB ≥ cualquier completación factible: la poda UB ≤ mejor es segura.
    """
    t0 = time.perf_counter()
    order = _density_order(inst)
    w = inst.w[order].tolist()
    v = inst.v[order].tolist()
    conf = inst.conf_masks(order)
    n, W = inst.n, inst.W

    def upper(i, cw, cv, banned):
        rem = W - cw
        ub = cv
        for j in range(i, n):
            if banned >> j & 1:
                continue
            if w[j] <= rem:
                rem -= w[j]
                ub += v[j]
            else:
                ub += v[j] * rem / w[j]
                break
        return ub

    best_v, best_s = _greedy(w, v, conf, W)
    nodes = 0
    max_open = 1
    status = "ok"
    if strategy == "dfs":
        # pila explícita: (i, cw, cv, chosen, banned)
        stack = [(0, 0, 0, 0, 0)]
        while stack:
            i, cw, cv, chosen, banned = stack.pop()
            nodes += 1
            if cv > best_v:
                best_v, best_s = cv, chosen
            if i == n:
                continue
            if upper(i, cw, cv, banned) <= best_v:
                continue
            if node_limit is not None and nodes >= node_limit:
                status = "limit"
                break
            if time_limit is not None and (nodes & 1023) == 0 and time.perf_counter() - t0 > time_limit:
                status = "limit"
                break
            bit = 1 << i
            stack.append((i + 1, cw, cv, chosen, banned))            # excluir (se explora después)
            if not (banned & bit) and cw + w[i] <= W:
                stack.append((i + 1, cw + w[i], cv + v[i], chosen | bit, banned | conf[i]))
            if len(stack) > max_open:
                max_open = len(stack)
    else:  # mejor-primero por UB
        heap = [(-upper(0, 0, 0, 0), 0, 0, 0, 0, (0, 0))]
        tie = 0
        while heap:
            negub, _, i, cw, cv, (chosen, banned) = heapq.heappop(heap)
            nodes += 1
            if -negub <= best_v:
                break                     # ningún nodo abierto puede mejorar
            if cv > best_v:
                best_v, best_s = cv, chosen
            if i == n:
                continue
            if node_limit is not None and nodes >= node_limit:
                status = "limit"
                break
            if (nodes & 1023) == 0 and ((time_limit is not None and time.perf_counter() - t0 > time_limit)
                                        or len(heap) > max_heap):
                status = "limit"
                break
            bit = 1 << i
            children = []
            if not (banned & bit) and cw + w[i] <= W:
                children.append((i + 1, cw + w[i], cv + v[i], chosen | bit, banned | conf[i]))
            children.append((i + 1, cw, cv, chosen, banned))
            for (j, a, b, c, d) in children:
                if b > best_v:
                    best_v, best_s = b, c
                ub = upper(j, a, b, d)
                if ub > best_v and j <= n:
                    tie += 1
                    heapq.heappush(heap, (-ub, tie, j, a, b, (c, d)))
            if len(heap) > max_open:
                max_open = len(heap)
    sel = sorted(int(order[i]) for i in range(n) if best_s >> i & 1)
    return {"value": int(best_v), "sel": sel, "nodes": nodes, "max_open": max_open,
            "time_s": time.perf_counter() - t0, "status": status}


# ------------------------------------------------------------ meet in the middle

def _components(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values(), key=len, reverse=True)


def split_for_mitm(inst: KnapInstance):
    """Divide los kits en dos mitades A y B minimizando los conflictos cruzados:
    se empaquetan componentes conexas completas del grafo de conflictos en la mitad
    más pequeña. Devuelve (A, B) como listas de índices."""
    n = inst.n
    A, B = [], []
    for comp in _components(n, inst.pairs):
        (A if len(A) <= len(B) else B).extend(comp)
    # balancear moviendo vértices sueltos si una mitad quedó muy grande
    while abs(len(A) - len(B)) > 1:
        big, small = (A, B) if len(A) > len(B) else (B, A)
        small.append(big.pop())
    return sorted(A), sorted(B)


def _enumerate(items, w, v, conf_full, other_pos):
    """Enumera los 2^m subconjuntos de `items` (vectorizado).

    Devuelve pesos, valores, factibilidad interna, máscara de subconjunto (bits locales)
    y máscara de kits de la otra mitad que quedan prohibidos (bits de la otra mitad).
    """
    local_pos = {it: k for k, it in enumerate(items)}
    W_ = np.zeros(1, dtype=np.int64)
    V_ = np.zeros(1, dtype=np.int64)
    ok = np.ones(1, dtype=bool)
    S = np.zeros(1, dtype=np.int64)
    F = np.zeros(1, dtype=np.int64)
    for k, it in enumerate(items):
        loc = 0
        cross = 0
        for j in conf_full[it]:
            if j in local_pos:
                loc |= 1 << local_pos[j]
            else:
                cross |= 1 << other_pos[j]
        W_ = np.concatenate([W_, W_ + int(w[it])])
        V_ = np.concatenate([V_, V_ + int(v[it])])
        ok = np.concatenate([ok, ok & ((S & loc) == 0)])
        F = np.concatenate([F, F | cross])
        S = np.concatenate([S, S | (1 << k)])
    return W_, V_, ok, S, F


def knap_mitm(inst: KnapInstance, objective: str = "value"):
    """Meet-in-the-middle exacto con conflictos.

    Sea ∂ ⊆ B el conjunto de kits de B con algún conflicto en A. Para cada patrón
    β ⊆ ∂ se combinan los subconjuntos de B cuya intersección con ∂ es β con los
    subconjuntos de A compatibles con β; dentro de cada grupo se aplica el MITM clásico:
    B ordenado por peso con máximo prefijo de valor y búsqueda binaria de W − w(A).
    Complejidad O(2^{|∂|} · 2^{n/2} · n).
    """
    t0 = time.perf_counter()
    n = inst.n
    vals = inst.w if objective == "weight" else inst.v
    conf_full = [[] for _ in range(n)]
    for i, j in inst.pairs:
        conf_full[i].append(j)
        conf_full[j].append(i)
    A, B = split_for_mitm(inst)
    posA = {it: k for k, it in enumerate(A)}
    posB = {it: k for k, it in enumerate(B)}
    wA, vA, okA, SA, FA = _enumerate(A, inst.w, vals, conf_full, posB)
    wB, vB, okB, SB, FB = _enumerate(B, inst.w, vals, conf_full, posA)
    keepA = okA & (wA <= inst.W)
    keepB = okB & (wB <= inst.W)
    wA, vA, SA, FA = wA[keepA], vA[keepA], SA[keepA], FA[keepA]
    wB, vB, SB = wB[keepB], vB[keepB], SB[keepB]
    boundary_mask = 0
    for it in A:
        for j in conf_full[it]:
            if j in posB:
                boundary_mask |= 1 << posB[j]
    bpos = [k for k in range(len(B)) if boundary_mask >> k & 1]
    best_val, best_pair = -1, (0, 0)
    patB = SB & boundary_mask
    for t in range(1 << len(bpos)):
        beta = 0
        for k, bp in enumerate(bpos):
            if t >> k & 1:
                beta |= 1 << bp
        selB = patB == beta
        if not selB.any():
            continue
        selA = (FA & beta) == 0
        if not selA.any():
            continue
        gw, gv, gS = wB[selB], vB[selB], SB[selB]
        o = np.argsort(gw, kind="stable")
        gw, gv, gS = gw[o], gv[o], gS[o]
        pref_idx = np.zeros(len(gv), dtype=np.int64)
        # índice del máximo valor en el prefijo
        run_best = np.maximum.accumulate(gv)
        is_new = np.concatenate([[True], gv[1:] > run_best[:-1]])
        pref_idx = np.maximum.accumulate(np.where(is_new, np.arange(len(gv)), 0))
        aw, av, aS = wA[selA], vA[selA], SA[selA]
        pos = np.searchsorted(gw, inst.W - aw, side="right") - 1
        valid = pos >= 0
        if not valid.any():
            continue
        tot = np.where(valid, av + run_best[np.maximum(pos, 0)], -1)
        k = int(np.argmax(tot))
        if tot[k] > best_val:
            best_val = int(tot[k])
            best_pair = (int(aS[k]), int(gS[pref_idx[pos[k]]]))
    sel = [A[k] for k in range(len(A)) if best_pair[0] >> k & 1] + \
          [B[k] for k in range(len(B)) if best_pair[1] >> k & 1]
    return {"value": best_val, "sel": sorted(sel), "time_s": time.perf_counter() - t0,
            "boundary": len(bpos), "split": (len(A), len(B)),
            "subsets": int(len(wA) + len(wB))}


# --------------------------------------------------------------------- GA (base)

def knap_ga(inst: KnapInstance, rng: np.random.Generator, pop_size: int = 60,
            budget: int = 6000, pc: float = 0.9, pm: float | None = None, record_every: int = 0):
    """GA binario con selección por torneo, cruza uniforme, mutación bit-flip y reparación
    voraz (quita kits de menor densidad hasta ser factible y luego rellena)."""
    t0 = time.perf_counter()
    n = inst.n
    order = _density_order(inst)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    conf = [[] for _ in range(n)]
    for i, j in inst.pairs:
        conf[i].append(j)
        conf[j].append(i)
    w, v, W = inst.w, inst.v, inst.W
    pm = pm if pm is not None else 1.0 / n

    def repair(x):
        keep = np.zeros(n, dtype=bool)
        cw = 0
        for i in order:                       # de mayor a menor densidad
            if x[i] and cw + w[i] <= W and not any(keep[j] for j in conf[i]):
                keep[i] = True
                cw += w[i]
        for i in order:                       # relleno voraz
            if not keep[i] and cw + w[i] <= W and not any(keep[j] for j in conf[i]):
                keep[i] = True
                cw += w[i]
        return keep

    evals = 0
    popu = [repair(rng.random(n) < 0.5) for _ in range(pop_size)]
    fit = np.array([v[p].sum() for p in popu], dtype=float)
    evals += pop_size
    best = float(fit.max())
    best_x = popu[int(fit.argmax())].copy()
    curve = [(evals, best)]
    while evals < budget:
        new = [best_x.copy()]
        while len(new) < pop_size and evals < budget:
            a = max(rng.integers(pop_size, size=3), key=lambda k: fit[k])
            b = max(rng.integers(pop_size, size=3), key=lambda k: fit[k])
            if rng.random() < pc:
                m = rng.random(n) < 0.5
                child = np.where(m, popu[a], popu[b])
            else:
                child = popu[a].copy()
            flip = rng.random(n) < pm
            child = np.where(flip, ~child, child)
            child = repair(child)
            new.append(child)
            evals += 1
            f = float(v[child].sum())
            if f > best:
                best, best_x = f, child.copy()
            if record_every and evals % record_every == 0:
                curve.append((evals, best))
        popu = new
        fit = np.array([v[p].sum() for p in popu], dtype=float)
    return {"value": int(best), "sel": sorted(np.flatnonzero(best_x).tolist()), "evals": evals,
            "time_s": time.perf_counter() - t0, "curve": curve}


# ------------------------------------------------------------- verificaciones

def knap_milp(inst: KnapInstance, objective: str = "value", time_limit: float | None = None,
              return_status: bool = False):
    """Óptimo exacto con HiGHS (para validar los algoritmos propios). Con time_limit puede
    terminar sin certificar el óptimo (status != 0)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    n = inst.n
    c = -(inst.w if objective == "weight" else inst.v).astype(float)
    rows = [inst.w.astype(float)]
    ub = [inst.W]
    for i, j in inst.pairs:
        r = np.zeros(n)
        r[i] = r[j] = 1
        rows.append(r)
        ub.append(1)
    cons = LinearConstraint(np.array(rows), -np.inf, np.array(ub, dtype=float))
    # mip_rel_gap = 0: sin tolerancia relativa (la de HiGHS por omisión, 1e-4, deja pasar
    # soluciones a unos gramos del óptimo cuando los pesos son grandes)
    opts = {"mip_rel_gap": 0.0}
    if time_limit:
        opts["time_limit"] = time_limit
    res = milp(c, constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1), options=opts)
    val = int(round(-res.fun)) if res.fun is not None else None
    if return_status:
        return val, (res.status == 0)
    return val


def knap_dp(w, v, W):
    """Programación dinámica O(nW) para la mochila sin conflictos (validación Kaggle)."""
    best = np.zeros(W + 1, dtype=np.int64)
    for wi, vi in zip(w, v):
        if wi <= W:
            best[wi:] = np.maximum(best[wi:], best[:W + 1 - wi] + vi)
    return int(best[W])
