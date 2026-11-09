"""Asignación de entregas a plataformas de aterrizaje y turnos como CSP.

Variables  X_d, una por entrega d = 1..D.
Dominios   D_d = {(p, t) : p ∈ plataformas permitidas para d, t ∈ [e_d, l_d]}
           (ventana de llegada del dron y restricción de plataformas reforzadas).
Restricción binaria entre todo par d ≠ d' con valores a = (p, t), b = (p', t'):
           t ≠ t'  ∨  p' ∉ N[p],   con N[p] = {p} ∪ vecinos(p)
           (un dron por plataforma por turno y nunca plataformas adyacentes a la vez).

Los dominios se guardan como bitsets (enteros de Python): el valor (p, t) es el bit
t·P + p y conflict[a] es la máscara de todos los valores incompatibles con a, de modo
que forward checking es dom[x] &= ~conflict[a].

Algoritmos: backtracking cronológico (BT), forward checking (FC), MAC con AC-3,
ordenamiento de variables MRV (+grado) y de valores LCV, AC-3 como preprocesamiento,
y min-conflicts (búsqueda local, incompleta) como referencia.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class PadCSP:
    rows: int
    cols: int
    T: int
    domains: list          # bitsets iniciales
    windows: list          # (e_d, l_d)
    heavy: list            # bool por entrega
    adjacency: str = "4"

    def __post_init__(self):
        self.P = self.rows * self.cols
        self.D = len(self.domains)
        self.nbrs = []
        for p in range(self.P):
            r, c = divmod(p, self.cols)
            nb = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) == (0, 0):
                        continue
                    if self.adjacency == "4" and dr and dc:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.rows and 0 <= cc < self.cols:
                        nb.append(rr * self.cols + cc)
            self.nbrs.append(nb)
        self.conflict = []
        for a in range(self.P * self.T):
            t, p = divmod(a, self.P)
            m = 1 << a
            for q in self.nbrs[p]:
                m |= 1 << (t * self.P + q)
            self.conflict.append(m)
        # grado estático: número de entregas cuya ventana se traslapa con la de d
        self.degree = []
        for d, (e, l) in enumerate(self.windows):
            self.degree.append(sum(1 for k, (e2, l2) in enumerate(self.windows)
                                   if k != d and e2 <= l and e <= l2))

    def decode(self, a):
        t, p = divmod(a, self.P)
        return p, t

    def check(self, assignment) -> bool:
        """Verificador independiente de una asignación completa."""
        if any(a is None for a in assignment):
            return False
        for d, a in enumerate(assignment):
            if not (self.domains[d] >> a & 1):
                return False
        for d in range(self.D):
            pd, td = self.decode(assignment[d])
            for k in range(d + 1, self.D):
                pk, tk = self.decode(assignment[k])
                if td == tk and (pd == pk or pk in self.nbrs[pd]):
                    return False
        return True


def make_csp(D: int, rows: int, cols: int, T: int, rng: np.random.Generator,
             window=(2, 4), heavy_frac: float = 0.3, adjacency: str = "4") -> PadCSP:
    """Instancia aleatoria. Las plataformas reforzadas (para drones pesados) son las de
    la primera y última fila; un dron pesado sólo puede usar ésas."""
    P = rows * cols
    reinforced = [p for p in range(P) if p // cols in (0, rows - 1)]
    domains, windows, heavy = [], [], []
    for _ in range(D):
        wl = int(rng.integers(window[0], window[1] + 1))
        e = int(rng.integers(0, max(1, T - wl + 1)))
        l = min(T - 1, e + wl - 1)
        hv = bool(rng.random() < heavy_frac)
        pads = reinforced if hv else range(P)
        m = 0
        for t in range(e, l + 1):
            for p in pads:
                m |= 1 << (t * P + p)
        domains.append(m)
        windows.append((e, l))
        heavy.append(hv)
    return PadCSP(rows, cols, T, domains, windows, heavy, adjacency)


def _bits(m):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


# ----------------------------------------------------------------------- AC-3

def ac3(csp: PadCSP, dom: list, unassigned=None, queue=None, stats=None) -> bool:
    """AC-3 sobre los dominios `dom` (se modifican en sitio). False si hay dominio vacío.

    revise(Xi, Xj): a ∈ D_i tiene soporte en D_j sii D_j & ~conflict[a] ≠ 0.
    """
    vars_ = list(range(csp.D)) if unassigned is None else list(unassigned)
    if queue is None:
        queue = deque((i, j) for i in vars_ for j in vars_ if i != j)
    in_q = set(queue)
    conflict = csp.conflict
    while queue:
        i, j = queue.popleft()
        in_q.discard((i, j))
        if stats is not None:
            stats["revisions"] += 1
        dj = dom[j]
        removed = 0
        for a in _bits(dom[i]):
            if not (dj & ~conflict[a]):
                removed |= 1 << a
        if removed:
            dom[i] &= ~removed
            if stats is not None:
                stats["pruned"] += removed.bit_count()
            if not dom[i]:
                return False
            for k in vars_:
                if k != i and k != j and (k, i) not in in_q:
                    queue.append((k, i))
                    in_q.add((k, i))
    return True


# ------------------------------------------------------------------ búsqueda

def solve(csp: PadCSP, inference: str = "fc", var_order: str = "mrv", val_order: str = "lcv",
          preprocess_ac3: bool = False, node_limit: int = 200_000, time_limit: float = 10.0):
    """Backtracking con inferencia 'none' (BT), 'fc' o 'mac' (AC-3 mantenido).

    var_order: 'static' (índice) o 'mrv' (mínimos valores restantes, desempate por grado).
    val_order: 'natural' o 'lcv' (menos restrictivo: elimina menos valores ajenos).
    Devuelve dict con status ('sat' | 'unsat' | 'limit'), asignación y contadores.
    """
    t0 = time.perf_counter()
    stats = {"nodes": 0, "backtracks": 0, "checks": 0, "revisions": 0, "pruned": 0}
    D = csp.D
    dom = list(csp.domains)
    conflict = csp.conflict
    degree = csp.degree
    if preprocess_ac3 or inference == "mac":
        if not ac3(csp, dom, stats=stats):
            return {"status": "unsat", "assignment": None, "time_s": time.perf_counter() - t0, **stats}
    assign = [None] * D
    unassigned = set(range(D))
    used = 0                       # OR de conflict[a] de los valores asignados (para BT)
    limit_hit = [False]

    def legal(x):
        """Valores legales restantes de x. Con FC/MAC los dominios ya están podados; en BT
        los dominios no cambian, así que hay que quitar los valores en conflicto con lo
        asignado (máscara `used`)."""
        return dom[x] & ~used if inference == "none" else dom[x]

    def select_var():
        if var_order == "static":
            return min(unassigned)
        best, key = None, None
        for x in unassigned:
            k = (legal(x).bit_count(), -degree[x], x)
            if key is None or k < key:
                best, key = x, k
        return best

    def order_vals(x):
        vals = list(_bits(dom[x]))
        if val_order == "lcv":
            others = [dom[y] for y in unassigned if y != x]
            vals.sort(key=lambda a: sum((dy & conflict[a]).bit_count() for dy in others))
        return vals

    def backtrack():
        nonlocal used
        if not unassigned:
            return True
        if stats["nodes"] >= node_limit or (stats["nodes"] & 255) == 0 and time.perf_counter() - t0 > time_limit:
            limit_hit[0] = True
            return False
        x = select_var()
        unassigned.discard(x)
        for a in order_vals(x):
            stats["nodes"] += 1
            if inference == "none":
                stats["checks"] += 1
                if used >> a & 1:
                    continue
            saved = None
            ok = True
            if inference in ("fc", "mac"):
                saved = dom[:]
                dom[x] = 1 << a
                ca = conflict[a]
                changed = []
                for y in unassigned:
                    stats["checks"] += 1
                    ny = dom[y] & ~ca
                    if ny != dom[y]:
                        stats["pruned"] += (dom[y] ^ ny).bit_count()
                        dom[y] = ny
                        changed.append(y)
                        if not ny:
                            ok = False
                            break
                if ok and inference == "mac" and changed:
                    # MAC: las reducciones que hizo forward checking deben propagarse entre las
                    # variables restantes, así que la cola empieza con los arcos (z, y) hacia
                    # cada variable y cuyo dominio cambió.
                    rest = list(unassigned)
                    q = deque((z, y) for y in changed for z in rest if z != y)
                    ok = ac3(csp, dom, unassigned=rest, queue=q, stats=stats)
            if ok:
                assign[x] = a
                prev_used = used
                used |= conflict[a]
                if backtrack():
                    return True
                used = prev_used
                assign[x] = None
                if limit_hit[0]:
                    if saved is not None:
                        dom[:] = saved
                    break
            if saved is not None:
                dom[:] = saved
            stats["backtracks"] += 1
        unassigned.add(x)
        return False

    found = backtrack()
    status = "sat" if found else ("limit" if limit_hit[0] else "unsat")
    out = {"status": status, "assignment": assign[:] if found else None,
           "time_s": time.perf_counter() - t0, **stats}
    if found:
        out["valid"] = csp.check(assign)
    return out


def min_conflicts(csp: PadCSP, rng: np.random.Generator, max_steps: int = 20_000):
    """Búsqueda local min-conflicts (incompleta: no puede demostrar insatisfacibilidad).

    occ[v] cuenta cuántas entregas usan el valor v, de modo que los conflictos de un valor a
    son Σ occ[v] sobre los bits de conflict[a] (a lo sumo 9): cada paso cuesta O(D + |D_d|).
    """
    t0 = time.perf_counter()
    vals = [list(_bits(d)) for d in csp.domains]
    if any(len(v) == 0 for v in vals):
        return {"status": "fail", "steps": 0, "time_s": time.perf_counter() - t0}
    nb = [list(_bits(m)) for m in csp.conflict]
    occ = [0] * (csp.P * csp.T)
    assign = [v[int(rng.integers(len(v)))] for v in vals]
    for a in assign:
        occ[a] += 1
    D = csp.D
    for step in range(max_steps):
        conflicted = [d for d in range(D) if sum(occ[v] for v in nb[assign[d]]) > 1]
        if not conflicted:
            return {"status": "sat", "steps": step, "assignment": assign,
                    "valid": csp.check(assign), "time_s": time.perf_counter() - t0}
        d = conflicted[int(rng.integers(len(conflicted)))]
        occ[assign[d]] -= 1
        scores = [sum(occ[v] for v in nb[a]) for a in vals[d]]
        m = min(scores)
        cands = [a for a, s in zip(vals[d], scores) if s == m]
        assign[d] = cands[int(rng.integers(len(cands)))]
        occ[assign[d]] += 1
    return {"status": "fail", "steps": max_steps, "time_s": time.perf_counter() - t0}
