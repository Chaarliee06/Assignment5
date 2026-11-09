"""
parte4_juego.py - Guardia contra ladron en una cuadricula pequena de la tienda.

Reglas (juego de suma cero, utilidad = ganancia del ladron):
  * Cuadricula n x n. La salida esta en la esquina (0, 0); el ladron entra ahi.
    Hay 3 zonas con articulos de valor 10, 6 y 3 (como en la parte 3).
  * Turnos alternados: el ladron mueve (arriba/abajo/izq/der/quedarse) y luego
    el guardia. Si el ladron pisa una zona con articulo, lo toma.
  * Si el guardia y el ladron quedan en la misma casilla: atrapado, utilidad -20
    (la multa f). Si el ladron llega a la salida con botin: escapa, utilidad =
    botin. Si se acaban los turnos: utilidad 0.

Algoritmos:
  * minimax con profundidad limitada y funcion de evaluacion
  * alfa-beta con orden aleatorio de jugadas y con buen orden (por evaluacion)
  * MCTS con UCT y simulaciones aleatorias
"""
import math
import time

import numpy as np
import pandas as pd

from comun import SEMILLAS, COLORES, guardar_csv, guardar_fig, guardar_json, memoria_pico, plt

MULTA = 20.0
VALORES = (10.0, 6.0, 3.0)
MOVS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


class Juego:
    def __init__(self, n=5, turnos=30, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.turnos = n, turnos
        libres = [(i, j) for i in range(n) for j in range(n) if i + j >= 2]
        idx = rng.choice(len(libres), 4, replace=False)
        self.items = tuple(libres[k] for k in idx[:3])       # posiciones de las 3 zonas
        self.guardia0 = libres[idx[3]]
        self.salida = (0, 0)

    def inicial(self):
        # (ladron, guardia, articulos que quedan (mascara), botin, turno 0=ladron, jugadas restantes)
        return (self.salida, self.guardia0, 0b111, 0.0, 0, self.turnos)

    def terminal(self, s):
        ladron, guardia, _, botin, _, resta = s
        if ladron == guardia:
            return -MULTA
        if ladron == self.salida and botin > 0:
            return botin
        if resta == 0:
            return 0.0
        return None

    def jugadas(self, s):
        pos = s[0] if s[4] == 0 else s[1]
        r = []
        for dx, dy in MOVS:
            x, y = pos[0] + dx, pos[1] + dy
            if 0 <= x < self.n and 0 <= y < self.n:
                r.append((x, y))
        return r

    def aplicar(self, s, a):
        ladron, guardia, mask, botin, turno, resta = s
        if turno == 0:
            ladron = a
            for k, p in enumerate(self.items):
                if mask >> k & 1 and p == ladron:
                    mask &= ~(1 << k)
                    botin += VALORES[k]
        else:
            guardia = a
        return (ladron, guardia, mask, botin, 1 - turno, resta - 1)

    def evaluar(self, s):
        """Heuristica para cortar la busqueda (punto de vista del ladron)."""
        ladron, guardia, mask, botin, _, _ = s
        d = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
        e = botin * 0.9 ** d(ladron, self.salida)
        if mask:
            e += 0.5 * max(VALORES[k] * 0.8 ** d(ladron, self.items[k]) for k in range(3) if mask >> k & 1)
        return e - MULTA * 0.5 ** d(ladron, guardia)


# ---------------------------------------------------------------------------
# Minimax y alfa-beta
# ---------------------------------------------------------------------------
class Buscador:
    def __init__(self, juego, prof, alfabeta=True, orden="bueno", seed=0):
        self.j, self.prof, self.ab, self.orden = juego, prof, alfabeta, orden
        self.rng = np.random.default_rng(seed)
        self.nodos = 0

    def _hijos(self, s):
        hijos = [(a, self.j.aplicar(s, a)) for a in self.j.jugadas(s)]
        if self.orden == "aleatorio":
            self.rng.shuffle(hijos)
        elif self.orden == "bueno":                     # primero las que parecen mejores
            hijos.sort(key=lambda h: self.j.evaluar(h[1]), reverse=(s[4] == 0))
        return hijos

    def valor(self, s, prof, a=-math.inf, b=math.inf):
        self.nodos += 1
        t = self.j.terminal(s)
        if t is not None:
            return t
        if prof == 0:
            return self.j.evaluar(s)
        if s[4] == 0:                                  # ladron: maximiza
            v = -math.inf
            for _, h in self._hijos(s):
                v = max(v, self.valor(h, prof - 1, a, b))
                if self.ab:
                    a = max(a, v)
                    if a >= b:
                        break
            return v
        v = math.inf                                   # guardia: minimiza
        for _, h in self._hijos(s):
            v = min(v, self.valor(h, prof - 1, a, b))
            if self.ab:
                b = min(b, v)
                if a >= b:
                    break
        return v

    def jugar(self, s):
        maxi = s[4] == 0
        mejor, mejor_v = None, -math.inf if maxi else math.inf
        a, b = -math.inf, math.inf
        for mov, h in self._hijos(s):
            v = self.valor(h, self.prof - 1, a, b)
            if (maxi and v > mejor_v) or (not maxi and v < mejor_v):
                mejor, mejor_v = mov, v
            if self.ab:
                if maxi:
                    a = max(a, v)
                else:
                    b = min(b, v)
        return mejor, mejor_v


# ---------------------------------------------------------------------------
# MCTS con UCT
# ---------------------------------------------------------------------------
class Nodo:
    __slots__ = ("s", "padre", "mov", "hijos", "pend", "N", "W")

    def __init__(self, s, padre, mov, jugadas):
        self.s, self.padre, self.mov = s, padre, mov
        self.hijos, self.pend = [], list(jugadas)
        self.N, self.W = 0, 0.0          # visitas y suma de utilidad (del ladron)


class MCTS:
    def __init__(self, juego, iteraciones=500, c=1.4, seed=0):
        self.j, self.it, self.c = juego, iteraciones, c
        self.rng = np.random.default_rng(seed)
        self.nodos = 0

    def _simular(self, s):
        while True:
            t = self.j.terminal(s)
            if t is not None:
                return t
            js = self.j.jugadas(s)
            s = self.j.aplicar(s, js[self.rng.integers(len(js))])

    def jugar(self, s0):
        raiz = Nodo(s0, None, None, self.j.jugadas(s0))
        for _ in range(self.it):
            nodo = raiz
            # 1) seleccion con UCT (el ladron maximiza, el guardia minimiza)
            while not nodo.pend and nodo.hijos:
                signo = 1 if nodo.s[4] == 0 else -1
                ln = math.log(nodo.N)
                nodo = max(nodo.hijos, key=lambda h: signo * h.W / h.N + self.c * math.sqrt(ln / h.N))
            # 2) expansion
            if nodo.pend and self.j.terminal(nodo.s) is None:
                mov = nodo.pend.pop(self.rng.integers(len(nodo.pend)))
                s = self.j.aplicar(nodo.s, mov)
                hijo = Nodo(s, nodo, mov, self.j.jugadas(s) if self.j.terminal(s) is None else [])
                nodo.hijos.append(hijo)
                nodo = hijo
                self.nodos += 1
            # 3) simulacion y 4) retropropagacion (utilidad escalada a [-1, 1])
            r = self._simular(nodo.s) / MULTA
            while nodo is not None:
                nodo.N += 1
                nodo.W += r
                nodo = nodo.padre
        mejor = max(raiz.hijos, key=lambda h: h.N)
        return mejor.mov, mejor.W / mejor.N * MULTA


class Aleatorio:
    def __init__(self, juego, seed=0):
        self.j, self.rng, self.nodos = juego, np.random.default_rng(seed), 0

    def jugar(self, s):
        js = self.j.jugadas(s)
        return js[self.rng.integers(len(js))], 0.0


def partida(juego, ladron, guardia):
    s = juego.inicial()
    tiempos = {0: 0.0, 1: 0.0}
    while juego.terminal(s) is None:
        ag = ladron if s[4] == 0 else guardia
        t0 = time.perf_counter()
        mov, _ = ag.jugar(s)
        tiempos[s[4]] += time.perf_counter() - t0
        s = juego.aplicar(s, mov)
    return juego.terminal(s), tiempos, juego.turnos - s[5]


# ---------------------------------------------------------------------------
def correr(sintetico=False, rapido=False):
    print("[Parte 4] Guardia vs. ladron ...", flush=True)
    semillas = SEMILLAS[:3] if rapido else SEMILLAS
    # ---------------- (a) costo de la busqueda ----------------
    filas = []
    tamanos = [4, 5] if rapido else [4, 5, 6]
    profundidades = [2, 4, 6] if rapido else [2, 4, 6, 8]
    for n in tamanos:
        for prof in profundidades:
            for s in semillas:
                j = Juego(n, seed=s)
                s0 = j.inicial()
                fila = dict(n=n, prof=prof, semilla=s)
                for nombre, kw in (("minimax", dict(alfabeta=False, orden="ninguno")),
                                   ("alfa-beta (orden aleatorio)", dict(alfabeta=True, orden="aleatorio")),
                                   ("alfa-beta (buen orden)", dict(alfabeta=True, orden="bueno"))):
                    if nombre == "minimax" and prof == 8 and n > 4:
                        continue                                    # demasiado lento sin podas
                    b = Buscador(j, prof, seed=s, **kw)
                    t0 = time.perf_counter()
                    _, v = b.jugar(s0)
                    filas.append(dict(fila, metodo=nombre, valor=v, nodos=b.nodos,
                                      tiempo_s=time.perf_counter() - t0))
        print(f"  cuadricula {n}x{n}: listo", flush=True)
    busq = pd.DataFrame(filas)
    # las tres variantes deben dar el mismo valor en la raiz
    chk = busq.groupby(["n", "prof", "semilla"]).valor.agg(lambda x: x.max() - x.min()).max()
    guardar_csv(busq, "p4_busqueda")
    j = Juego(5, seed=0)
    mem = {}
    for nombre, kw in (("minimax", dict(alfabeta=False, orden="ninguno")),
                       ("alfa-beta (buen orden)", dict(alfabeta=True, orden="bueno"))):
        _, mem[nombre] = memoria_pico(Buscador(j, 4, **kw).jugar, j.inicial())
    _, mem["MCTS (500)"] = memoria_pico(MCTS(j, 500).jugar, j.inicial())

    # ---------------- (b) partidas completas ----------------
    agentes = {
        "alfa-beta p=4": lambda j, s: Buscador(j, 4, seed=s),
        "MCTS 500": lambda j, s: MCTS(j, 500, seed=s),
        "aleatorio": lambda j, s: Aleatorio(j, seed=s),
    }
    enfrentamientos = [("alfa-beta p=4", "alfa-beta p=4"), ("alfa-beta p=4", "MCTS 500"),
                       ("MCTS 500", "alfa-beta p=4"), ("MCTS 500", "MCTS 500"),
                       ("alfa-beta p=4", "aleatorio")]
    filas = []
    for lad, gua in enfrentamientos:
        for s in semillas:
            j = Juego(5, turnos=24, seed=s)
            u, t, jug = partida(j, agentes[lad](j, s), agentes[gua](j, s))
            filas.append(dict(ladron=lad, guardia=gua, semilla=s, utilidad=u, escapa=u > 0,
                              atrapado=u == -MULTA, jugadas=jug, t_ladron=t[0], t_guardia=t[1]))
        print(f"  ladron {lad} vs guardia {gua}: listo", flush=True)
    part = pd.DataFrame(filas)
    guardar_csv(part, "p4_partidas")

    g = busq.groupby(["n", "prof", "metodo"]).agg(nodos=("nodos", "mean"), nodos_std=("nodos", "std"),
                                                 tiempo=("tiempo_s", "mean")).reset_index()
    resumen = dict(diferencia_max_valor=float(chk), memoria_mb=mem, busqueda=g.round(5).to_dict("records"),
                   partidas=part.groupby(["ladron", "guardia"]).agg(
                       utilidad=("utilidad", "mean"), utilidad_std=("utilidad", "std"),
                       escapa=("escapa", "mean"), atrapado=("atrapado", "mean"),
                       t_ladron=("t_ladron", "mean"), t_guardia=("t_guardia", "mean")).round(4).reset_index().to_dict("records"))
    print(f"  diferencia maxima de valor entre minimax y alfa-beta: {chk:g}")

    fig, ax = plt.subplots(1, 2, figsize=(7.5, 2.8))
    for k, nombre in enumerate(["minimax", "alfa-beta (orden aleatorio)", "alfa-beta (buen orden)"]):
        h = g[(g.metodo == nombre) & (g.n == 5)]
        ax[0].errorbar(h.prof, h.nodos, h.nodos_std, marker="o", capsize=3, color=COLORES[k], label=nombre)
        h = g[(g.metodo == nombre) & (g.prof == 4)]
        ax[1].plot(h.n, h.nodos, "o-", color=COLORES[k], label=nombre)
    ax[0].set_yscale("log"); ax[0].set_xlabel("profundidad (jugadas)"); ax[0].set_ylabel("nodos expandidos")
    ax[0].set_title("(a) cuadrícula 5×5"); ax[0].legend(fontsize=7)
    ax[1].set_yscale("log"); ax[1].set_xlabel("tamaño de la cuadrícula n"); ax[1].set_ylabel("nodos expandidos")
    ax[1].set_title("(b) profundidad 4"); ax[1].set_xticks(tamanos)
    fig.tight_layout()
    guardar_fig(fig, "p4_busqueda")
    guardar_json(resumen, "p4_resumen")
    return resumen


if __name__ == "__main__":
    import sys
    correr(sintetico="--sintetico" in sys.argv, rapido="--rapido" in sys.argv)
