"""
pruebas.py - Pruebas de correctitud del Problema 2.

No necesitan los datasets de Kaggle ni PyTorch:
  1. Caso del enunciado: v = (10, 6, 3), f = 20, L = 1, B = 0.6  ->  optimo 13
  2. Branch & Bound de la mochila contra fuerza bruta (300 instancias chicas)
  3. Alfa-beta (orden aleatorio y buen orden) da el mismo valor que minimax
  4. Gradiente de la perdida logistica contra diferencias finitas y
     Hessiana semidefinida positiva (convexidad)
  5. La cobertura de Stackelberg disuade las tres zonas

Uso:  python3 pruebas.py
"""
import itertools
import time

import numpy as np

from parte1_logistica_em import con_sesgo, gradiente_logistica, hessiana_logistica, perdida_logistica
from parte3_stackelberg import branch_and_bound, costos
from parte4_juego import Buscador, Juego
from parte5_aprendizaje import P_CAMARA, P_GUARDIA, UMBRAL, politica_stackelberg


def prueba_enunciado():
    v = np.array([10.0, 6.0, 3.0])
    c = costos(v)
    val, sel, _ = branch_and_bound(v, c, 0.6)
    assert abs(val - 13) < 1e-9 and sel == [0, 2], (val, sel)
    return "optimo 13 con las zonas 1 y 3"


def fuerza_bruta(v, c, B):
    mejor = 0.0
    for sel in itertools.product((False, True), repeat=len(v)):
        s = np.array(sel)
        if c[s].sum() <= B + 1e-12:
            mejor = max(mejor, v[s].sum())
    return mejor


def prueba_mochila(n_instancias=300):
    rng = np.random.default_rng(2026)
    for k in range(n_instancias):
        n = int(rng.integers(1, 13))
        if k % 2:
            v = np.round(rng.uniform(1.1, 15.0, n), 2)       # valores continuos
        else:
            v = rng.choice([2.0, 3.0, 6.0, 10.0], n)          # valores repetidos (copias identicas)
        c = costos(v)
        B = rng.uniform(0.1, 0.7) * c.sum()
        val, sel, _ = branch_and_bound(v, c, B)
        assert abs(val - fuerza_bruta(v, c, B)) < 1e-9, f"instancia {k}: el B&B no encontro el optimo"
        assert abs(v[sel].sum() - val) < 1e-9 and c[sel].sum() <= B + 1e-9, f"instancia {k}: solucion invalida"
    return f"{n_instancias} instancias, todas optimas"


def prueba_alfabeta():
    casos = 0
    for n in (4, 5):
        for prof in (2, 4, 6):
            for s in range(5):
                j = Juego(n, seed=s)
                s0 = j.inicial()
                ref = Buscador(j, prof, alfabeta=False, orden="ninguno").jugar(s0)[1]
                for orden in ("aleatorio", "bueno"):
                    v = Buscador(j, prof, alfabeta=True, orden=orden, seed=s).jugar(s0)[1]
                    assert abs(v - ref) < 1e-9, (n, prof, s, orden, v, ref)
                casos += 1
    return f"{casos} posiciones con el mismo valor en la raiz"


def prueba_logistica():
    rng = np.random.default_rng(0)
    X = con_sesgo(rng.normal(size=(200, 4)))
    y = (rng.random(200) < 0.4).astype(float)
    lam, h = 1e-3, 1e-6
    w = rng.normal(size=X.shape[1])
    g = gradiente_logistica(w, X, y, lam)
    num = np.array([(perdida_logistica(w + h * e, X, y, lam) - perdida_logistica(w - h * e, X, y, lam)) / (2 * h)
                    for e in np.eye(len(w))])
    assert np.allclose(g, num, atol=1e-6), (g, num)
    for _ in range(20):
        w = rng.normal(0, 3, X.shape[1])
        assert np.linalg.eigvalsh(hessiana_logistica(w, X, y, lam)).min() > -1e-12
    return "gradiente correcto y Hessiana semidefinida positiva en 20 puntos"


def prueba_stackelberg():
    c = politica_stackelberg()
    atrapar = P_GUARDIA * c + P_CAMARA * (1 - c)
    assert abs(c.sum() - 1) < 1e-12 and np.all(atrapar >= UMBRAL - 1e-12), (c, atrapar)
    return "cobertura (" + ", ".join(f"{x:.2f}" for x in c) + ") disuade las 3 zonas"


PRUEBAS = [
    ("Caso del enunciado", prueba_enunciado),
    ("B&B contra fuerza bruta", prueba_mochila),
    ("Alfa-beta contra minimax", prueba_alfabeta),
    ("Perdida logistica", prueba_logistica),
    ("Cobertura de Stackelberg", prueba_stackelberg),
]

if __name__ == "__main__":
    for nombre, prueba in PRUEBAS:
        t0 = time.perf_counter()
        detalle = prueba()
        print(f"[ok] {nombre}: {detalle} ({time.perf_counter() - t0:.1f} s)")
    print("Todas las pruebas pasaron.")
