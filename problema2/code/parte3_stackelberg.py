"""
parte3_stackelberg.py - Colocar guardias como juego de Stackelberg.

La tienda (lider) elige la probabilidad de vigilancia p_z de cada zona y el
ladron (seguidor) responde. Una zona queda disuadida si
        p_z >= (v_z - L) / f
(robar ya no le conviene). Proteger la zona z cuesta c_z = (v_z - L)/f de
presupuesto y evita perder v_z, con  sum c_z <= B.
Elegir que zonas proteger es una mochila 0/1:
        max sum v_z y_z   s.a.  sum c_z y_z <= B,  y_z en {0,1}.

  * Branch & Bound exacto: busqueda en profundidad con cota superior de la
    mochila fraccionaria (Dantzig).
  * ACO: colonia de hormigas que construye soluciones factibles.
Caso del enunciado: v = (10, 6, 3), f = 20, L = 1, B = 0.6  ->  optimo 13.
Zonas de 30 a 120: valores = precios de articulos de Online Retail II.
"""
import numpy as np
import pandas as pd

from comun import SEMILLAS, COLORES, cronometro, guardar_csv, guardar_fig, guardar_json, memoria_pico, plt
import datos

F_MULTA, L_LEGAL = 20.0, 1.0


def costos(v, f=F_MULTA, L=L_LEGAL):
    return np.maximum(v - L, 0) / f


# ---------------------------------------------------------------------------
# Branch & Bound
# ---------------------------------------------------------------------------
def branch_and_bound(v, c, B, max_nodos=30_000_000):
    """Regresa (valor optimo, zonas elegidas, nodos expandidos).
    Si llega a max_nodos se detiene y regresa la mejor solucion encontrada
    (en ese caso nodos == max_nodos y el optimo no esta probado)."""
    orden = np.lexsort((-v, -v / c))              # por valor/costo (empates: mayor valor)
    v, c = v[orden], c[orden]
    n = len(v)
    # sig[i] = primer objeto distinto de i. Si se descarta el objeto i, se
    # descartan tambien sus copias identicas (evita explorar soluciones simetricas)
    sig = list(range(1, n + 1))
    for i in range(n - 2, -1, -1):
        if v[i] == v[i + 1] and c[i] == c[i + 1]:
            sig[i] = sig[i + 1]
    mejor_val, mejor_sel, nodos = 0.0, [], 0

    def cota(i, cap, val):
        # mochila fraccionaria con los objetos i..n-1 (cota superior valida)
        while i < n and c[i] <= cap:
            cap -= c[i]
            val += v[i]
            i += 1
        if i < n:
            val += v[i] * cap / c[i]
        return val

    pila = [(0, B, 0.0, [])]
    while pila and nodos < max_nodos:
        i, cap, val, sel = pila.pop()
        nodos += 1
        if val > mejor_val:
            mejor_val, mejor_sel = val, sel
        if i == n or cota(i, cap, val) <= mejor_val + 1e-12:
            continue                                  # podar
        pila.append((sig[i], cap, val, sel))           # sin el objeto i (ni sus copias)
        if c[i] <= cap + 1e-12:
            pila.append((i + 1, cap - c[i], val + v[i], sel + [i]))   # con el objeto i (se explora primero)
    return mejor_val, sorted(orden[mejor_sel].tolist()), nodos


# ---------------------------------------------------------------------------
# ACO para la mochila
# ---------------------------------------------------------------------------
def aco_mochila(v, c, B, seed, hormigas=20, iteraciones=100, alfa=1.0, beta=2.0, rho=0.1):
    rng = np.random.default_rng(seed)
    n = len(v)
    tau = np.ones(n)
    eta = v / c                                     # heuristica: valor por unidad de costo
    mejor_val, mejor_sel, curva = 0.0, [], []
    for _ in range(iteraciones):
        it_val, it_sel = 0.0, None
        for _ in range(hormigas):
            cap, sel = B, []
            libre = np.ones(n, bool)
            while True:
                cand = libre & (c <= cap + 1e-12)
                if not cand.any():
                    break
                w = np.cumsum((tau ** alfa) * (eta ** beta) * cand)
                j = int(np.searchsorted(w, rng.random() * w[-1], side="right"))   # ruleta
                sel.append(j); libre[j] = False; cap -= c[j]
            val = v[sel].sum()
            if val > it_val:
                it_val, it_sel = val, sel
        if it_val > mejor_val:
            mejor_val, mejor_sel = it_val, it_sel
        tau *= 1 - rho                              # evaporacion
        tau[mejor_sel] += rho * it_val / max(mejor_val, 1e-12) + rho   # deposito del mejor
        curva.append(mejor_val)
    return mejor_val, sorted(mejor_sel), np.array(curva)


# ---------------------------------------------------------------------------
def instancia(precios, Z, seed):
    """Z zonas; el valor de cada zona es el precio real de un articulo (en
    libras, misma escala que v = (10, 6, 3)). Solo se usan articulos con
    v > L, porque los demas nunca conviene robarlos."""
    rng = np.random.default_rng(seed)
    p = precios[precios > L_LEGAL + 0.05]
    v = rng.choice(p, Z, replace=len(p) < Z)
    c = costos(v)
    return v, c, 0.3 * c.sum()                     # presupuesto: 30 % del costo total


def correr(sintetico=False, rapido=False):
    print("[Parte 3] Stackelberg / mochila ...", flush=True)
    # ---------------- Caso del enunciado ----------------
    v = np.array([10.0, 6.0, 3.0]); c = costos(v)
    val, sel, nodos = branch_and_bound(v, c, 0.6)
    val_aco, sel_aco, _ = aco_mochila(v, c, 0.6, 0)
    caso = dict(v=v, costos=c, B=0.6, optimo_bb=val, zonas_bb=[z + 1 for z in sel], nodos=nodos,
                optimo_aco=val_aco, probabilidades={f"zona {z + 1}": c[z] for z in sel})
    print(f"  Caso del enunciado: B&B = {val:g} (zonas {[z + 1 for z in sel]}), ACO = {val_aco:g}")
    assert abs(val - 13) < 1e-9, "El optimo del caso del enunciado debe ser 13"

    # ---------------- Zonas 30-120 con valores reales ----------------
    df = datos.cargar_transacciones(sintetico)
    precios = datos.valores_articulos(df, 120)
    tamanos = [30, 60] if rapido else [30, 60, 90, 120]
    semillas = SEMILLAS[:3] if rapido else SEMILLAS
    filas, curvas = [], {}
    for Z in tamanos:
        for s in semillas:
            v, c, B = instancia(precios, Z, s)
            (opt, _, nodos), t_bb = cronometro(branch_and_bound, v, c, B)
            (va, _, curva), t_aco = cronometro(aco_mochila, v, c, B, s)
            filas.append(dict(Z=Z, semilla=s, optimo=opt, nodos_bb=nodos, optimo_probado=nodos < 30_000_000,
                              t_bb=t_bb, aco=va,
                              brecha_aco_pct=100 * (opt - va) / opt, t_aco=t_aco))
            curvas.setdefault(Z, []).append(100 * (opt - curva) / opt)
        v, c, B = instancia(precios, Z, 0)
        _, m_bb = memoria_pico(branch_and_bound, v, c, B)
        _, m_aco = memoria_pico(aco_mochila, v, c, B, 0)
        for fila in filas[-len(semillas):]:
            fila.update(mem_bb_mb=m_bb, mem_aco_mb=m_aco)
        print(f"  Z={Z}: listo", flush=True)
    res = pd.DataFrame(filas)
    guardar_csv(res, "p3_mochila")
    tabla = res.groupby("Z").agg(["mean", "std"])
    resumen = dict(caso_enunciado=caso, tabla=res.groupby("Z").mean(numeric_only=True).round(5).reset_index().to_dict("records"),
                   tabla_std=res.groupby("Z").std(numeric_only=True).round(5).reset_index().to_dict("records"))

    fig, ax = plt.subplots(1, 3, figsize=(10, 2.7))
    for col, mk, et in (("t_bb", "o", "B&B"), ("t_aco", "s", "ACO")):     # media, barras = min y max
        g = res.groupby("Z")[col].agg(["mean", "min", "max"]) * 1e3
        ax[0].errorbar(g.index, g["mean"], [g["mean"] - g["min"], g["max"] - g["mean"]], marker=mk, capsize=3, label=et)
    ax[0].set_yscale("log"); ax[0].set_xlabel("zonas Z"); ax[0].set_ylabel("tiempo [ms]"); ax[0].legend(fontsize=7)
    ax[0].set_title("(a) tiempo (media, mín.–máx.)")
    ax[1].errorbar(tabla.index, tabla[("brecha_aco_pct", "mean")], tabla[("brecha_aco_pct", "std")], marker="s",
                   capsize=3, color=COLORES[1])
    ax[1].set_xlabel("zonas Z"); ax[1].set_ylabel("brecha ACO vs. óptimo [%]"); ax[1].set_title("(b) calidad de ACO")
    for k, Z in enumerate(tamanos):
        Cv = np.array(curvas[Z])
        ax[2].plot(np.arange(1, Cv.shape[1] + 1), Cv.mean(0), color=COLORES[k], label=f"Z={Z}")
    ax[2].set_xlabel("iteración de ACO"); ax[2].set_ylabel("brecha [%]"); ax[2].set_yscale("symlog", linthresh=0.1)
    ax[2].set_title("(c) convergencia de ACO"); ax[2].legend(fontsize=7)
    for a in ax[:2]:
        a.set_xticks(tamanos)
    fig.tight_layout()
    guardar_fig(fig, "p3_mochila")
    guardar_json(resumen, "p3_resumen")
    return resumen


if __name__ == "__main__":
    import sys
    correr(sintetico="--sintetico" in sys.argv, rapido="--rapido" in sys.argv)
