"""
parte2_precios.py - Precios de d productos relacionados (ingreso NO convexo).

Modelo de demanda del producto i (x_i = precio_i / precio de referencia_i):
    D_i(x) = q_i * exp(-e (x_i - 1)) * [ (1 - lam) + lam * s(k (c_i - x_i)) ]
                 * prod_{j misma categoria} (x_j / x_i)^g
  * q_i, precio de referencia y c_i (precio del competidor comp_1, relativo)
    salen de Retail Price Optimization.
  * lam = fraccion de clientes que comparan precios = peso de los segmentos
    de bajo gasto que encontro EM en la parte 1. Si el precio pasa al del
    competidor, esos clientes se van (s = sigmoide).
  * El ultimo factor relaciona productos de la misma categoria (sustitutos).
Ingreso:  R(x) = sum_i precio_ref_i * x_i * D_i(x).
R no es concava: para cada producto conviene quedarse justo debajo del
competidor (atraer a los que comparan) o subir mucho el precio (solo a los
leales), asi que hay muchos maximos locales.

Metodos (todos con el mismo presupuesto de evaluaciones de R):
  gradiente con reinicios, Hill Climbing, Recocido Simulado, GA real, PSO
  y (extra) GA memetico = GA + hill climbing sobre el mejor individuo.
"""
import numpy as np
import pandas as pd

from comun import (SEMILLAS, COLORES, banda, cronometro, guardar_csv, guardar_fig,
                   guardar_json, leer_json, memoria_pico, plt)
import datos

LO, HI = 0.5, 2.5            # x_i = precio / precio de referencia, en [0.5, 2.5]


class Ingreso:
    """Funcion objetivo. Cuenta cuantas veces se evalua y guarda la mejor."""

    def __init__(self, prods, segmentos, e=0.6, k=12.0, g=0.3):
        self.ref = prods["precio"].values
        self.q = prods["demanda"].values
        self.c = (prods["competidor"] / prods["precio"]).values
        pesos, gasto = np.array(segmentos["pesos"]), np.array(segmentos["gasto_medio"])
        self.lam = float(pesos[gasto < np.median(gasto)].sum() or pesos[0])
        cat = np.asarray(prods["categoria"].astype(str))
        self.M = (cat[:, None] == cat[None, :]) & ~np.eye(len(cat), dtype=bool)   # misma categoria
        self.e, self.k, self.g, self.d = e, k, g, len(cat)
        self.reset(10 ** 9)

    def reset(self, presupuesto):
        self.evals, self.presupuesto = 0, presupuesto
        self.mejor, self.hist = -np.inf, []

    def __call__(self, x):
        if self.evals >= self.presupuesto:
            raise StopIteration
        self.evals += 1
        x = np.clip(x, LO, HI)
        compara = (1 - self.lam) + self.lam / (1 + np.exp(-self.k * (self.c - x)))
        lx = np.log(x)
        cruzado = np.exp(self.g * (self.M @ lx - self.M.sum(1) * lx))
        D = self.q * np.exp(-self.e * (x - 1)) * compara * cruzado
        r = float(np.sum(self.ref * x * D))
        if r > self.mejor:
            self.mejor = r
        self.hist.append(self.mejor)
        return r


# ---------------------------------------------------------------------------
# Metodos de optimizacion. Todos reciben f, d, rng y se detienen cuando
# f lanza StopIteration (se acabo el presupuesto).
# ---------------------------------------------------------------------------
def gradiente_reinicios(f, d, rng, paso=0.05, h=1e-4, max_it=40):
    """Ascenso de gradiente (gradiente por diferencias centrales, 2d evaluaciones)
    reiniciando desde un punto al azar cada max_it pasos."""
    while True:
        x = rng.uniform(LO, HI, d)
        fx = f(x)
        for _ in range(max_it):
            g = np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(d)])
            g /= np.linalg.norm(g) + 1e-12
            nuevo = np.clip(x + paso * g, LO, HI)
            fn = f(nuevo)
            if fn <= fx:
                break
            x, fx = nuevo, fn


def hill_climbing(f, d, rng, sigma=0.1, paciencia=100):
    """Cambia una coordenada al azar; acepta si mejora. Reinicia si se atora."""
    while True:
        x = rng.uniform(LO, HI, d)
        fx, sin_mejora = f(x), 0
        while sin_mejora < paciencia:
            y = x.copy()
            i = rng.integers(d)
            y[i] = np.clip(y[i] + rng.normal(0, sigma), LO, HI)
            fy = f(y)
            if fy > fx:
                x, fx, sin_mejora = y, fy, 0
            else:
                sin_mejora += 1


def recocido_simulado(f, d, rng, sigma=0.1, presupuesto=None):
    """Acepta peores soluciones con prob exp(delta/T); T baja geometricamente."""
    x = rng.uniform(LO, HI, d)
    fx = f(x)
    T0 = 0.05 * abs(fx) + 1e-9
    Tf = 1e-4 * T0
    k = 0
    while True:
        T = T0 * (Tf / T0) ** (k / presupuesto)
        y = x.copy()
        i = rng.integers(d)
        y[i] = np.clip(y[i] + rng.normal(0, sigma), LO, HI)
        fy = f(y)
        if fy > fx or rng.random() < np.exp((fy - fx) / T):
            x, fx = y, fy
        k += 1


def _hc_local(f, x, fx, rng, pasos, sigma=0.05):
    for _ in range(pasos):
        y = x.copy()
        i = rng.integers(len(x))
        y[i] = np.clip(y[i] + rng.normal(0, sigma), LO, HI)
        fy = f(y)
        if fy > fx:
            x, fx = y, fy
    return x, fx


def algoritmo_genetico(f, d, rng, pob=40, pc=0.9, alfa=0.5, memetico=False):
    """GA de codificacion real: torneo de 3, cruce BLX-alfa, mutacion
    Gaussiana (prob 1/d por gen) y elitismo de 2.
    memetico=True: ademas aplica hill climbing (2d pasos) al mejor de cada generacion."""
    X = rng.uniform(LO, HI, (pob, d))
    F = np.array([f(x) for x in X])
    while True:
        orden = np.argsort(-F)
        hijos = [X[orden[0]].copy(), X[orden[1]].copy()]          # elitismo
        while len(hijos) < pob:
            p1 = X[max(rng.choice(pob, 3), key=lambda i: F[i])]
            p2 = X[max(rng.choice(pob, 3), key=lambda i: F[i])]
            if rng.random() < pc:
                lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
                rango = hi - lo
                h = rng.uniform(lo - alfa * rango, hi + alfa * rango)
            else:
                h = p1.copy()
            m = rng.random(d) < 1.0 / d
            h[m] += rng.normal(0, 0.1 * (HI - LO), m.sum())
            hijos.append(np.clip(h, LO, HI))
        X = np.array(hijos)
        F = np.concatenate([F[orden[:2]], [f(x) for x in X[2:]]])
        if memetico:
            b = np.argmax(F)
            X[b], F[b] = _hc_local(f, X[b], F[b], rng, 2 * d)


def pso(f, d, rng, n=30, w=0.72, c1=1.49, c2=1.49):
    """PSO global estandar (coeficientes de Clerc)."""
    X = rng.uniform(LO, HI, (n, d))
    V = rng.uniform(-0.1, 0.1, (n, d)) * (HI - LO)
    P = X.copy()
    FP = np.array([f(x) for x in X])
    g = P[np.argmax(FP)].copy()
    while True:
        r1, r2 = rng.random((n, d)), rng.random((n, d))
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
        X = np.clip(X + V, LO, HI)
        for k in range(n):
            fk = f(X[k])
            if fk > FP[k]:
                P[k], FP[k] = X[k].copy(), fk
        g = P[np.argmax(FP)].copy()


METODOS = {
    "Gradiente + reinicios": gradiente_reinicios,
    "Hill Climbing": hill_climbing,
    "Recocido simulado": recocido_simulado,
    "GA real": algoritmo_genetico,
    "PSO": pso,
    "GA memético (extra)": lambda f, d, rng: algoritmo_genetico(f, d, rng, memetico=True),
}


def correr_metodo(f, nombre, d, seed, presupuesto):
    f.reset(presupuesto)
    rng = np.random.default_rng(seed)
    fn = METODOS[nombre]
    try:
        if nombre == "Recocido simulado":
            fn(f, d, rng, presupuesto=presupuesto)
        else:
            fn(f, d, rng)
    except StopIteration:
        pass
    return f.mejor, np.array(f.hist)


# ---------------------------------------------------------------------------
def correr(sintetico=False, rapido=False):
    print("[Parte 2] Precios de productos relacionados ...", flush=True)
    segmentos = leer_json("segmentos")
    dims = [10, 20] if rapido else [10, 20, 40]
    semillas = SEMILLAS[:3] if rapido else SEMILLAS
    filas, curvas = [], {}
    resumen = dict(dims=dims, presupuesto_por_d=300, segmentos=segmentos["K"])
    for d in dims:
        prods = datos.productos(d, sintetico)
        f = Ingreso(prods, segmentos)
        E = 300 * d                                      # mismo presupuesto para todos
        if d == 10:
            resumen["productos_d10"] = prods[["productid", "categoria", "precio", "competidor", "demanda"]].round(2).to_dict("records")
            resumen["lam_comparan_precio"] = f.lam
            resumen["competidor_relativo"] = f.c.round(3)
            resumen["ingreso_precios_actuales"] = f(np.ones(d))
        for nombre in METODOS:
            for s in semillas:
                (val, hist), t = cronometro(correr_metodo, f, nombre, d, s, E)
                filas.append(dict(d=d, metodo=nombre, semilla=s, ingreso=val, tiempo_s=t, evaluaciones=E))
                if d == 10:
                    curvas.setdefault(nombre, []).append(hist)
            _, mem = memoria_pico(correr_metodo, f, nombre, d, 0, E)
            for fila in filas[-len(semillas):]:
                fila["memoria_mb"] = mem
        print(f"  d={d}: listo", flush=True)
    df = pd.DataFrame(filas)
    # Brecha contra el mejor valor encontrado por cualquier metodo (referencia)
    mejor = df.groupby("d").ingreso.transform("max")
    df["brecha_pct"] = 100 * (mejor - df.ingreso) / mejor
    guardar_csv(df, "p2_precios")
    tabla = df.groupby(["d", "metodo"]).agg(ingreso=("ingreso", "mean"), ingreso_std=("ingreso", "std"),
                                            brecha=("brecha_pct", "mean"), brecha_std=("brecha_pct", "std"),
                                            tiempo=("tiempo_s", "mean"), memoria=("memoria_mb", "mean")).reset_index()
    guardar_csv(tabla, "p2_tabla")
    resumen["tabla"] = tabla.round(4).to_dict("records")

    # ---------------- Evidencia de no convexidad (d = 10) ----------------
    prods = datos.productos(10, sintetico)
    f = Ingreso(prods, segmentos)
    finales = []
    for s in range(30 if rapido else 100):
        f.reset(10 ** 9)
        rng = np.random.default_rng(1000 + s)
        x = rng.uniform(LO, HI, 10)
        # ascenso de gradiente hasta converger desde un punto al azar
        for _ in range(400):
            g = np.array([(f(x + 1e-4 * e) - f(x - 1e-4 * e)) / 2e-4 for e in np.eye(10)])
            if np.linalg.norm(g) < 1e-3:
                break
            x = np.clip(x + 0.02 * g / (np.linalg.norm(g) + 1e-12) * min(1.0, np.linalg.norm(g)), LO, HI)
        finales.append(f(x))
    finales = np.array(finales)
    distintos = np.unique(np.round(finales / finales.max(), 3))
    resumen["optimos_locales"] = dict(n_inicios=len(finales), valores_distintos=len(distintos),
                                      min=finales.min(), max=finales.max())
    # Corte 1D: el ingreso de un producto con los demas fijos
    xs = np.linspace(LO, HI, 200)
    i_mejor = df[(df.d == 10)].sort_values("ingreso").index[-1]
    corte = []
    for v in xs:
        z = np.ones(10); z[0] = v
        corte.append(f(z))

    # ---------------- Figuras ----------------
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for k, (nombre, hs) in enumerate(curvas.items()):
        L = min(len(h) for h in hs)
        ev = np.arange(1, L + 1)
        banda(ax[0], ev, [h[:L] for h in hs], nombre, COLORES[k])
    ax[0].set_xlabel("evaluaciones de R"); ax[0].set_ylabel("mejor ingreso encontrado")
    ax[0].set_title("(a) convergencia, d = 10"); ax[0].legend(fontsize=6)
    lo = np.percentile(np.concatenate([h[len(h) // 10:] for hs in curvas.values() for h in hs]), 2)
    ax[0].set_ylim(bottom=lo)
    for k, nombre in enumerate(METODOS):
        g = tabla[tabla.metodo == nombre]
        ax[1].errorbar(g.d + (k - 2.5) * 0.4, g.brecha, g.brecha_std, marker="o", ms=3, capsize=2,
                       color=COLORES[k], label=nombre, lw=1)
    ax[1].set_xlabel("número de productos d"); ax[1].set_ylabel("brecha vs. mejor [%]")
    ax[1].set_title("(b) calidad final (10 semillas)"); ax[1].set_xticks(dims)
    ax[2].hist(finales, bins=30, color=COLORES[0])
    ax[2].set_xlabel("ingreso al converger"); ax[2].set_ylabel("número de inicios")
    ax[2].set_title(f"(c) ascenso de gradiente desde {len(finales)} inicios")
    fig.tight_layout()
    guardar_fig(fig, "p2_precios")

    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for k, nombre in enumerate(METODOS):
        g = tabla[tabla.metodo == nombre]
        ax[0].plot(g.d, g.tiempo * 1e3, "o-", color=COLORES[k], label=nombre, ms=3)
        ax[1].plot(g.d, g.memoria, "o-", color=COLORES[k], ms=3)
    ax[0].set_yscale("log"); ax[0].set_xlabel("d"); ax[0].set_ylabel("tiempo [ms]"); ax[0].legend(fontsize=6)
    ax[1].set_xlabel("d"); ax[1].set_ylabel("memoria pico [MB]")
    for a in ax:
        a.set_xticks(dims)
    fig.tight_layout()
    guardar_fig(fig, "p2_escalamiento")

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(xs * prods["precio"].values[0], corte)
    ax.set_xlabel(f"precio del producto 1 (los demás fijos)"); ax.set_ylabel("ingreso R")
    guardar_fig(fig, "p2_corte")
    guardar_json(resumen, "p2_resumen")
    print(f"  {len(distintos)} óptimos locales distintos desde {len(finales)} inicios", flush=True)
    return resumen


if __name__ == "__main__":
    import sys
    correr(sintetico="--sintetico" in sys.argv, rapido="--rapido" in sys.argv)
