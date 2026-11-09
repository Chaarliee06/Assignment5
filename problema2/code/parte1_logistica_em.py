"""
parte1_logistica_em.py - Modelo de compra (regresion logistica con descenso
de gradiente) y segmentos ocultos de clientes (EM, mezcla de Gaussianas).

Datos: Online Retail II (ver datos.py).

Regresion logistica
  perdida  L(w) = -1/n sum[ y log s(x·w) + (1-y) log(1-s(x·w)) ] + (lam/2)|w|^2
  gradiente  g = X^T (s - y)/n + lam w
  Hessiana   H = X^T S X / n + lam I,  S = diag(s(1-s))  ->  H es semidefinida
             positiva, asi que L es convexa (estricta con lam > 0).

EM (mezcla de K Gaussianas) sobre el log del gasto total de cada cliente;
K se elige con BIC.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from comun import (SEMILLAS, COLORES, banda, cronometro, guardar_csv, guardar_fig,
                   guardar_json, memoria_pico, plt)
import datos


# ---------------------------------------------------------------------------
# Regresion logistica
# ---------------------------------------------------------------------------
def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def perdida_logistica(w, X, y, lam):
    s = sigmoide(X @ w)
    eps = 1e-12
    return -np.mean(y * np.log(s + eps) + (1 - y) * np.log(1 - s + eps)) + 0.5 * lam * w[1:] @ w[1:]


def gradiente_logistica(w, X, y, lam):
    g = X.T @ (sigmoide(X @ w) - y) / len(y)
    g[1:] += lam * w[1:]            # no se regulariza el sesgo
    return g


def descenso_gradiente(X, y, lam=1e-3, lr=1.0, max_iter=5000, tol=1e-5, w0=None):
    """Descenso de gradiente con paso fijo. Para con |g| < tol."""
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    historia = []
    for it in range(max_iter):
        g = gradiente_logistica(w, X, y, lam)
        historia.append(perdida_logistica(w, X, y, lam))
        if np.linalg.norm(g) < tol:
            break
        w -= lr * g
    return w, np.array(historia), it + 1


def hessiana_logistica(w, X, y, lam):
    s = sigmoide(X @ w)
    H = (X * (s * (1 - s))[:, None]).T @ X / len(y)
    H[1:, 1:] += lam * np.eye(len(w) - 1)
    return H


def con_sesgo(X):
    return np.column_stack([np.ones(len(X)), X])


# ---------------------------------------------------------------------------
# EM para mezcla de Gaussianas (1D: log del gasto)
# ---------------------------------------------------------------------------
def em_gmm(x, K, seed, max_iter=500, tol=1e-7, var_min=1e-3):
    """EM clasico para K Gaussianas en 1D. Inicializa las medias con K
    clientes al azar. Regresa pesos, medias, varianzas y log-verosimilitud."""
    rng = np.random.default_rng(seed)
    mu = rng.choice(x, K, replace=False)
    var = np.full(K, x.var())
    pi = np.full(K, 1.0 / K)
    historia = []
    for it in range(max_iter):
        # Paso E: responsabilidades r[i,k] (con log-sum-exp)
        logp = np.log(pi) - 0.5 * np.log(2 * np.pi * var) - 0.5 * (x[:, None] - mu) ** 2 / var
        m = logp.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logp - m).sum(1))
        historia.append(lse.sum())
        r = np.exp(logp - lse[:, None])
        # Paso M
        nk = r.sum(0) + 1e-12
        pi = nk / len(x)
        mu = (r * x[:, None]).sum(0) / nk
        var = np.maximum((r * (x[:, None] - mu) ** 2).sum(0) / nk, var_min)
        if it > 0 and abs(historia[-1] - historia[-2]) < tol * abs(historia[-1]):
            break
    return dict(pesos=pi, medias=mu, var=var, ll=historia[-1], historia=np.array(historia), iter=it + 1)


def bic(ll, K, n):
    return -2 * ll + (3 * K - 1) * np.log(n)


# ---------------------------------------------------------------------------
def correr(sintetico=False, rapido=False):
    print("[Parte 1] Cargando Online Retail II ...", flush=True)
    df = datos.cargar_transacciones(sintetico)
    X, y, nombres = datos.datos_compra(df)
    G = datos.gasto_clientes(df)
    resumen = dict(filas_originales=int(df.attrs["filas_originales"]), filas_limpias=len(df),
                   clientes_modelo=len(y), tasa_recompra=float(y.mean()), clientes_em=len(G),
                   variables=nombres)
    print(f"  {len(df):,} transacciones limpias, {len(y):,} clientes para el modelo de compra")

    # ---------------- Logistica: 10 particiones train/test ----------------
    Xb = con_sesgo(X)
    filas, curvas = [], []
    for s in SEMILLAS:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        tr, te = idx[:cut], idx[cut:]
        (w, hist, its), t = cronometro(descenso_gradiente, Xb[tr], y[tr])
        p = sigmoide(Xb[te] @ w)
        filas.append(dict(semilla=s, iteraciones=its, tiempo_s=t, perdida_train=hist[-1],
                          perdida_test=perdida_logistica(w, Xb[te], y[te], 0),
                          exactitud_test=np.mean((p > 0.5) == y[te]), auc_test=roc_auc_score(y[te], p)))
        curvas.append(hist)
    log = pd.DataFrame(filas)
    guardar_csv(log, "p1_logistica")
    # Evidencia numerica de convexidad: 10 inicios al azar llegan al mismo minimo
    finales = []
    for s in SEMILLAS:
        w0 = np.random.default_rng(100 + s).normal(0, 3, Xb.shape[1])
        w, hist, _ = descenso_gradiente(Xb, y, w0=w0)
        finales.append(hist[-1])
    H = hessiana_logistica(w, Xb, y, 1e-3)
    eig_min = float(np.linalg.eigvalsh(H).min())
    resumen["logistica"] = dict(
        exactitud=[log.exactitud_test.mean(), log.exactitud_test.std()],
        auc=[log.auc_test.mean(), log.auc_test.std()],
        perdida_test=[log.perdida_test.mean(), log.perdida_test.std()],
        iteraciones=[log.iteraciones.mean(), log.iteraciones.std()],
        tiempo_s=[log.tiempo_s.mean(), log.tiempo_s.std()],
        perdida_inicios_aleatorios=[float(np.min(finales)), float(np.max(finales))],
        eig_min_hessiana=eig_min, pesos=dict(zip(["sesgo"] + nombres, w.round(4))),
        tasa_base=float(max(y.mean(), 1 - y.mean())))
    print(f"  Logistica: exactitud {log.exactitud_test.mean():.3f}, AUC {log.auc_test.mean():.3f}, "
          f"perdida final con 10 inicios en [{min(finales):.6f}, {max(finales):.6f}]")

    # ---------------- EM: eleccion de K por BIC ----------------
    Ks = range(1, 6)
    filas_em = []
    mejor = {}
    for K in Ks:
        for s in SEMILLAS[: (3 if rapido else 10)]:
            r, t = cronometro(em_gmm, G, K, s)
            filas_em.append(dict(K=K, semilla=s, ll=r["ll"], bic=bic(r["ll"], K, len(G)),
                                 iteraciones=r["iter"], tiempo_s=t))
            if K not in mejor or r["ll"] > mejor[K]["ll"]:
                mejor[K] = r
    em = pd.DataFrame(filas_em)
    guardar_csv(em, "p1_em")
    bics = em.groupby("K").bic.min()
    K_opt = int(bics.idxmin())
    seg = mejor[K_opt]
    orden = np.argsort(seg["medias"])
    segmentos = dict(K=K_opt, pesos=seg["pesos"][orden], medias=seg["medias"][orden],
                     var=seg["var"][orden], gasto_medio=np.exp(seg["medias"][orden]))
    guardar_json(segmentos, "segmentos")
    llK = em[em.K == K_opt].ll
    resumen["em"] = dict(K_bic=K_opt, pesos=seg["pesos"][orden].round(3),
                         gasto_medio=np.exp(seg["medias"][orden]).round(1),
                         ll_semillas=[llK.mean(), llK.std(), llK.min(), llK.max()])
    print(f"  EM: K={K_opt} por BIC, pesos {np.round(seg['pesos'][orden], 2)}, "
          f"log-verosimilitud entre semillas [{llK.min():.1f}, {llK.max():.1f}]")

    # ---------------- Escalamiento: tiempo y memoria vs n ----------------
    tamanos = [500, 1000, 2000] if rapido else [1000, 2000, 4000, 8000]
    tamanos = sorted({min(t, len(G)) for t in tamanos} | {len(G)})
    esc = []
    sem_esc = SEMILLAS[: (3 if rapido else 10)]
    for n in tamanos:
        for s in sem_esc:
            rng = np.random.default_rng(s)
            iy = rng.choice(len(y), min(n, len(y)), replace=False)
            ig = rng.choice(len(G), n, replace=False)
            (_, _, its), t1 = cronometro(descenso_gradiente, Xb[iy], y[iy])
            r, t2 = cronometro(em_gmm, G[ig], K_opt, s)
            esc.append(dict(n=n, semilla=s, t_logistica=t1, it_logistica=its, t_em=t2, it_em=r["iter"]))
        _, m1 = memoria_pico(descenso_gradiente, Xb[iy], y[iy])
        _, m2 = memoria_pico(em_gmm, G[ig], K_opt, 0)
        for e in esc[-len(sem_esc):]:
            e.update(mem_logistica_mb=m1, mem_em_mb=m2)
    esc = pd.DataFrame(esc)
    guardar_csv(esc, "p1_escalamiento")

    # ---------------- Figuras ----------------
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    L = max(len(c) for c in curvas)
    C = np.array([np.pad(c, (0, L - len(c)), mode="edge") for c in curvas])
    banda(ax[0], np.arange(L), C, "10 particiones", COLORES[0])
    ax[0].set_xscale("log")
    ax[0].set_xlabel("iteración"); ax[0].set_ylabel("pérdida logística (train)")
    ax[0].set_title("(a) descenso de gradiente")
    g = em.groupby("K").bic.agg(["mean", "std", "min"])
    ax[1].errorbar(g.index, g["mean"], g["std"], marker="o", capsize=3, color=COLORES[1], label="media ± std")
    ax[1].plot(g.index, g["min"], "k^", ms=4, label="mejor semilla")
    ax[1].set_xlabel("K (componentes)"); ax[1].set_ylabel("BIC"); ax[1].set_title("(b) EM: elección de K")
    ax[1].legend(fontsize=7)
    ax[2].hist(G, bins=60, density=True, color="lightgray")
    xx = np.linspace(G.min(), G.max(), 300)
    total = np.zeros_like(xx)
    for j, k in enumerate(orden):
        dens = seg["pesos"][k] * np.exp(-0.5 * (xx - seg["medias"][k]) ** 2 / seg["var"][k]) \
            / np.sqrt(2 * np.pi * seg["var"][k])
        total += dens
        ax[2].plot(xx, dens, color=COLORES[2 + j], lw=1.2, label=f"seg. {j + 1} ({seg['pesos'][k]:.0%})")
    ax[2].plot(xx, total, "k--", lw=1, label="mezcla")
    ax[2].set_xlabel("log gasto total del cliente"); ax[2].set_ylabel("densidad")
    ax[2].legend(fontsize=7)
    ax[2].set_title("(c) segmentos encontrados"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    guardar_fig(fig, "p1_logistica_em")

    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    ge = esc.groupby("n").agg(["mean", "std"])
    ax[0].errorbar(ge.index, ge[("t_logistica", "mean")] * 1e3, ge[("t_logistica", "std")] * 1e3,
                   marker="o", capsize=3, label="logística (GD)")
    ax[0].errorbar(ge.index, ge[("t_em", "mean")] * 1e3, ge[("t_em", "std")] * 1e3,
                   marker="s", capsize=3, label=f"EM (K={K_opt})")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("número de clientes n"); ax[0].set_ylabel("tiempo [ms]"); ax[0].legend(fontsize=7)
    ax[1].plot(ge.index, ge[("mem_logistica_mb", "mean")], "o-", label="logística (GD)")
    ax[1].plot(ge.index, ge[("mem_em_mb", "mean")], "s-", label="EM")
    ax[1].set_xscale("log"); ax[1].set_xlabel("número de clientes n"); ax[1].set_ylabel("memoria pico [MB]")
    ax[1].legend(fontsize=7)
    for a in ax:                                   # marcas solo en los tamanos probados
        a.set_xticks(ge.index, [f"{n:,}".replace(",", " ") for n in ge.index])
        a.xaxis.set_minor_formatter(plt.NullFormatter())
    fig.tight_layout()
    guardar_fig(fig, "p1_escalamiento")
    guardar_json(resumen, "p1_resumen")
    return resumen


if __name__ == "__main__":
    import sys
    correr(sintetico="--sintetico" in sys.argv, rapido="--rapido" in sys.argv)
