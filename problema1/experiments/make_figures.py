"""Genera las figuras del reporte a partir de resultados/*.csv y *.json."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .common import FIGS, RES  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.bbox": "tight",
})
PAL = {"UCS": "#7f7f7f", "A*": "#1f77b4", "WA* (w=1.5)": "#9467bd", "IDA*": "#d62728",
       "IDA*-CR": "#ff7f0e", "A*-θ (MSE)": "#2ca02c", "A*-θ (q=0.2)": "#17becf",
       "SA": "#1f77b4", "Tabu": "#ff7f0e", "GA-OX": "#2ca02c", "ACO (MMAS)": "#d62728",
       "UCB-HH": "#9467bd", "GA memético": "#8c564b", "ACO+LS": "#e377c2"}


def _read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def band(ax, groups, label=None, color=None, marker="o", scale=1.0):
    """Mediana con rango intercuartil (adecuado para ejes logarítmicos)."""
    med = groups.median() * scale
    q1 = groups.quantile(0.25) * scale
    q3 = groups.quantile(0.75) * scale
    ax.errorbar(med.index, med.values, yerr=[med.values - q1.values, q3.values - med.values],
                marker=marker, ms=3, capsize=2, label=label, color=color)


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print("figura", name)


# --------------------------------------------------------------------- búsqueda

def fig_busqueda():
    d = _read("busqueda_rejillas.csv")
    if d is None:
        return
    order = ["UCS", "A*", "WA* (w=1.5)", "A*-θ (MSE)", "A*-θ (q=0.2)", "IDA*-CR", "IDA*"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1), layout="constrained")
    for m in order:
        s = d[(d.metodo == m)]
        if s.empty:
            continue
        ok = s[s.estado == "ok"]
        g = ok.groupby("N")
        band(ax[0], g.exp, m, PAL.get(m))
        band(ax[1], g.t_s, m, PAL.get(m), scale=1e3)
        solved = s.groupby("N").estado.apply(lambda x: 100 * (x == "ok").mean())
        ax[2].plot(solved.index, solved.values, marker="o", ms=3, label=m, color=PAL.get(m))
    ax[0].set(yscale="log", xlabel="N (celdas por lado)", ylabel="nodos expandidos",
              title="(a) nodos expandidos (resueltas)")
    ax[2].text(0.03, 0.55, "UCS, A*, WA* y A*-θ:\n100 % en todos los tamaños", transform=ax[2].transAxes,
               fontsize=7.5, color="#444")
    ax[1].set(yscale="log", xlabel="N", ylabel="tiempo [ms]", title="(b) tiempo por consulta")
    ax[2].set(xlabel="N", ylabel="% consultas resueltas", ylim=(-5, 105),
              title="(c) dentro del límite (15 s)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=7, fontsize=7.5, frameon=False)
    save(fig, "fig_busqueda_escala.png")

    s = _read("busqueda_ida_pequenas.csv")
    if s is not None:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
        for m, c in (("A*", "#1f77b4"), ("IDA* sin TT", "#d62728"), ("IDA*-CR sin TT", "#ff7f0e"),
                     ("IDA* con TT", "#9467bd"), ("IDA*-CR con TT", "#2ca02c")):
            q = s[s.metodo == m]
            g = q.groupby("N")
            ax[0].plot(g.exp.median().index, g.exp.median(), marker="o", ms=3, label=m, color=c)
            ax[1].plot(g.estado.apply(lambda x: 100 * (x == "ok").mean()), marker="o", ms=3, label=m, color=c)
        ax[0].set(yscale="log", xlabel="N", ylabel="nodos expandidos (mediana)",
                  title="(a) re-expansiones de IDA*")
        for a in ax:
            a.set_xticks(sorted(s.N.unique()))
        ax[1].set(xlabel="N", ylabel="% resueltas (≤ 3·10⁶ nodos, 10 s)", ylim=(-5, 105),
                  title="(b) consultas resueltas")
        ax[0].legend(fontsize=7)
        save(fig, "fig_ida_pequenas.png")


def fig_heuristica():
    d = _read("busqueda_heuristica_prueba.csv")
    r = _read("busqueda_rejillas.csv")
    if d is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
    for m, c in (("A*-θ (MSE)", "#2ca02c"), ("A*-θ (q=0.2)", "#17becf")):
        s = d[d.metodo == m]
        ratio = s.costo / s.c_opt
        ax[0].scatter(np.maximum(1, s.w_P), ratio, s=10, alpha=0.7, label=f"{m} (dataset)", color=c)
        if r is not None:
            t = r[r.metodo == m]
            ax[0].scatter(np.maximum(1, t.w_P), t.costo / t.c_opt, s=10, alpha=0.7, marker="^",
                          label=f"{m} (ciudades)", color=c, edgecolor="k", linewidth=0.3)
    ax[0].set_xscale("log")
    lim = ax[0].get_xlim()
    xs = np.geomspace(1, max(lim[1], 1.05), 50)
    ax[0].plot(xs, xs, "k--", lw=1, label=r"cota: costo $= w_P\,C^*$")
    ymax = max(1.05, float(np.nanmax(d[d.metodo.str.startswith("A*-θ")].costo / d[d.metodo.str.startswith("A*-θ")].c_opt)))
    if r is not None:
        rr = r[r.metodo.str.startswith("A*-θ")]
        ymax = max(ymax, float(np.nanmax(rr.costo / rr.c_opt)))
    ax[0].set_ylim(0.98, ymax * 1.08)
    ax[0].set(xlabel=r"$w_P=\max_{n\in P^*}\, h_\theta(n)/h^*(n)$", ylabel=r"costo $/\ C^*$",
              title=r"(a) verificación de la cota $w_P\,C^*$")
    ax[0].legend(fontsize=6.5)
    meth = ["A*", "WA* (w=1.5)", "A*-θ (MSE)", "A*-θ (q=0.2)"]
    exp = [d[d.metodo == m].exp.mean() for m in meth]
    ax[1].bar(range(len(meth)), exp, color=[PAL[m] for m in meth])
    ax[1].set_xticks(range(len(meth)), [m.replace(" ", "\n", 1) for m in meth], fontsize=7)
    ax[1].set(ylabel="nodos expandidos (media)", title="(b) mapas de prueba del dataset")
    gaps = [d[d.metodo == m].brecha_pct for m in meth]
    ax[2].boxplot(gaps, tick_labels=[m.replace(" ", "\n", 1) for m in meth], showfliers=True)
    ax[2].set(ylabel="brecha de costo [%]", title="(c) sub-optimalidad")
    ax[2].tick_params(axis="x", labelsize=7)
    save(fig, "fig_heuristica.png")


# ------------------------------------------------------------------------ mochila

def fig_mochila():
    bt = _read("mochila_backtracking.csv")
    bb = _read("mochila_bnb.csv")
    mm = _read("mochila_mitm.csv")
    if bt is None or bb is None or mm is None:
        return
    fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.0), layout="constrained")
    for m, c in (("BT sin cota", "#d62728"), ("BT con cota", "#1f77b4")):
        band(ax[0], bt[bt.metodo == m].groupby("n").nodos, m, c)
    ns = np.array(sorted(bt.n.unique()))
    ax[0].plot(ns, 2.0 ** (ns + 1), "k:", lw=1, label=r"$2^{n+1}$")
    ax[0].set(yscale="log", xlabel="n (kits)", ylabel="nodos", title="(a) backtracking con bitmask")
    ax[0].legend(fontsize=7)
    for m, c in (("B&B DFS", "#1f77b4"), ("B&B mejor-primero", "#ff7f0e")):
        g = bb[bb.metodo == m].groupby("n")
        band(ax[1], g.nodos, m, c)
        band(ax[2], g.t_s, m, c, scale=1e3)
    band(ax[2], bb[bb.metodo == "HiGHS (MILP)"].groupby("n").t_s, "HiGHS (MILP)", "#7f7f7f", "s", 1e3)
    ax[1].set(yscale="log", xlabel="n", ylabel="nodos", title="(b) B&B con cota fraccional")
    ax[2].set(yscale="log", xlabel="n", ylabel="tiempo [ms]", title="(c) tiempo B&B")
    ax[1].legend(fontsize=7)
    ax[2].legend(fontsize=7)
    g = mm.groupby("n")
    band(ax[3], g.t_mitm_s, "MITM", "#2ca02c", "o", 1e3)
    band(ax[3], g.t_bnb_s, "B&B (lím. 5 s)", "#1f77b4", "s", 1e3)
    if "t_milp_s" in mm:
        band(ax[3], g.t_milp_s, "HiGHS (lím. 20 s)", "#7f7f7f", "^", 1e3)
    nn = np.array(sorted(mm.n.unique()))
    ref = 1e3 * g.t_mitm_s.median().iloc[-1] * 2.0 ** ((nn - nn[-1]) / 2)
    ax[3].plot(nn, ref, "k:", lw=1, label=r"$\propto 2^{n/2}$")
    ax[3].set(yscale="log", xlabel="n", ylabel="tiempo [ms]", title="(d) carga más cercana a W")
    ax[3].legend(fontsize=7)
    save(fig, "fig_mochila.png")


# ---------------------------------------------------------------------------- CSP

def fig_csp():
    d = _read("csp_transicion.csv")
    if d is None:
        return
    meth = ["BT", "BT+MRV", "FC", "FC+MRV", "FC+MRV+LCV", "AC-3+FC+MRV+LCV", "MAC+MRV+LCV", "Min-conflicts"]
    cols = ["#7f7f7f", "#bcbd22", "#d62728", "#ff7f0e", "#1f77b4", "#17becf", "#2ca02c", "#9467bd"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1), layout="constrained")
    for m, c in zip(meth, cols):
        s = d[d.metodo == m]
        if s.empty:
            continue
        g = s.groupby("D")
        done = g.estado.apply(lambda x: 100 * x.isin(["sat", "unsat"]).mean() if m != "Min-conflicts"
                              else 100 * (x == "sat").mean())
        ax[0].plot(done.index, done.values, marker="o", ms=3, label=m, color=c)
        ax[1].plot(g.nodos.median().index, g.nodos.median(), marker="o", ms=3, label=m, color=c)
        ax[2].plot(g.t_s.mean().index, 1e3 * g.t_s.mean(), marker="o", ms=3, label=m, color=c)
    # fracción satisfacible según cualquier método completo
    comp = d[d.metodo != "Min-conflicts"]
    sat = comp.groupby(["D", "inst"]).estado.apply(lambda x: "sat" if (x == "sat").any() else
                                                    ("unsat" if (x == "unsat").any() else "?"))
    frac = sat.groupby(level=0).apply(lambda x: 100 * (x == "sat").mean())
    ax[0].plot(frac.index, frac.values, "k--", lw=1, label="% satisfacibles")
    ax[0].set(xlabel="D (entregas)", ylabel="% instancias terminadas", ylim=(-5, 105),
              title="(a) terminadas (≤ 10⁵ nodos, 5 s)")
    ax[1].set(yscale="log", xlabel="D", ylabel="nodos (mediana)", title="(b) nodos")
    ax[2].set(yscale="log", xlabel="D", ylabel="tiempo medio [ms]", title="(c) tiempo")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, fontsize=7.5, frameon=False)
    save(fig, "fig_csp.png")


# --------------------------------------------------------------------------- ATSP

def fig_atsp():
    meta = _read("atsp_meta.csv")
    ex = _read("atsp_exacto.csv")
    if ex is not None:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
        for m, c in (("Fuerza bruta", "#7f7f7f"), ("Held–Karp (DP)", "#9467bd"), ("B&B (MST)", "#1f77b4"),
                     ("B&B (MST) + VMC", "#ff7f0e"), ("B&B (MST) + ACO+LS", "#2ca02c")):
            g = ex[ex.metodo == m].groupby("K")
            if g.ngroups == 0:
                continue
            band(ax[0], g.nodos, m, c)
            band(ax[1], g.t_s, m, c, scale=1e3)
        ax[0].set(yscale="log", xlabel="K (clínicas)", ylabel="nodos / estados", title="(a) tamaño de la búsqueda")
        ax[1].set(yscale="log", xlabel="K", ylabel="tiempo [ms]", title="(b) tiempo (incluye ACO+LS)")
        ax[0].legend(fontsize=7)
        save(fig, "fig_atsp_exacto.png")
    if meta is None:
        return
    curves_p = RES / "atsp_curvas.json"
    curves = json.load(open(curves_p)) if curves_p.exists() else {}
    inst = _read("atsp_instancias.csv")
    opt = {(r.N, r.K): r.opt for r in inst.itertuples()} if inst is not None else {}
    meths = ["SA", "Tabu", "GA-OX", "ACO (MMAS)", "UCB-HH", "GA memético", "ACO+LS"]
    Ks = sorted(meta.K.unique())
    fig, ax = plt.subplots(1, len(Ks), figsize=(3.0 * len(Ks), 3.0), layout="constrained")
    ax = np.atleast_1d(ax)
    N0 = 100 if 100 in meta.N.unique() else meta.N.unique()[0]
    for a, K in zip(ax, Ks):
        for m in meths:
            key = f"{N0}|{K}|{m}"
            if key not in curves:
                continue
            runs = list(curves[key].values())
            L = min(len(r) for r in runs)
            ev = np.array([p[0] for p in runs[0][:L]])
            val = np.array([[p[1] for p in r[:L]] for r in runs])
            gap = 100 * (val / opt.get((N0, K), val.min()) - 1)
            mu = gap.mean(0)
            q1, q3 = np.percentile(gap, 25, axis=0), np.percentile(gap, 75, axis=0)
            a.plot(ev, np.maximum(mu, 1e-3), label=m, color=PAL[m], lw=1.2)
            a.fill_between(ev, np.maximum(q1, 1e-3), np.maximum(q3, 1e-3), color=PAL[m], alpha=0.12)
        a.set(yscale="log", xlabel="evaluaciones", title=f"K = {K} (N = {N0})")
        a.set_ylim(bottom=0.05)
    ax[0].set_ylabel("brecha vs óptimo [%]")
    ax[-1].legend(fontsize=6.5)
    save(fig, "fig_atsp_convergencia.png")

    g = meta.groupby(["K", "metodo"]).brecha_pct.mean().unstack("metodo")[meths]
    t = meta.groupby(["K", "metodo"]).t_s.mean().unstack("metodo")[meths]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.1), layout="constrained")
    x = np.arange(len(g.index))
    w = 0.11
    for k, m in enumerate(meths):
        ax[0].bar(x + (k - 3) * w, g[m].values, w, label=m, color=PAL[m])
        ax[1].bar(x + (k - 3) * w, t[m].values, w, label=m, color=PAL[m])
    for a in ax:
        a.set_xticks(x, [f"K={k}" for k in g.index])
    ax[0].set(ylabel="brecha media [%] (4 rejillas × 10 semillas)", yscale="symlog",
              title="(a) calidad con presupuesto igual (3·10⁴ evaluaciones)")
    ax[0].set_ylim(0, float(np.nanmax(g.values)) * 2.5)
    ax[1].set(ylabel="tiempo medio [s]", title="(b) tiempo por corrida")
    ax[0].legend(fontsize=6.5, ncol=2)
    save(fig, "fig_atsp_brechas.png")

    key = [k for k in curves if k.startswith(f"UCB|{N0}|")]
    if key:
        Ksel = [k for k in key if k.endswith("|50")] or key
        data = curves[Ksel[0]]
        hist = np.array([np.array(v["hist"]) for v in data.values()])
        L = min(len(h) for h in hist)
        hist = np.array([h[:L] for h in hist])
        ev = hist[0][:, 0]
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        for j, (nm, c) in enumerate((("swap", "#1f77b4"), ("insert", "#ff7f0e"), ("2-opt", "#2ca02c"))):
            mu = hist[:, :, j + 1].mean(0)
            ax.plot(ev, mu, label=nm, color=c)
        ax.set(xlabel="evaluaciones", ylabel="fracción acumulada de uso",
               title=f"UCB: operador elegido (N={Ksel[0].split('|')[1]}, K={Ksel[0].split('|')[2]})", ylim=(0, 1))
        ax.legend()
        save(fig, "fig_ucb.png")


# ----------------------------------------------------------------------- misión

def fig_mision():
    p = RES / "mision.json"
    if not p.exists():
        return
    m = json.load(open(p))
    N = m["N"]
    blocked = np.array(m["blocked"]).reshape(N, N)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.9))
    cmap = ListedColormap(["#f7f7f7", "#4d4d4d"])
    for a in ax:
        a.imshow(blocked, cmap=cmap, origin="upper", interpolation="nearest")
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    wind = np.array(m["wind"])
    yy, xx = np.mgrid[0:wind.shape[0], 0:wind.shape[1]] * 4
    ax[0].quiver(xx, yy, wind[..., 1], -wind[..., 0], color="#1f77b4", alpha=0.45, scale=30, width=0.0025)
    route = m["routes"]["Óptimo (HiGHS)"]
    for leg in route:
        rc = np.array([divmod(u, N) for u in leg])
        ax[0].plot(rc[:, 1], rc[:, 0], color="#d62728", lw=1.4)
    for k, h in enumerate(m["hospitals"]):
        r, c = divmod(h["node"], N)
        if k == 0:
            ax[0].plot(c, r, marker="*", ms=14, color="#ffbf00", mec="k", zorder=5)
        else:
            ax[0].plot(c, r, "o", ms=4.5, color="#2ca02c", mec="k", mew=0.5, zorder=5)
    ax[0].set_title(f"(a) ruta óptima, costo {m['tour_costs']['Óptimo (HiGHS)']:.1f} "
                    f"(depósito ★, {m['K']} clínicas, viento →)")
    colors = {"UCS": "#9e9e9e", "A*": "#1f77b4", "A*-θ (MSE)": "#2ca02c"}
    for name in ("UCS", "A*", "A*-θ (MSE)"):
        s = m["searches"][name]
        rc = np.array([divmod(u, N) for u in s["closed"]])
        if len(rc):
            ax[1].scatter(rc[:, 1], rc[:, 0], s=2.2, color=colors[name], alpha=0.55,
                          label=f"{name}: {s['expanded']} expandidos, costo {s['cost']:.2f}")
    p = np.array([divmod(u, N) for u in m["searches"]["A*"]["path"]])
    ax[1].plot(p[:, 1], p[:, 0], color="#d62728", lw=1.6, label="camino óptimo")
    ax[1].legend(fontsize=7, loc="lower left", markerscale=3)
    ax[1].set_title("(b) nodos expandidos: depósito → clínica más lejana")
    save(fig, "fig_mision.png")


if __name__ == "__main__":
    fig_busqueda()
    fig_heuristica()
    fig_mochila()
    fig_csp()
    fig_atsp()
    fig_mision()
