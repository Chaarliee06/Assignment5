"""Genera las tablas LaTeX (reporte/tablas/*.tex) y reporte/numeros.tex desde resultados/."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import RES, ROOT

TAB = ROOT / "reporte" / "tablas"
TAB.mkdir(parents=True, exist_ok=True)
NUM = {}


def num(name, value):
    NUM[name] = value


def f(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if abs(x) < 0.5 * 10 ** (-d):        # evita "-0.00"
        x = 0.0
    return f"{x:,.{d}f}".replace(",", "\\,")


def pm(mu, sd, d=2):
    if not np.isfinite(mu):
        return "--"
    if not np.isfinite(sd):
        return f(mu, d)
    return f"{f(mu, d)} $\\pm$ {f(sd, d)}"


def esc(s):
    return s.replace("&", "\\&").replace("θ", "$\\theta$").replace("_", "\\_")


def write(name, body):
    (TAB / name).write_text(body, encoding="utf-8")
    print("tabla", name)


# ------------------------------------------------------------------------- búsqueda

def t_busqueda():
    v = pd.read_csv(RES / "busqueda_validacion_dataset.csv")
    num("valMapas", f(len(v), 0))
    num("valIguales", f(int(v.igual.sum()), 0))
    num("valObst", f(100 * v.obstaculos.mean(), 1))
    num("valExpUCS", f(v.exp_ucs.mean(), 0))
    num("valExpA", f(v.exp_astar.mean(), 0))
    num("valCaminosValidos", f(int(v.camino_valido.sum()), 0))
    num("valMedRatio", f(100 * (v.gt_sobre_opt.median() - 1), 1))
    num("valMasCortos", int((v.gt_sobre_opt < 1 - 1e-9).sum()))
    if "igual_holgura" in v:
        rh = v.gt_sobre_opt_holgura
        num("valIgualesHolg", f(int(v.igual_holgura.sum()), 0))
        num("valPctHolg", f(100 * v.igual_holgura.mean(), 1))
        num("valMayorHolg", f(int((rh > 1 + 1e-9).sum()), 0))
        num("valMenorHolg", int((rh < 1 - 1e-9).sum()))
        num("valMedExcHolg", f(100 * (rh[rh > 1 + 1e-9].median() - 1), 2))
        num("valMaxExcHolg", f(100 * (rh.max() - 1), 1))
    from drones.datasets import list_map_files
    from .common import MAPAS_KAGGLE
    from .extraer_mapas_kaggle import LISTA, leer_lista
    files = [x.replace("\\", "/") for x in list_map_files(str(MAPAS_KAGGLE))]
    if not files and LISTA.exists():   # sin los .pt (p. ej. en el repo de GitHub) se usa la lista
        files = leer_lista()
    num("kagTrain", f(sum("/train/" in x for x in files), 0))
    num("kagVal", f(sum("/validation/" in x for x in files), 0))
    num("kagTestN", f(sum("/test/" in x for x in files), 0))
    d = pd.read_csv(RES / "busqueda_rejillas.csv")
    order = ["UCS", "A*", "IDA*-CR", "IDA*", "WA* (w=1.5)", "A*-θ (MSE)", "A*-θ (q=0.2)"]
    lines = []
    for N in sorted(d.N.unique()):
        first = True
        for m in order:
            s = d[(d.N == N) & (d.metodo == m)]
            if s.empty:
                continue
            ok = s[s.estado == "ok"]
            mem = s.mem_kib.dropna() if "mem_kib" in s else pd.Series(dtype=float)
            row = [f"\\multirow{{7}}{{*}}{{{N}}}" if first else "", esc(m),
                   f"{len(ok)}/{len(s)}", pm(ok.exp.mean(), ok.exp.std(), 0), f(ok.ebf.mean(), 3),
                   pm(1e3 * ok.t_s.mean(), 1e3 * ok.t_s.std(), 1),
                   f(mem.mean(), 0) if len(mem) else "--", f(ok.brecha_pct.mean(), 2)]
            lines.append(" & ".join(row) + " \\\\")
            first = False
        lines.append("\\midrule")
    body = ("\\begin{tabular}{llrrrrrr}\n\\toprule\n$N$ & método & resueltas & nodos expandidos & $b^*$ & "
            "tiempo [ms] & mem. [KiB] & brecha [\\%] \\\\\n\\midrule\n" + "\n".join(lines[:-1]) +
            "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_busqueda.tex", body)
    for N in sorted(d.N.unique()):
        for m, key in (("A*", "Astar"), ("UCS", "Ucs"), ("IDA*-CR", "IdaCr"), ("IDA*", "Ida")):
            q = d[(d.N == N) & (d.metodo == m)].mem_kib.dropna() if "mem_kib" in d else pd.Series(dtype=float)
            num(f"mem{key}N{int(N)}", f(q.mean(), 0) if len(q) else "--")
    a = d[d.metodo == "A*"]
    u = d[d.metodo == "UCS"]
    num("redAvsUCS", f(100 * (1 - a.exp.sum() / u.exp.sum()), 1))
    cr = d[(d.metodo == "IDA*-CR")]
    ida = d[(d.metodo == "IDA*")]
    num("idaCrResueltas", f"{int((cr.estado == 'ok').sum())}/{len(cr)}")
    num("idaResueltas", f"{int((ida.estado == 'ok').sum())}/{len(ida)}")
    for N in sorted(d.N.unique()):
        key = {50: "A", 100: "B", 150: "C", 200: "D"}[int(N)]
        num(f"idaCr{key}", f"{int((cr[cr.N == N].estado == 'ok').sum())}/{len(cr[cr.N == N])}")
        num(f"mmin{key}", f(d[d.N == N].m_min.iloc[0], 3))
    s = pd.read_csv(RES / "busqueda_ida_pequenas.csv")
    if "mem_kib" in s:
        for N in sorted(s.N.unique()):
            for m, key in (("IDA* sin TT", "SinTT"), ("IDA*-CR sin TT", "CrSinTT"), ("IDA*-CR con TT", "CrTT"),
                           ("A*", "Astar"), ("IDA* con TT", "TT")):
                q = s[(s.N == N) & (s.metodo == m)].mem_kib.dropna()
                num(f"idaMem{key}N{int(N)}", f(q.median(), 1) if len(q) else "--")
    for N in sorted(s.N.unique()):
        for m, key in (("IDA* sin TT", "SinTT"), ("IDA*-CR con TT", "CrTT"), ("A*", "Astar"), ("IDA* con TT", "TT")):
            q = s[(s.N == N) & (s.metodo == m)]
            num(f"ida{key}N{int(N)}", f(q.exp.median(), 0))
            num(f"ida{key}OkN{int(N)}", f"{int((q.estado == 'ok').sum())}/{len(q)}")


def t_heuristica():
    d = pd.read_csv(RES / "busqueda_heuristica_prueba.csv")
    r = pd.read_csv(RES / "busqueda_rejillas.csv")
    info = json.load(open(RES / "heuristica_entrenamiento.json"))
    num("hMapas", info["mapas"])
    num("hFilas", f(info["filas"], 0))
    num("hMaeMse", f(info["mse"]["mae_ratio"], 3))
    num("hMaeQ", f(info["q02"]["mae_ratio"], 3))
    num("hSobreMse", f(100 * info["mse"]["frac_sobreestima"], 1))
    num("hSobreQ", f(100 * info["q02"]["frac_sobreestima"], 1))
    num("hTest", d.mapa.nunique())
    lines = []
    for label, src in (("mapas de prueba del dataset (100$\\times$100)", d), ("ciudades $N\\in\\{50,\\dots,200\\}$", r)):
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for m in ["UCS", "A*", "WA* (w=1.5)", "A*-θ (MSE)", "A*-θ (q=0.2)"]:
            s = src[src.metodo == m]
            if s.empty:
                continue
            opt = 100 * (s.brecha_pct < 1e-6).mean()
            wp = s.w_P if "w_P" in s and m.startswith("A*-θ") else None
            row = [esc(m), f(s.brecha_pct.mean(), 2), f(s.brecha_pct.max(), 2), f(opt, 0),
                   f(s.exp.mean(), 0), f(1e3 * s.t_s.mean(), 1),
                   f(wp.mean(), 3) if wp is not None else ("1.5" if "WA" in m else "--"),
                   f(wp.max(), 3) if wp is not None else ("1.5" if "WA" in m else "--"),
                   f(100 * s.cota_ok.mean(), 0) if "cota_ok" in s and s.cota_ok.notna().any() else "--",
                   f(s.mem_kib.dropna().mean(), 0) if "mem_kib" in s and s.mem_kib.notna().any() else "--"]
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    body = ("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\nmétodo & brecha [\\%] & máx. [\\%] & óptimas [\\%] & "
            "expandidos & tiempo [ms] & $\\bar w_P$ & $\\max w_P$ & cota ok [\\%] & mem. [KiB] \\\\\n\\midrule\n" +
            "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_heuristica.tex", body)
    for m, key in (("A*-θ (MSE)", "Mse"), ("A*-θ (q=0.2)", "Q"), ("A*", "A"), ("WA* (w=1.5)", "W")):
        s = d[d.metodo == m]
        num(f"hGap{key}", f(s.brecha_pct.mean(), 2))
        num(f"hGapMax{key}", f(s.brecha_pct.max(), 2))
        num(f"hExp{key}", f(s.exp.mean(), 0))
        num(f"hT{key}", f(1e3 * s.t_s.mean(), 1))
        if "t_inf_s" in s:
            num(f"hInf{key}", f(1e3 * s.t_inf_s.mean(), 1))
        if m.startswith("A*-θ"):
            num(f"hWmax{key}", f(s.w_P.max(), 3))
            num(f"hWall{key}", f(s.w_all.max(), 3))
            num(f"hCota{key}", f(100 * s.cota_ok.mean(), 0))
            num(f"hFrac{key}", f(100 * s.frac_sobre.mean(), 1))
            rr = r[r.metodo == m]
            num(f"rCota{key}", f(100 * rr.cota_ok.mean(), 0))
            num(f"rGap{key}", f(rr.brecha_pct.mean(), 2))
            num(f"rWmax{key}", f(rr.w_P.max(), 3))
    num("hExpUCS", f(d[d.metodo == "UCS"].exp.mean(), 0))
    num("hWpA", f(d[d.metodo == "A*"].w_P.mean(), 2))
    for N in sorted(r.N.unique()):
        for m, key in (("A*", "Astar"), ("UCS", "Ucs"), ("A*-θ (MSE)", "Mse"), ("A*-θ (q=0.2)", "Q"),
                       ("WA* (w=1.5)", "W")):
            s = r[(r.N == N) & (r.metodo == m)]
            num(f"rExp{key}N{int(N)}", f(s.exp.mean(), 0))
            num(f"rT{key}N{int(N)}", f(1e3 * s.t_s.mean(), 1))
        lr = r[(r.N == N) & (r.metodo == "A*-θ (MSE)")]
        num(f"rInfN{int(N)}", f(1e3 * lr.t_inf_s.mean(), 1))
        num(f"rReabN{int(N)}", f(lr.reabiertos.mean(), 0))
        num(f"rMedMseN{int(N)}", f(lr.exp.median(), 0))
        num(f"rMedAstarN{int(N)}", f(r[(r.N == N) & (r.metodo == "A*")].exp.median(), 0))


# --------------------------------------------------------------------------- mochila

def t_mochila():
    k = pd.read_csv(RES / "mochila_kaggle.csv")
    num("kagInst", f(int(k.instancias.iloc[0]), 0))
    num("kagDisc", int(k[k.metodo != "best_picks_consistentes"].discrepancias.sum()))
    for m, key in (("bt", "Bt"), ("bnb", "Bnb"), ("mitm", "Mitm"), ("dp", "Dp"), ("bnb_best", "Best")):
        num(f"kagT{key}", f(float(k[k.metodo == m].tiempo_medio_us.iloc[0]), 0))
        if "mem_kib" in k:
            num(f"kagMem{key}", f(float(k[k.metodo == m].mem_kib.iloc[0]), 1))
    bt = pd.read_csv(RES / "mochila_backtracking.csv")
    for n in sorted(bt.n.unique()):
        for m, key in (("BT con cota", "Con"), ("BT sin cota", "Sin")):
            s = bt[(bt.n == n) & (bt.metodo == m)]
            if len(s):
                num(f"bt{key}N{int(n)}", f(s.nodos.mean(), 0))
    num("btTodasOk", int(bt.ok.all()))
    bb = pd.read_csv(RES / "mochila_bnb.csv")
    lines = []
    for n in sorted(bb.n.unique()):
        first = True
        for m in ["BT con cota", "B&B DFS", "B&B mejor-primero", "HiGHS (MILP)", "GA (base bioinspirada)"]:
            s = bb[(bb.n == n) & (bb.metodo == m)]
            if s.empty:
                continue
            mem = s.mem_kib.dropna() if "mem_kib" in s else pd.Series(dtype=float)
            nod = s.nodos.dropna() if "nodos" in s else pd.Series(dtype=float)
            cert = (f"{int((s.estado == 'ok').sum())}/{len(s)}" if "estado" in s and s.estado.notna().any()
                    else "--")
            row = [f"\\multirow{{{5 if n <= 30 else 4}}}{{*}}{{{n}}}" if first else "", esc(m),
                   pm(nod.mean(), nod.std(), 0) if len(nod) else "--",
                   pm(1e3 * s.t_s.mean(), 1e3 * s.t_s.std(), 2), f(mem.mean(), 0) if len(mem) else "--",
                   pm(s.brecha_pct.mean(), s.brecha_pct.std(), 3), f(100 * s.ok.mean(), 0), cert]
            lines.append(" & ".join(row) + " \\\\")
            first = False
        lines.append("\\midrule")
    body = ("\\begin{tabular}{llrrrrrr}\n\\toprule\n$n$ & método & nodos & tiempo [ms] & mem. [KiB] & "
            "brecha [\\%] & óptimo [\\%] & termina \\\\\n\\midrule\n" + "\n".join(lines[:-1]) +
            "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_mochila_bnb.tex", body)
    lim = bb[bb.metodo.str.startswith("B&B") & (bb.estado == "limit")]
    num("bnbLimites", len(lim))
    hard = bb[(bb.n == 120) & bb.metodo.str.startswith("B&B")]
    num("bnbDfsMaxNodos", f(hard[hard.metodo == "B&B DFS"].nodos.max(), 0))
    num("bnbBestMemMax", f(hard[hard.metodo == "B&B mejor-primero"].mem_kib.max() / 1024, 1))
    num("bnbDfsMem", f(hard[hard.metodo == "B&B DFS"].mem_kib.max(), 0))
    hi = bb[(bb.n == 120) & (bb.metodo == "HiGHS (MILP)")]
    num("milpKnapT", f(1e3 * hi.t_s.max(), 0))
    hard_inst = hard[hard.estado == "limit"].inst
    if len(hard_inst):
        hi_ = int(hard_inst.iloc[0])
        h2 = hard[hard.inst == hi_]
        dfs_r = h2[h2.metodo == "B&B DFS"].iloc[0]
        best_r = h2[h2.metodo == "B&B mejor-primero"].iloc[0]
        num("bnbDfsHardVal", int(dfs_r.valor))
        num("bnbBestHardVal", int(best_r.valor))
        num("bnbHardOpt", int(dfs_r.opt))
        num("bnbBestHardGap", f(best_r.brecha_pct, 2))
    num("bnbOk", f(100 * bb[bb.metodo.str.startswith("B&B")].ok.mean(), 0))
    ga = bb[bb.metodo.str.startswith("GA")]
    num("gaGap", f(ga.brecha_pct.mean(), 2))
    num("gaOpt", f(100 * ga.ok.mean(), 0))
    mm = pd.read_csv(RES / "mochila_mitm.csv")
    lines = []
    for n in sorted(mm.n.unique()):
        s = mm[mm.n == n]
        lim = int((s.estado_bnb == "limit").sum())
        mm_mem = s.mem_mitm_kib.dropna() if "mem_mitm_kib" in s else pd.Series(dtype=float)
        bb_mem = s.mem_bnb_kib.dropna() if "mem_bnb_kib" in s else pd.Series(dtype=float)
        lines.append(" & ".join([str(n), pm(1e3 * s.t_mitm_s.mean(), 1e3 * s.t_mitm_s.std(), 1),
                                 f(s.subconjuntos.mean(), 0), pm(s.frontera.mean(), s.frontera.std(), 1),
                                 f(s.holgura.mean(), 1), f(100 * s.ok.mean(), 0),
                                 f(mm_mem.mean(), 0) if len(mm_mem) else "--",
                                 pm(1e3 * s.t_bnb_s.mean(), 1e3 * s.t_bnb_s.std(), 0),
                                 f"{lim}/{len(s)}", f(bb_mem.mean(), 0) if len(bb_mem) else "--"]) + " \\\\")
    body = ("\\begin{tabular}{rrrrrrrrrr}\n\\toprule\n$n$ & MITM [ms] & subconjuntos & $|\\partial B|$ & "
            "$W-$carga & consistente [\\%] & mem. MITM [KiB] & B\\&B [ms] & B\\&B al límite & mem. B\\&B [KiB] "
            "\\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_mochila_mitm.tex", body)
    s40 = mm[mm.n == mm.n.max()]
    num("mitmT", f(1e3 * s40.t_mitm_s.mean(), 0))
    num("mitmHolg", f(s40.holgura.mean(), 1))
    num("mitmOk", f(100 * mm.ok.mean(), 0))
    num("mitmLim", f"{int((s40.estado_bnb == 'limit').sum())}/{len(s40)}")
    num("mitmVerif", f"{int((mm.verificado_por >= 1).sum())}/{len(mm)}")
    num("mitmMilpCert", f"{int(mm.milp_certifica.sum())}/{len(mm)}")
    num("mitmMilpT", f(mm.t_milp_s.mean(), 1))
    if "mem_mitm_kib" in mm:
        num("mitmMem", f(s40.mem_mitm_kib.dropna().mean() / 1024, 1))
    if "mem_bnb_kib" in mm:
        num("mitmBnbMem", f(s40.mem_bnb_kib.dropna().mean(), 0))


# ------------------------------------------------------------------------------- CSP

def t_csp():
    d = pd.read_csv(RES / "csp_transicion.csv")
    meth = ["BT", "BT+MRV", "FC", "FC+MRV", "FC+MRV+LCV", "AC-3+FC+MRV+LCV", "MAC+MRV+LCV", "Min-conflicts"]
    comp = d[d.metodo != "Min-conflicts"]
    lines = []
    for D in sorted(d.D.unique()):
        sat = comp[comp.D == D].groupby("inst").estado.apply(
            lambda x: "sat" if (x == "sat").any() else ("unsat" if (x == "unsat").any() else "?"))
        row = [str(D), f"{(sat == 'sat').sum()}/{(sat == 'unsat').sum()}/{(sat == '?').sum()}"]
        for m in meth:
            s = d[(d.D == D) & (d.metodo == m)]
            if m == "Min-conflicts":
                row.append(f(100 * (s.estado == "sat").mean(), 0))
            else:
                done = 100 * s.estado.isin(["sat", "unsat"]).mean()
                row.append(f"{f(done, 0)} ({f(s.nodos.median(), 0)})")
        lines.append(" & ".join(row) + " \\\\")
    def two(m):
        parts = m.split("+")
        if len(parts) <= 2 and len(m) <= 8:
            return m
        k = (len(parts) + 1) // 2
        return "\\makecell{" + "+".join(parts[:k]) + "\\\\+" + "+".join(parts[k:]) + "}"
    head = " & ".join(["$D$", "\\makecell{sat/unsat\\\\/?}"] +
                      [two(m) if m != "Min-conflicts" else "\\makecell{Min-\\\\conflicts}" for m in meth])
    body = ("\\begin{tabular}{rr" + "r" * len(meth) + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n" +
            "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_csp.tex", body)
    e = pd.read_csv(RES / "csp_escala.csv")
    lines = []
    for D in sorted(e.D.unique()):
        row = [str(D)]
        for m in ["BT+MRV", "FC+MRV", "FC+MRV+LCV", "MAC+MRV+LCV", "Min-conflicts"]:
            s = e[(e.D == D) & (e.metodo == m)]
            ok = (s.estado == "sat").mean() if m == "Min-conflicts" else s.estado.isin(["sat", "unsat"]).mean()
            row.append(f"{f(100 * ok, 0)} / {f(1e3 * s.t_s.mean(), 1)}")
        lines.append(" & ".join(row) + " \\\\")
    body = ("\\begin{tabular}{rrrrrr}\n\\toprule\n$D$ & BT+MRV & FC+MRV & FC+MRV+LCV & MAC+MRV+LCV & "
            "Min-conflicts \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_csp_escala.tex", body)
    meth_m = ["BT+MRV", "FC+MRV", "FC+MRV+LCV", "MAC+MRV+LCV", "Min-conflicts"]
    if "mem_kib" in e:
        lines = []
        for m in meth_m:
            row = [esc(m)]
            for D in sorted(e.D.unique()):
                q = e[(e.D == D) & (e.metodo == m)].mem_kib.dropna()
                row.append(f(q.mean(), 0) if len(q) else "--")
            lines.append(" & ".join(row) + " \\\\")
        body = ("\\begin{tabular}{l" + "r" * e.D.nunique() + "}\n\\toprule\nmétodo & " +
                " & ".join(f"$D={int(D)}$" for D in sorted(e.D.unique())) + " \\\\\n\\midrule\n" +
                "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
        write("tab_csp_memoria.tex", body)
    big = e[e.D == e.D.max()]
    for m, key in (("FC+MRV+LCV", "Fc"), ("MAC+MRV+LCV", "Mac"), ("BT+MRV", "Bt"), ("Min-conflicts", "Mc")):
        s = big[big.metodo == m].mem_kib.dropna()
        num(f"cspMem{key}", f(s.mean(), 0) if len(s) else "--")
        q = big[big.metodo == m]
        num(f"cspT{key}", f(1e3 * q.t_s.mean(), 1))
    num("cspDmax", int(e.D.max()))
    for D in (28, 30, 32):
        for m, key in (("MAC+MRV+LCV", "Mac"), ("FC+MRV+LCV", "Fc"), ("BT+MRV", "Bt"), ("FC+MRV", "FcMrv")):
            s = d[(d.D == D) & (d.metodo == m)]
            if len(s):
                num(f"cspNod{key}D{D}", f(s.nodos.median(), 0))
                num(f"cspT{key}D{D}", f(1e3 * s.t_s.mean(), 0))
                num(f"cspDone{key}D{D}", f"{int(s.estado.isin(['sat', 'unsat']).sum())}/{len(s)}")
    small = d[(d.D == d.D.max())]
    for m, key in (("MAC+MRV+LCV", "Mac"), ("FC+MRV+LCV", "Fc")):
        q = small[small.metodo == m].mem_kib.dropna()
        num(f"cspMemSmall{key}", f(q.mean(), 0) if len(q) else "--")
    valid = d[d.estado == "sat"].valida
    num("cspValidas", f(100 * valid.astype(bool).mean(), 0))
    num("cspSoluciones", int(len(valid)))


# ------------------------------------------------------------------------------ ATSP

def t_atsp():
    ex = pd.read_csv(RES / "atsp_exacto.csv")
    lines = []
    for K in sorted(ex.K.unique()):
        first = True
        meths = [m for m in ["Fuerza bruta", "Held–Karp (DP)", "B&B (MST)", "B&B (MST) + VMC",
                             "B&B (MST) + ACO+LS"] if not ex[(ex.K == K) & (ex.metodo == m)].empty]
        for m in meths:
            s = ex[(ex.K == K) & (ex.metodo == m)]
            mem = s.mem_kib.dropna()
            row = [f"\\multirow{{{len(meths)}}}{{*}}{{{K}}}" if first else "", esc(m).replace("–", "--"),
                   pm(s.nodos.mean(), s.nodos.std(), 0), pm(1e3 * s.t_s.mean(), 1e3 * s.t_s.std(), 1),
                   f(mem.mean(), 1) if len(mem) else "--", f(100 * s.ok.mean(), 0)]
            lines.append(" & ".join(row) + " \\\\")
            first = False
        lines.append("\\midrule")
    body = ("\\begin{tabular}{llrrrr}\n\\toprule\n$K$ & método & nodos / estados & tiempo [ms] & mem. [KiB] & "
            "óptimo [\\%] \\\\\n\\midrule\n" + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_atsp_exacto.tex", body)
    for m, key in (("B&B (MST)", "Bb"), ("B&B (MST) + VMC", "BbVmc"), ("B&B (MST) + ACO+LS", "BbAco"),
                   ("Held–Karp (DP)", "Hk")):
        s = ex[(ex.K == 12) & (ex.metodo == m)]
        num(f"ex{key}Nodos", f(s.nodos.mean(), 0))
        num(f"ex{key}T", f(1e3 * s.t_s.mean(), 0))
    num("exTodosOk", f(100 * ex.ok.mean(), 0))
    for m, key in (("B&B (MST)", "Bb"), ("Held–Karp (DP)", "Hk"), ("B&B (MST) + ACO+LS", "BbAco")):
        q = ex[(ex.K == 12) & (ex.metodo == m)].mem_kib.dropna()
        num(f"ex{key}Mem", f(q.mean(), 1) if len(q) else "--")
    meta = pd.read_csv(RES / "atsp_meta.csv")
    inst = pd.read_csv(RES / "atsp_instancias.csv")
    meths = ["SA", "Tabu", "GA-OX", "ACO (MMAS)", "UCB-HH", "GA memético", "ACO+LS"]
    lines = []
    for r in inst.sort_values(["K", "N"]).itertuples():
        s = meta[(meta.N == r.N) & (meta.K == r.K)]
        best = s.groupby("metodo").brecha_pct.mean().min()
        row = [str(r.K), str(r.N), f(r.opt, 1)]
        for m in meths:
            q = s[s.metodo == m].brecha_pct
            cell = pm(q.mean(), q.std(), 2)
            if abs(q.mean() - best) < 1e-9:
                cell = "\\textbf{" + cell + "}"
            row.append(cell)
        lines.append(" & ".join(row) + " \\\\")
    body = ("\\begin{tabular}{rrr" + "r" * len(meths) + "}\n\\toprule\n$K$ & $N$ & óptimo & " +
            " & ".join(esc(m) for m in meths) + " \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_atsp_meta.tex", body)
    lines = []
    for K in sorted(meta.K.unique()):
        s = meta[meta.K == K]
        row = [str(K)]
        for m in meths:
            q = s[s.metodo == m]
            row.append(f"{f(q.t_s.mean(), 2)} / {f(q.mem_kib.dropna().mean(), 0)}")
        lines.append(" & ".join(row) + " \\\\")
    body = ("\\begin{tabular}{r" + "r" * len(meths) + "}\n\\toprule\n$K$ & " + " & ".join(esc(m) for m in meths) +
            " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_atsp_tiempo.tex", body)
    for m, key in (("ACO (MMAS)", "Aco"), ("SA", "Sa"), ("GA-OX", "Ga"), ("Tabu", "Tabu")):
        q = meta[(meta.K == meta.K.max()) & (meta.metodo == m)].mem_kib.dropna()
        num(f"memMeta{key}", f(q.mean(), 0) if len(q) else "--")
    g = meta.groupby(["K", "metodo"]).brecha_pct.mean().unstack()
    for K in g.index:
        bestm = g.loc[K].idxmin()
        num(f"mejorK{int(K)}", esc(bestm))
        num(f"mejorGapK{int(K)}", f(g.loc[K].min(), 2))
        for m, key in zip(meths, ("Sa", "Tabu", "Ga", "Aco", "Ucb", "Mem", "AcoLs")):
            num(f"gap{key}K{int(K)}", f(g.loc[K, m], 2))
    num("milpMaxT", f(inst.t_milp_s.max(), 1))
    num("apGap", f(100 * (1 - (inst.AP / inst.opt)).mean(), 1))
    num("asimMedia", f(100 * inst.asimetria.mean(), 1))
    hits = meta.groupby("metodo").brecha_pct.apply(lambda x: 100 * (x < 1e-6).mean())
    for m, key in zip(meths, ("Sa", "Tabu", "Ga", "Aco", "Ucb", "Mem", "AcoLs")):
        num(f"hit{key}", f(hits[m], 0))
    curves = json.load(open(RES / "atsp_curvas.json"))
    ucb = [v for k, v in curves.items() if k.startswith("UCB|")]
    if ucb:
        tot = np.zeros(3)
        for inst_runs in ucb:
            for run in inst_runs.values():
                c = np.array(run["counts"])
                tot += c / c.sum()
        tot /= tot.sum()
        num("ucbSwap", f(100 * tot[0], 1))
        num("ucbIns", f(100 * tot[1], 1))
        num("ucbTwo", f(100 * tot[2], 1))


def write_numbers():
    lines = ["% Generado por experiments/make_tables.py — no editar a mano"]
    for k, v in sorted(NUM.items()):
        key = "".join(ch for ch in k if ch.isalpha())
        digits = "".join(ch for ch in k if ch.isdigit())
        if digits:  # LaTeX no admite dígitos en nombres de macro
            key += "".join("ABCDEFGHIJ"[int(c)] for c in digits)
        lines.append(f"\\newcommand{{\\{key}}}{{{v}}}")
    (ROOT / "reporte" / "numeros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("numeros.tex:", len(NUM), "valores")


if __name__ == "__main__":
    for fn in (t_busqueda, t_heuristica, t_mochila, t_csp, t_atsp):
        try:
            fn()
        except FileNotFoundError as e:
            print("falta", e.filename)
    write_numbers()
