"""
dashboard.py - (Extra) Tablero de visualizacion con los resultados principales.

Genera:
  * ../reporte/figuras/dashboard.pdf    una pagina con un panel por parte
  * results/dashboard.html              la misma figura + tablas resumen
                                         (se abre con cualquier navegador)
"""
import base64
import io
import os

import numpy as np
import pandas as pd

from comun import COLORES, RES, guardar_fig, leer_json, plt


def _png(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110)
    return base64.b64encode(b.getvalue()).decode()


def construir():
    leer = lambda n: pd.read_csv(os.path.join(RES, n + ".csv"))
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Smart Retail Store — tablero de resultados", fontsize=13, weight="bold")

    # 1) Logistica y EM
    r1 = leer_json("p1_resumen")
    seg = leer_json("segmentos")
    ax[0, 0].bar([f"seg. {k + 1}\n£{g:,.0f}" for k, g in enumerate(seg["gasto_medio"])], seg["pesos"],
                 color=COLORES[:seg["K"]])
    ax[0, 0].set_ylabel("fracción de clientes")
    ax[0, 0].set_title(f"1. EM: {seg['K']} segmentos | logística AUC {r1['logistica']['auc'][0]:.3f}")

    # 2) Precios
    t2 = leer("p2_tabla")
    t2 = t2[t2.d == 10].sort_values("brecha")
    ax[0, 1].barh(t2.metodo, t2.brecha, xerr=t2.brecha_std, color=COLORES[0])
    ax[0, 1].invert_yaxis(); ax[0, 1].set_xlabel("brecha vs. mejor [%] (d = 10)")
    ax[0, 1].set_title("2. Precios no convexos (menor es mejor)")

    # 3) Stackelberg
    t3 = leer("p3_mochila").groupby("Z").mean(numeric_only=True)
    a = ax[0, 2]
    a.plot(t3.index, t3.t_bb * 1e3, "o-", label="B&B (exacto)")
    a.plot(t3.index, t3.t_aco * 1e3, "s-", label="ACO")
    a.set_yscale("log"); a.set_xlabel("zonas"); a.set_ylabel("tiempo [ms]"); a.legend(fontsize=8)
    a2 = a.twinx()
    a2.plot(t3.index, t3.brecha_aco_pct, "^--", color="gray"); a2.set_ylabel("brecha ACO [%]", color="gray")
    a.set_title("3. Guardias: B&B vs. ACO (caso 3 zonas: 13)")

    # 4) Juego
    b4 = leer("p4_busqueda")
    g = b4[b4.n == 5].groupby(["prof", "metodo"]).nodos.mean().unstack()
    for k, c in enumerate(g.columns):
        ax[1, 0].plot(g.index, g[c], "o-", color=COLORES[k], label=c)
    ax[1, 0].set_yscale("log"); ax[1, 0].set_xlabel("profundidad"); ax[1, 0].set_ylabel("nodos")
    ax[1, 0].legend(fontsize=7); ax[1, 0].set_title("4. Guardia vs. ladrón (5×5)")

    # 5) Bandits
    b5 = leer("p5_bandits").groupby(["p", "algoritmo"]).regret_final.mean().unstack()
    b5.T.plot.bar(ax=ax[1, 1], rot=0, color=COLORES[3:5])
    ax[1, 1].set_ylabel("regret final"); ax[1, 1].set_title("5a. Bandits (2×10⁴ visitas)")
    ax[1, 1].legend([f"p = {p}" for p in b5.index], fontsize=8)

    # 5b) Guardia
    g5 = leer("p5_guardia").groupby("politica").tasa_robo_final.agg(["mean", "std"]).sort_values("mean")
    ax[1, 2].barh(g5.index, g5["mean"], xerr=g5["std"], color=COLORES[5])
    ax[1, 2].invert_yaxis(); ax[1, 2].set_xlabel("tasa de robo (últimas 2000 visitas)")
    ax[1, 2].set_title("5b. Política del guardia")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    img = _png(fig)
    guardar_fig(fig, "dashboard")

    tablas = {
        "Precios (d = 10)": leer("p2_tabla").query("d == 10").round(3),
        "Mochila de Stackelberg": leer("p3_mochila").groupby("Z").mean(numeric_only=True).round(4).reset_index(),
        "Bandits": leer("p5_bandits").groupby(["p", "algoritmo"]).mean(numeric_only=True).round(3).reset_index(),
        "Política del guardia": leer("p5_guardia").groupby("politica").mean(numeric_only=True).round(4).reset_index(),
    }
    html = ["<html><head><meta charset='utf-8'><title>Smart Retail Store</title>",
            "<style>body{font-family:sans-serif;margin:24px;max-width:1300px}"
            "table{border-collapse:collapse;font-size:13px;margin-bottom:24px}"
            "td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}</style></head><body>",
            "<h1>Smart Retail Store — tablero de resultados</h1>",
            f"<img src='data:image/png;base64,{img}' style='width:100%'>"]
    for titulo, t in tablas.items():
        html += [f"<h2>{titulo}</h2>", t.drop(columns=["semilla"], errors="ignore").to_html(index=False)]
    html.append("</body></html>")
    with open(os.path.join(RES, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print("  dashboard: results/dashboard.html y ../reporte/figuras/dashboard.pdf")


if __name__ == "__main__":
    construir()
