"""
datos.py - Carga y preprocesamiento de los datasets de Kaggle.

Datasets usados (se descargan a mano y se ponen en code/data/):
  1. Retail Price Optimization
     https://www.kaggle.com/datasets/suddharshan/retail-price-optimization
     archivo: retail_price.csv  (676 filas, 30 columnas)
  2. Online Retail II UCI
     https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
     archivo: online_retail_II.csv  (~1.07 millones de filas)

Preprocesamiento (se describe tambien en el reporte):
  * Online Retail II: se quitan filas sin cliente, facturas canceladas
    (Invoice que empieza con 'C'), cantidades o precios <= 0 y precios
    atipicos (> percentil 99.9). Despues se agrega por cliente.
  * Retail Price Optimization: se agrupa por product_id (precio de
    referencia = mediana de unit_price, demanda base = media de qty,
    precio del competidor = mediana de comp_1).

La opcion `sintetico=True` genera archivos FALSOS con las mismas columnas.
Solo sirve para probar que el codigo corre; los resultados del reporte
deben salir de los datos reales.
"""
import os

import numpy as np
import pandas as pd

from comun import DATOS

ARCH_PRECIOS = "retail_price.csv"
ARCH_RETAIL = "online_retail_II.csv"
URL_PRECIOS = "https://www.kaggle.com/datasets/suddharshan/retail-price-optimization"
URL_RETAIL = "https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci"


def _carpeta(sintetico):
    return os.path.join(DATOS, "sintetico") if sintetico else DATOS


def _normaliza_columnas(df):
    df.columns = [c.strip().lower().replace(" ", "").replace("_", "") for c in df.columns]
    return df


def revisar_archivos(sintetico=False):
    """Revisa que existan los CSV; si no, explica como conseguirlos."""
    if sintetico:
        generar_sinteticos()
        return
    faltan = [a for a in (ARCH_PRECIOS, ARCH_RETAIL) if not os.path.exists(os.path.join(DATOS, a))]
    if faltan:
        raise SystemExit(
            "\nFaltan archivos en code/data/: " + ", ".join(faltan) +
            f"\n  - {ARCH_PRECIOS}: {URL_PRECIOS}\n  - {ARCH_RETAIL}: {URL_RETAIL}\n"
            "Descargalos de Kaggle, descomprimelos y copia los .csv a code/data/.\n"
            "(Para solo probar que el codigo corre: python correr_todo.py --sintetico)\n")


def info_archivos(sintetico=False):
    """Fuente, tamano y fecha de modificacion de cada archivo (para el reporte)."""
    info = {}
    for a, url in ((ARCH_PRECIOS, URL_PRECIOS), (ARCH_RETAIL, URL_RETAIL)):
        p = os.path.join(_carpeta(sintetico), a)
        info[a] = dict(url=url, existe=os.path.exists(p),
                       mb=round(os.path.getsize(p) / 2 ** 20, 2) if os.path.exists(p) else None,
                       modificado=pd.Timestamp(os.path.getmtime(p), unit="s").strftime("%Y-%m-%d")
                       if os.path.exists(p) else None,
                       fuente="SINTETICO (solo prueba)" if sintetico else "Kaggle")
    return info


# ---------------------------------------------------------------------------
# Online Retail II
# ---------------------------------------------------------------------------
def cargar_transacciones(sintetico=False):
    p = os.path.join(_carpeta(sintetico), ARCH_RETAIL)
    df = pd.read_csv(p, encoding="latin-1", dtype={"Invoice": str, "InvoiceNo": str, "StockCode": str})
    df = _normaliza_columnas(df)
    df = df.rename(columns={"invoiceno": "invoice", "unitprice": "price"})
    n0 = len(df)
    df = df.dropna(subset=["customerid"])
    df = df[~df["invoice"].astype(str).str.startswith("C")]
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    df = df[df["price"] <= df["price"].quantile(0.999)]
    fechas = pd.to_datetime(df["invoicedate"], format="ISO8601", errors="coerce")
    if fechas.isna().mean() > 0.5:          # otro formato de fecha (p. ej. 12/1/2009 7:45)
        fechas = pd.to_datetime(df["invoicedate"], format="mixed", errors="coerce")
    df["invoicedate"] = fechas
    df = df.dropna(subset=["invoicedate"])
    df["total"] = df["quantity"] * df["price"]
    df["customerid"] = df["customerid"].astype(int)
    df.attrs["filas_originales"] = n0
    return df


def datos_compra(df, ventana_dias=90):
    """Modelo de compra: para cada cliente, ¿vuelve a comprar en los ultimos
    `ventana_dias` del periodo?  Las variables se calculan SOLO con lo que
    paso antes del corte (para no hacer trampa).

    Regresa X (n x 5, estandarizada), y (0/1) y los nombres de las variables."""
    fin = df["invoicedate"].max()
    corte = fin - pd.Timedelta(days=ventana_dias)
    antes = df[df["invoicedate"] < corte]
    despues = df[df["invoicedate"] >= corte]
    g = antes.groupby("customerid")
    feats = pd.DataFrame({
        "recencia": (corte - g["invoicedate"].max()).dt.days,
        "frecuencia": g["invoice"].nunique(),
        "monto": g["total"].sum(),
        "ticket": g["total"].sum() / g["invoice"].nunique(),
        "antiguedad": (corte - g["invoicedate"].min()).dt.days,
    })
    feats = np.log1p(feats.clip(lower=0))
    y = feats.index.isin(despues["customerid"].unique()).astype(float)
    X = feats.values.astype(float)
    X = (X - X.mean(0)) / X.std(0)
    return X, y, list(feats.columns)


def gasto_clientes(df):
    """Gasto por cliente para EM: log del gasto total de cada cliente."""
    return np.log(df.groupby("customerid")["total"].sum().values)


def valores_articulos(df, n):
    """Valor (precio mediano) de los n articulos mas vendidos. Se usan como
    valor de las zonas de la tienda en el juego de Stackelberg."""
    top = df.groupby("stockcode")["quantity"].sum().nlargest(max(n, 500)).index
    precios = df[df["stockcode"].isin(top)].groupby("stockcode")["price"].median()
    return precios.values.astype(float)


# ---------------------------------------------------------------------------
# Retail Price Optimization
# ---------------------------------------------------------------------------
def productos(d, sintetico=False):
    """Los d productos con mas ventas: precio de referencia, demanda media,
    precio mediano del competidor 1 (comp_1) y categoria."""
    p = os.path.join(_carpeta(sintetico), ARCH_PRECIOS)
    df = _normaliza_columnas(pd.read_csv(p))
    g = df.groupby("productid").agg(precio=("unitprice", "median"), demanda=("qty", "mean"),
                                    competidor=("comp1", "median"),
                                    categoria=("productcategoryname", "first"),
                                    meses=("qty", "size"))
    g = g.sort_values("demanda", ascending=False).head(d)
    if len(g) < d:
        raise ValueError(f"El archivo solo tiene {len(g)} productos (se pidieron {d}).")
    return g.reset_index()


# ---------------------------------------------------------------------------
# Datos sinteticos con el mismo esquema (solo para probar el codigo)
# ---------------------------------------------------------------------------
def generar_sinteticos():
    carpeta = _carpeta(True)
    os.makedirs(carpeta, exist_ok=True)
    rng = np.random.default_rng(123)
    # --- retail_price.csv (columnas reales del dataset) ---
    p = os.path.join(carpeta, ARCH_PRECIOS)
    if not os.path.exists(p):
        cats = ["bed_bath_table", "garden_tools", "consoles_games", "health_beauty",
                "cool_stuff", "perfumery", "computers_accessories", "watches_gifts", "furniture_decor"]
        filas = []
        for k in range(52):
            base = rng.uniform(20, 300)
            for m in range(13):
                up = base * rng.uniform(0.85, 1.15)
                q = max(1, int(rng.poisson(30 * (base / up) ** 2 * rng.uniform(0.3, 3))))
                filas.append(dict(product_id=f"p{k:02d}", product_category_name=cats[k % 9],
                                  month_year=f"01-{m + 1:02d}-2017", qty=q, total_price=q * up,
                                  freight_price=rng.uniform(5, 30), unit_price=up,
                                  product_name_lenght=40, product_description_lenght=500,
                                  product_photos_qty=2, product_weight_g=500, product_score=4.0,
                                  customers=q * 3, weekday=22, weekend=8, holiday=1, month=m + 1,
                                  year=2017, s=10.0, volume=3000, comp_1=up * 1.05, ps1=4.0,
                                  fp1=15.0, comp_2=up * 0.95, ps2=4.1, fp2=14.0,
                                  comp_3=up, ps3=4.2, fp3=16.0, lag_price=up))
        pd.DataFrame(filas).to_csv(p, index=False)
    # --- online_retail_II.csv (columnas reales del dataset de Kaggle) ---
    p = os.path.join(carpeta, ARCH_RETAIL)
    if not os.path.exists(p):
        n_cli, filas = 3000, []
        seg = rng.choice(3, n_cli, p=[0.5, 0.35, 0.15])
        precios_item = np.round(rng.lognormal(1.0, 0.8, 600), 2)
        inicio = pd.Timestamp("2009-12-01")
        inv = 500000
        for c in range(n_cli):
            n_fac = 1 + rng.poisson([2, 6, 15][seg[c]])
            for _ in range(n_fac):
                inv += 1
                fecha = inicio + pd.Timedelta(minutes=int(rng.uniform(0, 740 * 24 * 60)))
                for _ in range(1 + rng.poisson(4)):
                    it = rng.integers(600)
                    filas.append((str(inv), f"S{it:04d}", "ITEM", int(1 + rng.poisson([3, 8, 20][seg[c]])),
                                  fecha.strftime("%Y-%m-%d %H:%M:%S"), precios_item[it], 12000.0 + c,
                                  "United Kingdom"))
        df = pd.DataFrame(filas, columns=["Invoice", "StockCode", "Description", "Quantity",
                                          "InvoiceDate", "Price", "Customer ID", "Country"])
        canc = df.sample(frac=0.02, random_state=1).copy()
        canc["Invoice"] = "C" + canc["Invoice"]
        canc["Quantity"] *= -1
        sin_cli = df.sample(frac=0.05, random_state=2).copy()
        sin_cli["Customer ID"] = np.nan
        pd.concat([df, canc, sin_cli]).to_csv(p, index=False)
