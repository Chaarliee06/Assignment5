"""Lectura de los datasets y preprocesamiento.

Datasets sugeridos (Kaggle)
---------------------------
* 2D Path Planning Dataset — https://www.kaggle.com/datasets/dcaffo/2dpathplanningdataset
  Archivos .pt (objetos MapSample de PyTorch) con `map` (100×100, 1 = obstáculo),
  `start`, `goal` y `path` (camino de D* Lite). Aquí se leen tanto esos .pt originales
  como los .npz reproducidos con el generador oficial (mismos campos).
* US Hospital Locations — https://www.kaggle.com/datasets/andrewmvd/us-hospital-locations
  (HIFLD). us_hospital_locations.csv: columnas NAME, LATITUDE, LONGITUDE, BEDS, STATUS, ID, …
* Knapsack Problem Data — https://www.kaggle.com/datasets/warcoder/knapsack-problem
  knapsack_5_items.csv: 10 000 instancias de 5 objetos con la solución óptima;
  knapsack_5_items_new.csv: 9 885 instancias en columnas W1..W5, P1..P5, BP1..BP5.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- knapsack

_num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in _num.findall(s)])


def load_knapsack_kaggle(path: str) -> pd.DataFrame:
    """Devuelve un DataFrame con columnas weights, prices (arreglos), capacity,
    best_picks (arreglo 0/1) y best_price."""
    raw = pd.read_csv(path)
    weights = raw["Weights"].map(lambda s: _parse_vec(s).astype(int))
    # 121 filas traen "Best picks" vacío: ningún kit cabe y "Best price" = 0,
    # así que se interpretan como el vector cero.
    picks = [(_parse_vec(s) if isinstance(s, str) else np.zeros(len(w))).astype(int)
             for s, w in zip(raw["Best picks"], weights)]
    out = pd.DataFrame({
        "weights": weights,
        "prices": raw["Prices"].map(lambda s: _parse_vec(s).astype(int)),
        "capacity": raw["Capacity"].astype(int),
        "best_picks": picks,
        "best_price": raw["Best price"].astype(float),
    })
    return out


def load_knapsack_kaggle_new(path: str) -> pd.DataFrame:
    """knapsack_5_items_new.csv (segundo archivo del mismo dataset): columnas W1..W5, P1..P5,
    Capacity, BP1..BP5 y Best price. Devuelve el mismo formato que load_knapsack_kaggle."""
    raw = pd.read_csv(path)
    w = raw[[f"W{i}" for i in range(1, 6)]].to_numpy().round().astype(int)
    p = raw[[f"P{i}" for i in range(1, 6)]].to_numpy().round().astype(int)
    bp = raw[[f"BP{i}" for i in range(1, 6)]].to_numpy().round().astype(int)
    return pd.DataFrame({
        "weights": list(w),
        "prices": list(p),
        "capacity": raw["Capacity"].round().astype(int),
        "best_picks": list(bp),
        "best_price": raw["Best price"].astype(float),
    })


def kit_pool(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Todos los kits (peso, valor) del dataset, para armar instancias más grandes."""
    w = np.concatenate(df["weights"].to_numpy())
    v = np.concatenate(df["prices"].to_numpy())
    return w, v


# ---------------------------------------------------------------------------- hospitales

LA_BBOX = (33.70, 34.35, -118.70, -117.90)   # lat_min, lat_max, lon_min, lon_max


def load_hospitals(path: str) -> pd.DataFrame:
    """Normaliza a columnas: name, city, lat, lon, beds, hifld_id."""
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    if "latitude" in cols and "name" in cols:
        out = pd.DataFrame({
            "name": df[cols["name"]].astype(str),
            "city": df[cols.get("city", cols["name"])].astype(str),
            "lat": pd.to_numeric(df[cols["latitude"]], errors="coerce"),
            "lon": pd.to_numeric(df[cols["longitude"]], errors="coerce"),
        })
        beds_col = cols.get("beds") or cols.get("staffed all beds")
        out["beds"] = pd.to_numeric(df[beds_col], errors="coerce") if beds_col else np.nan
        id_col = cols.get("hifld-id") or cols.get("id")
        out["hifld_id"] = df[id_col] if id_col else np.nan
        if "status" in cols:  # formato Kaggle/HIFLD: sólo hospitales abiertos
            out = out[df[cols["status"]].astype(str).str.upper().eq("OPEN").to_numpy()]
    else:
        raise ValueError("formato de CSV de hospitales no reconocido")
    out["beds"] = out["beds"].where(out["beds"] > 0)
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def region(df: pd.DataFrame, bbox=LA_BBOX) -> pd.DataFrame:
    lat0, lat1, lon0, lon1 = bbox
    sel = df[df.lat.between(lat0, lat1) & df.lon.between(lon0, lon1)]
    return sel.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)


def project_to_grid(df: pd.DataFrame, n: int, allowed: np.ndarray, bbox=LA_BBOX):
    """Proyecta (lat, lon) a celdas de una rejilla n×n y las ajusta a la celda permitida
    libre más cercana que no esté ocupada por otro hospital (BFS en anillos).

    allowed: máscara (n*n,) de celdas utilizables (p. ej. la componente conexa mayor).
    Devuelve un arreglo de índices de nodo (uno por fila de df).
    """
    lat0, lat1, lon0, lon1 = bbox
    rows = np.clip(np.round((lat1 - df.lat.to_numpy()) / (lat1 - lat0) * (n - 1)), 0, n - 1).astype(int)
    cols = np.clip(np.round((df.lon.to_numpy() - lon0) / (lon1 - lon0) * (n - 1)), 0, n - 1).astype(int)
    ok = allowed.reshape(n, n).copy()
    taken = np.zeros((n, n), dtype=bool)
    nodes = []
    for r, c in zip(rows, cols):
        found = None
        for rad in range(0, n):
            r0, r1 = max(0, r - rad), min(n - 1, r + rad)
            c0, c1 = max(0, c - rad), min(n - 1, c + rad)
            best, bd = None, None
            for rr in range(r0, r1 + 1):
                for cc in (range(c0, c1 + 1) if rr in (r0, r1) else (c0, c1)):
                    if ok[rr, cc] and not taken[rr, cc]:
                        d = (rr - r) ** 2 + (cc - c) ** 2
                        if bd is None or d < bd:
                            best, bd = (rr, cc), d
            if best is not None:
                found = best
                break
        if found is None:
            raise ValueError("no hay celdas libres suficientes")
        taken[found] = True
        nodes.append(found[0] * n + found[1])
    return np.array(nodes)


# --------------------------------------------------------------- mapas (path planning)

# ---- lector de archivos .pt de PyTorch sin instalar PyTorch -------------------------
# Un .pt (formato zip de torch ≥ 1.6) contiene `<prefijo>/data.pkl` (el objeto serializado
# con pickle) y `<prefijo>/data/<k>` (los bytes crudos de cada tensor). Los tensores se
# reconstruyen con torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...).

_STORAGE_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.dtype = np.dtype(_STORAGE_DTYPES[name])


class _MapSampleStub:
    """Contenedor para des-serializar dataset.map_sample.MapSample."""

    def __setstate__(self, state):
        self.__dict__.update(state)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *args):
    item = storage.dtype.itemsize
    size, stride = tuple(size), tuple(stride)
    if len(size) == 0:
        return storage[storage_offset].copy()
    if 0 in size:
        return np.zeros(size, dtype=storage.dtype)
    return np.lib.stride_tricks.as_strided(storage[storage_offset:], shape=size,
                                           strides=tuple(s * item for s in stride)).copy()


def _load_pt_numpy(path: str):
    import collections
    import pickle
    import zipfile

    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("data.pkl")]

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name.startswith("_rebuild_tensor"):
                    return _rebuild_tensor_v2
                if module == "torch" and name in _STORAGE_DTYPES:
                    return _StorageType(name)
                if module == "dataset.map_sample" and name == "MapSample":
                    return _MapSampleStub
                if module == "collections" and name == "OrderedDict":
                    return collections.OrderedDict
                if module == "torch" and name == "device":
                    return lambda *a, **k: "cpu"
                return super().find_class(module, name)

            def persistent_load(self, pid):
                _, storage_type, key, _location, _numel = pid
                raw = zf.read(f"{prefix}data/{key}")
                return np.frombuffer(raw, dtype=storage_type.dtype)

        with zf.open(pkl) as fh:
            return _Unpickler(fh).load()


def load_map_sample(path: str) -> dict:
    """Lee un mapa (.npz reproducido o .pt original de Kaggle).

    Devuelve dict con map (bool, True = obstáculo), start, goal y path (k×2). Los .pt se leen
    sin PyTorch (ver `_load_pt_numpy`).
    """
    if path.endswith(".npz"):
        z = np.load(path)
        return {"map": z["map"].astype(bool), "start": z["start"].astype(int),
                "goal": z["goal"].astype(int), "path": z["path"].astype(int), "file": path}
    if path.endswith(".pt"):
        s = _load_pt_numpy(path)
        return {"map": np.asarray(s.map) > 0, "start": np.asarray(s.start).astype(int).ravel(),
                "goal": np.asarray(s.goal).astype(int).ravel(),
                "path": np.asarray(s.path).astype(int).reshape(-1, 2), "file": path}
    raise ValueError(path)


def list_map_files(folder: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(folder, "**", "*.pt"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "**", "*.npz"), recursive=True))
    return files
