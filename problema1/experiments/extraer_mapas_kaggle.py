"""Recrea, a partir de archive.zip, la muestra de mapas de Kaggle que usa el reporte.

El 2D Path Planning Dataset (https://www.kaggle.com/datasets/dcaffo/2dpathplanningdataset) se
descarga de Kaggle como archive.zip (646 MB). Los experimentos usan una muestra sistemática de
2 506 de sus mapas; `datos/mapas_kaggle_lista.txt` tiene sus rutas exactas. Este script extrae
sólo esos archivos a `datos/mapas_kaggle/`, sin descomprimir el zip completo.

Uso:  python -m experiments.extraer_mapas_kaggle ruta/a/archive.zip
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

from .common import DATOS

LISTA = DATOS / "mapas_kaggle_lista.txt"
DESTINO = DATOS / "mapas_kaggle"


def leer_lista(path: Path = LISTA) -> list[str]:
    """Rutas de la muestra, relativas a datos/mapas_kaggle/."""
    lineas = path.read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lineas if x.strip() and not x.startswith("#")]


def _clave(ruta: str) -> str:
    # carpeta + nombre del archivo: tolera otro prefijo dentro del zip
    return "/".join(ruta.replace("\\", "/").split("/")[-2:])


def extraer(zip_path: str, destino: Path = DESTINO, lista: Path = LISTA) -> int:
    pendientes = {_clave(r): r for r in leer_lista(lista)}
    total = len(pendientes)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            rel = pendientes.pop(_clave(info.filename), None)
            if rel is None:
                continue
            # la ruta de salida sale de la lista, no del zip
            salida = destino / rel
            salida.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(salida, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"{total - len(pendientes)} de {total} mapas extraídos en {destino}")
    if pendientes:
        print("No se encontraron en el zip, por ejemplo:", *list(pendientes.values())[:5], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(extraer(sys.argv[1]))
