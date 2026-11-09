"""Experimentos del Problema 1.

Se fija un solo hilo para las bibliotecas numéricas (OpenMP/BLAS): los tiempos se miden por
proceso y varios hilos compitiendo por 2 núcleos distorsionaban la inferencia del modelo.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
