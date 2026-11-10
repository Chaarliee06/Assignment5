# Problema 1: Flota de drones para respuesta a desastres

**Assignment 5 (Capítulo 5)** · TC2038 Análisis y Diseño de Algoritmos Avanzados, Gpo. 603, Equipo 8 · Tecnológico de Monterrey, Campus Monterrey

**Autores:** Carlos Emiliano Nava Flores (A00842650) · Luis Enrique Castillo Carrillo (A00842390)

Reporte: `reporte/P1_drones.pdf` · Panel interactivo (puntos extra): `dashboard/panel_drones.html`

## Resultados principales

- **Rutas**: en los 2 506 mapas originales de Kaggle UCS y A* coinciden siempre entre sí. Los caminos
  de referencia se generaron dejando una celda de holgura alrededor de los obstáculos; con ese modelo el
  costo de A* coincide en 2 091 mapas (83.4 %) y en el resto A* encuentra uno ligeramente más barato
  (mediana 0.84 %), nunca uno más caro. En las ciudades (N = 50…200, viento con c(u,v) ≠ c(v,u)) A*
  expande 47.5 % menos nodos que UCS. IDA* resolvió 25/40 consultas con umbral controlado (IDA*-CR) y
  7/40 con el umbral clásico: en rejillas re-expande demasiado.
- **Heurística aprendida** (entrenada con 300 mapas de `train/` de Kaggle): la cota costo ≤ w_P·C* se
  cumplió en el 100 % de las consultas. En la escala de entrenamiento ahorra nodos (en los mapas de
  prueba, 2 389 con el modelo de cuantil 0.2 y 2 835 con MSE contra 3 855 de A*; en las ciudades con
  N = 100, 699 y 653 contra 857); fuera de ella (N = 200) deja de ayudar.
- **Carga útil**: 0 discrepancias en las 10 000 instancias de Kaggle (ni en las 9 885 del segundo
  archivo del dataset); B&B con cota fraccional resuelve
  casi todas las instancias en milisegundos hasta n = 120 (una con kits densos incompatibles llegó al
  límite); GA a 0.34 % del óptimo en promedio; MITM resuelve el "lo más cerca de W" con n = 40.
- **CSP**: transición de fase clara; MRV es lo que más ayuda y LCV ayuda en las instancias difíciles.
  MAC (AC-3 mantenido) recorre entre 2 y 4.5 veces menos nodos (mediana) que forward checking en la región difícil,
  pero cada nodo cuesta mucho más.
- **ATSP**: Held–Karp y B&B (cota MST) coinciden con K ≤ 12. Con 3·10⁴ evaluaciones y 10 semillas,
  ACO+LS (híbrido) tuvo la menor brecha media con K = 25 (0.07 %), K = 50 (2.22 %) y K = 100 (8.62 %)
  contra el óptimo exacto de HiGHS.

## Qué hay aquí

```
drones/                 biblioteca (todo el código de los algoritmos)
  grid.py               ciudad N×N como grafo dirigido: zonas de exclusión y viento (c(u,v) ≠ c(v,u))
  mapgen.py             generador de mapas con el procedimiento del 2D Path Planning Dataset
  search.py             UCS, A*, A* ponderado, A* con reapertura, IDA* (clásico e IDA*-CR, con/sin TT), b*
  learned.py            heurística aprendida (gradient boosting) y factores de sobreestimación w
  knapsack.py           backtracking con máscaras de bits, B&B con cota fraccional, MITM, GA, HiGHS
  csp.py                CSP de plataformas/turnos: BT, FC, AC-3, MAC, MRV, LCV, min-conflicts
  atsp.py               Held–Karp, B&B con cota MST, SA, Tabú, GA-OX, ACO (MMAS), UCB-HH, híbridos, HiGHS
  scenario.py           escenario integrado con hospitales reales de Los Ángeles
  datasets.py           lectura de los datasets de Kaggle y proyección a la rejilla
  metrics.py            tiempo, memoria (tracemalloc), CSV
experiments/            un script por bloque + figuras, tablas LaTeX y panel
datos/                  datasets usados (ver abajo)
resultados/             CSV/JSON de todos los experimentos y los modelos de la heurística aprendida
figuras/                figuras del reporte
reporte/                fuente LaTeX (tablas y números generados automáticamente) y PDF
dashboard/              panel HTML autocontenido
run_all.py              corre todo
```

## Cómo correrlo

```bash
cd problema1
pip install -r requirements.txt
python run_all.py --rapido   # figuras, tablas y panel a partir de resultados/ (no necesita los mapas)
python run_all.py            # todos los experimentos (≈ 1.5 h con 2 núcleos; necesita los mapas, ver Datasets)
```

Cada bloque también se puede correr por separado, por ejemplo
`python -m experiments.exp_search --parte rejillas` o `python -m experiments.exp_atsp --parte meta`.
`python -m experiments.exp_memoria` va al final: mide en corridas aparte la memoria pico que les
faltaba a algunos métodos y la agrega a los CSV.
Después, `cd reporte && pdflatex P1_drones.tex` (dos veces).

## Datasets

| Kaggle | Fuente usada | Archivo |
|---|---|---|
| [2D Path Planning Dataset](https://www.kaggle.com/datasets/dcaffo/2dpathplanningdataset) | **archivo original de Kaggle** (`archive.zip`, 646 MB, descargado el 8 nov 2026); muestra sistemática de 2 506 mapas (uno de cada *k* archivos del zip) | `datos/mapas_kaggle/map_dataset/{train,validation,test}/*.pt` |
| [US Hospital Locations](https://www.kaggle.com/datasets/andrewmvd/us-hospital-locations) | **archivo original de Kaggle** (descargado el 9 nov 2026); sólo hospitales abiertos (`STATUS = OPEN`) de la zona de Los Ángeles | `datos/us_hospital_locations.csv` |
| [Knapsack Problem Data](https://www.kaggle.com/datasets/warcoder/knapsack-problem) | **archivos originales de Kaggle** (descargados el 9 nov 2026; SHA-256 de `knapsack_5_items.csv`: `55b98b9d…044c`) | `datos/knapsack_5_items.csv`, `datos/knapsack_5_items_new.csv` |

- La muestra de mapas no está en el repositorio de GitHub (son 2 506 archivos binarios, ≈ 110 MB).
  `datos/mapas_kaggle_lista.txt` tiene sus rutas exactas: descarga
  `archive.zip` de Kaggle y corre `python -m experiments.extraer_mapas_kaggle ruta/a/archive.zip` para
  recrearla en `datos/mapas_kaggle/`. Para usar el dataset completo, descomprime todo `archive.zip` ahí.
- Los `.pt` de Kaggle se leen **sin PyTorch** (`drones/datasets.py` reconstruye los tensores desde el zip).
- `datos/mapas_2d/*.npz` son 364 mapas regenerados con el generador oficial del autor
  ([dcaffo98/path-planning-cnn](https://github.com/dcaffo98/path-planning-cnn)) que se usaron en la
  primera versión; sólo se usan si `datos/mapas_kaggle/` está vacío (sin los mapas de Kaggle, una
  corrida completa daría otros números que los del reporte).
