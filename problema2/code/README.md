# Código del Problema 2 (Smart Retail Store)

Código del Problema 2 de la tarea del capítulo 5. Todo está en Python.

## Estructura

| Archivo | Contenido |
|---|---|
| `datos.py` | carga y limpieza de los datasets de Kaggle |
| `parte1_logistica_em.py` | modelo de compra (regresión logística con descenso de gradiente) y segmentos de clientes (EM) |
| `parte2_precios.py` | precios de 10 productos (ingreso no convexo): gradiente con reinicios, Hill Climbing, Recocido Simulado, GA, PSO y GA memético (extra) |
| `parte3_stackelberg.py` | guardias como juego de Stackelberg → mochila 0/1 con Branch & Bound y ACO |
| `parte4_juego.py` | guardia contra ladrón: minimax, alfa-beta (orden aleatorio y buen orden) y MCTS (UCT) |
| `parte5_aprendizaje.py` | clientes con bandits (ε-greedy, UCB1, Thompson) y Q-learning; guardia con PPO |
| `dashboard.py` | (extra) tablero con los resultados principales (`results/dashboard.html`) |
| `correr_todo.py` | corre todo y guarda resultados en `results/` y figuras en `../reporte/figuras/` |
| `pruebas.py` | pruebas de correctitud (B&B contra fuerza bruta, alfa-beta contra minimax, gradiente y convexidad de la logística, cobertura de Stackelberg) |

## 1. Instalar

```bash
cd code
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 2. Descargar los datasets (Kaggle)

1. Retail Price Optimization: https://www.kaggle.com/datasets/suddharshan/retail-price-optimization → `retail_price.csv`
2. Online Retail II UCI: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci → `online_retail_II.csv`

Descomprimir y copiar los dos `.csv` a `code/data/`. Los datos del reporte se descargaron el 9 de noviembre de 2026.

## 3. Correr

```bash
python3 pruebas.py                  # pruebas de correctitud (no necesitan los datos)
python3 correr_todo.py --rapido     # prueba corta (~1–2 min)
python3 correr_todo.py              # experimentos completos (10 semillas, 2×10⁴ visitas)
```

`python3 correr_todo.py --sintetico` usa datos falsos con las mismas columnas; sirve solo para revisar
que el código corre sin haber descargado nada. Los resultados del reporte salen de los datos reales.

## Resultados

* `results/*.csv` – una fila por método, tamaño y semilla
* `results/*_resumen.json` – números resumidos de cada parte
* `results/ejecucion.json` – equipo, versiones, archivos de datos y tiempo de cada parte
* `results/dashboard.html` – tablero (extra)
* `../reporte/figuras/*.pdf` – figuras del reporte

Tiempo: `time.perf_counter`. Memoria: pico de `tracemalloc` en una corrida aparte (semilla 0),
porque `tracemalloc` hace más lento el código; para PPO se reporta la memoria máxima del proceso.
