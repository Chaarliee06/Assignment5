# Problema 2: Tienda inteligente (Smart Retail Store)

**Assignment 5 (Capítulo 5)** · TC2038 Análisis y Diseño de Algoritmos Avanzados, Gpo. 603, Equipo 8 · Tecnológico de Monterrey, Campus Monterrey

**Autores:** Desiderio Iván Ortegón Mortón · Fernando Robles Vidrio

Una tienda fija precios y coloca guardias, mientras que los clientes aprenden si les conviene comprar, robar o irse sin conocer la probabilidad de ser atrapados. El problema combina optimización convexa y no convexa, búsqueda exacta con cotas, búsqueda adversarial, metaheurísticas bioinspiradas y aprendizaje por refuerzo, con experimentos sobre datasets reales de Kaggle.

**Reporte:** [`reporte/P2_tienda.pdf`](reporte/P2_tienda.pdf) (fuente LaTeX en `reporte/P2_tienda.tex` y `reporte/problema2.tex`).

## Contenido

| Parte | Problema | Métodos |
|---|---|---|
| 1 | Modelo de compra y segmentos de clientes | Regresión logística con descenso de gradiente (con demostración de convexidad), EM para mezcla de Gaussianas |
| 2 | Precios de 10 productos relacionados (ingreso no convexo) | Gradiente con reinicios, Hill Climbing, Recocido Simulado, GA real, PSO y GA memético |
| 3 | Colocación de guardias (juego de Stackelberg como mochila 0/1) | Branch & Bound con cota fraccionaria, ACO |
| 4 | Guardia contra ladrón en una cuadrícula | Minimax, alfa-beta (orden aleatorio y buen orden), MCTS (UCT) |
| 5 | Clientes y guardia que aprenden | ε-greedy, UCB1, Thompson sampling, Q-learning, PPO (Stable-Baselines3) |

Puntos extra: GA memético (híbrido búsqueda local + GA) y tablero de resultados (`code/results/dashboard.html`).

## Resultados principales

- **Logística y EM:** exactitud de 0.725 ± 0.011 y AUC de 0.791 ± 0.015 en prueba; EM con BIC encuentra 3 segmentos de gasto.
- **Precios:** con d = 10 todos los métodos quedan a 1 % o menos del mejor ingreso; con d = 40 el GA real queda a 2.2 % y el gradiente con reinicios a 54 %.
- **Guardias:** el Branch & Bound obtiene el óptimo de 13 del enunciado y resuelve de forma exacta instancias de hasta 120 zonas en menos de 4 ms; ACO queda a menos de 0.03 % del óptimo, pero es entre 380 y 740 veces más lento.
- **Juego:** alfa-beta con buen orden da el mismo valor que minimax expandiendo hasta 45 veces menos nodos.
- **Aprendizaje:** Thompson sampling tiene el menor regret; la política de Stackelberg y la aprendida con PPO bajan el robo a ≈3 %, contra 88 % con un guardia predecible.

Todos los métodos estocásticos se corrieron con 10 semillas (media ± desviación estándar).

## Estructura

```
problema2/
├── README.md
├── reporte/
│   ├── P2_tienda.pdf              reporte compilado
│   ├── P2_tienda.tex              documento principal (portada)
│   ├── problema2.tex              texto del Problema 2 (se puede integrar con \input)
│   └── figuras/                   figuras generadas por el código
└── code/
    ├── README.md                  detalle del código
    ├── requirements.txt
    ├── correr_todo.py             corre todos los experimentos
    ├── pruebas.py                 pruebas de correctitud
    ├── comun.py                   utilidades (tiempo, memoria, figuras)
    ├── datos.py                   carga y preprocesamiento de los datasets
    ├── parte1_logistica_em.py
    ├── parte2_precios.py
    ├── parte3_stackelberg.py
    ├── parte4_juego.py
    ├── parte5_aprendizaje.py
    ├── dashboard.py               tablero de resultados
    ├── data/                      aquí van los CSV de Kaggle (no se incluyen)
    └── results/                   CSV y JSON de cada experimento + dashboard.html
```

## Cómo reproducir

```bash
cd problema2/code
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

python3 pruebas.py          # pruebas de correctitud (no necesitan los datos)
python3 correr_todo.py      # experimentos completos (~3 min)
```

Los resultados quedan en `code/results/` y las figuras en `reporte/figuras/`. Para compilar el reporte: `cd reporte && latexmk -pdf P2_tienda.tex`.

## Datos

| Dataset | Archivo | Uso |
|---|---|---|
| [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) | `online_retail_II.csv` | modelo de compra, segmentos y valores de los artículos |
| [Retail Price Optimization](https://www.kaggle.com/datasets/suddharshan/retail-price-optimization) | `retail_price.csv` | modelo de demanda para los precios |

Descargados el 9 de noviembre de 2026. El preprocesamiento se describe en el reporte y en `code/datos.py`. Los archivos no se incluyen en el repositorio por su tamaño; hay que copiarlos a `code/data/`.

## Entorno

MacBook Pro (Apple M5 Pro, 24 GB), macOS 26.6.2, Python 3.13.15, NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, PyTorch 2.15.0, Stable-Baselines3 2.9.2 y Gymnasium 1.4.0. El registro completo de la corrida está en `code/results/ejecucion.json`.
