# Assignment 5: Análisis y Diseño de Algoritmos Avanzados (TC2038)

Capítulo 5 · Gpo. 603 · Equipo 8 · Tecnológico de Monterrey, Campus Monterrey

**Equipo 8:** Luis Enrique Castillo Carrillo · Carlos Emiliano Nava Flores · Desiderio Iván Ortegón Mortón · Fernando Robles Vidrio

| Problema | Autores | Carpeta | Reporte |
|---|---|---|---|
| 1. Flota de drones para respuesta a desastres | Carlos Emiliano Nava Flores, Luis Enrique Castillo Carrillo | [`problema1/`](problema1/) | [`problema1/reporte/P1_drones.pdf`](problema1/reporte/P1_drones.pdf) |
| 2. Tienda inteligente (Smart Retail Store) | Desiderio Iván Ortegón Mortón, Fernando Robles Vidrio | [`problema2/`](problema2/) | [`problema2/reporte/P2_tienda.pdf`](problema2/reporte/P2_tienda.pdf) |

Cada carpeta es independiente: tiene su propio README con los resultados, la descripción de los datos y las instrucciones para reproducir los experimentos.

## Estructura

```
.
├── README.md
├── LICENSE
├── problema1/     Problema 1: código (drones/, experiments/), datos, resultados, figuras, reporte y panel interactivo
└── problema2/     Problema 2: código, resultados, figuras y reporte
```

## Problema 1: Flota de drones para respuesta a desastres

Tras un sismo, una flota de drones lleva kits médicos desde un depósito a clínicas de campo. La ciudad es una rejilla N×N con zonas de exclusión aérea y viento, así que los costos son asimétricos (c(u,v) ≠ c(v,u)). Detalles y resultados en [`problema1/README.md`](problema1/README.md).

| Parte | Métodos |
|---|---|
| Rutas | UCS, A*, IDA* (clásico e IDA*-CR) con heurística octil × costo mínimo y una heurística aprendida |
| Carga útil de cada dron | Backtracking con máscaras de bits, Branch & Bound con cota fraccional, meet-in-the-middle y GA |
| Plataformas y turnos (CSP) | Forward checking, AC-3 / MAC, MRV y LCV |
| Orden de visita (TSP asimétrico) | B&B con cota MST y Held–Karp; recocido simulado, búsqueda tabú, GA con OX, ACO, hiperheurística UCB e híbridos |

Datasets: los tres sugeridos en el enunciado (2D Path Planning Dataset, US Hospital Locations y Knapsack Problem Data); la fuente exacta de cada uno está en el README del problema. Todos los métodos estocásticos se corrieron con 10 semillas.

## Problema 2: Tienda inteligente (Smart Retail Store)

Una tienda fija precios y coloca guardias, mientras que los clientes aprenden si les conviene comprar, robar o irse sin conocer la probabilidad de ser atrapados. Detalles y resultados en [`problema2/README.md`](problema2/README.md).

| Parte | Métodos |
|---|---|
| Modelo de compra y segmentos | Regresión logística con descenso de gradiente (pérdida convexa), EM para mezcla de Gaussianas |
| Precios de 10 productos (ingreso no convexo) | Gradiente con reinicios, Hill Climbing, Recocido Simulado, GA real, PSO y GA memético |
| Guardias como juego de Stackelberg | Mochila 0/1 con Branch & Bound (óptimo 13 del enunciado) y ACO para 30–120 zonas |
| Guardia contra ladrón | Minimax, alfa-beta (orden aleatorio y buen orden), MCTS (UCT) |
| Clientes y guardia que aprenden | ε-greedy, UCB1, Thompson sampling, Q-learning, PPO (Stable-Baselines3) |

Datasets: [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) y [Retail Price Optimization](https://www.kaggle.com/datasets/suddharshan/retail-price-optimization) (Kaggle). Todos los métodos estocásticos se corrieron con 10 semillas.

## Licencia

El código se distribuye bajo la licencia MIT (ver [`LICENSE`](LICENSE)). Los datasets pertenecen a sus autores y se usan bajo las condiciones de sus fuentes originales.
