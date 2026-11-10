"""Problema 1 (Capítulo 5): flota de drones para respuesta a desastres.

Módulos
-------
grid      modelo de la ciudad: rejilla N×N, zonas de exclusión y viento (costos asimétricos)
mapgen    generador de mapas (mismo procedimiento que el 2D Path Planning Dataset de Kaggle)
search    UCS, A*, IDA*, IDA*-CR, A* ponderado y A* con reapertura (heurística aprendida)
learned   heurística aprendida (gradient boosting) y verificación de la cota w·C*
knapsack  carga útil: backtracking con máscaras de bits, B&B con cota fraccional, MITM, GA
csp       asignación de plataformas y turnos: BT, FC, AC-3 (MAC), MRV/LCV, min-conflicts
atsp      orden de visitas (TSP asimétrico): Held-Karp, B&B con cota MST, SA, Tabu, GA-OX,
          ACO (MMAS), hiperheurística UCB e híbridos
datasets  lectura de los datasets de Kaggle y proyección de hospitales a la rejilla
metrics   medición de tiempo, memoria y factor de ramificación efectivo
"""
