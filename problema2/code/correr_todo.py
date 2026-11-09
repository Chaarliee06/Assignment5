"""
correr_todo.py - Corre todas las partes del Problema 2 y guarda resultados.

Uso:
  python3 correr_todo.py              # experimentos completos (datos de Kaggle en data/)
  python3 correr_todo.py --rapido     # prueba corta (menos semillas y tamanos)
  python3 correr_todo.py --sintetico  # usa datos FALSOS solo para probar que corre
  python3 correr_todo.py --partes 3 5 # solo algunas partes (la 2 necesita la 1)
"""
import argparse
import datetime
import importlib
import platform
import subprocess
import sys
import time

import comun
import datos


def equipo():
    info = dict(fecha=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), sistema=platform.platform(),
                python=platform.python_version(), procesador=platform.processor(), maquina=platform.machine())
    try:
        if sys.platform == "darwin":
            info["procesador"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                capture_output=True, text=True).stdout.strip()
            info["memoria_gb"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                                    text=True).stdout) / 2 ** 30
        else:
            for ln in open("/proc/cpuinfo"):
                if ln.startswith("model name"):
                    info["procesador"] = ln.split(":")[1].strip(); break
    except Exception:
        pass
    for m in ("numpy", "pandas", "sklearn", "matplotlib", "torch", "stable_baselines3", "gymnasium"):
        try:
            info[m] = importlib.import_module(m).__version__
        except Exception:
            info[m] = "no instalado"
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--partes", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    datos.revisar_archivos(a.sintetico)
    if a.sintetico:
        print("*** OJO: datos SINTETICOS, solo para probar que el codigo corre ***")
    info = dict(equipo=equipo(), datos=datos.info_archivos(a.sintetico), rapido=a.rapido, tiempos_s={})
    print(f"Equipo: {info['equipo']['procesador']} | Python {info['equipo']['python']}")
    modulos = {1: "parte1_logistica_em", 2: "parte2_precios", 3: "parte3_stackelberg",
               4: "parte4_juego", 5: "parte5_aprendizaje"}
    for k in a.partes:
        t0 = time.perf_counter()
        importlib.import_module(modulos[k]).correr(sintetico=a.sintetico, rapido=a.rapido)
        info["tiempos_s"][f"parte{k}"] = round(time.perf_counter() - t0, 1)
        print(f"  -> parte {k} en {info['tiempos_s'][f'parte{k}']} s\n", flush=True)
    comun.guardar_json(info, "ejecucion")
    if set(a.partes) == {1, 2, 3, 4, 5}:
        import dashboard
        dashboard.construir()
    print("Listo. Resultados en code/results/ y figuras en reporte/figuras/")


if __name__ == "__main__":
    main()
