#!/usr/bin/env bash
# Compila el reporte y arma los zips de entrega (código completo y proyecto de Overleaf).
set -euo pipefail
cd "$(dirname "$0")"
rm -rf reporte/figuras && mkdir -p reporte/figuras && cp figuras/*.png reporte/figuras/
(cd reporte && pdflatex -interaction=nonstopmode -halt-on-error P1_drones.tex >/dev/null \
            && pdflatex -interaction=nonstopmode -halt-on-error P1_drones.tex >/dev/null)
mkdir -p entrega
cp reporte/P1_drones.pdf entrega/P1_drones.pdf
cp dashboard/panel_drones.html entrega/panel_drones.html
rm -f entrega/*.zip
(cd reporte && zip -qr ../entrega/P1_drones_overleaf.zip P1_drones.tex numeros.tex tablas figuras)
zip -qr entrega/Assignment5_P1_completo.zip README.md requirements.txt run_all.py empaquetar.sh \
    drones experiments datos resultados figuras dashboard \
    reporte/P1_drones.tex reporte/P1_drones.pdf reporte/numeros.tex reporte/tablas \
    -x "*/__pycache__/*" "*.pyc" "resultados/*_a.*" "resultados/*_b.*" "resultados/*_reproducidos.*"
ls -la entrega
