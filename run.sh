#!/usr/bin/env bash
#   bash run.sh [--force]
#   --force   ignora la caché de data/raw/ y vuelve a descargar
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONIOENCODING=utf-8  # tildes legibles en la consola de Windows

if [[ -x venv/Scripts/python.exe ]]; then
    PY=venv/Scripts/python.exe
elif [[ -x venv/bin/python ]]; then
    PY=venv/bin/python
else
    echo "No se encontró el venv. Crea uno con: python -m venv venv && pip install -e ." >&2
    exit 1
fi

FORCE=()
[[ "${1:-}" == "--force" ]] && FORCE=(--force)

CENTRES=(ncep dwd bom)
VARIABLE=precipitation

echo "== 1/4 Descarga =="
for src in "${CENTRES[@]}" chirps; do
    "$PY" -m ideam_seasonal.ingest "$src" "${FORCE[@]}"
done

echo "== 2/4 Verificación de cada GRIB =="
for c in "${CENTRES[@]}"; do
    for f in data/raw/"${c}_${VARIABLE}"_m*_box.grib; do
        "$PY" -m ideam_seasonal.verify "$f"
    done
done

echo "== 3/4 Homogeneización =="
"$PY" -m ideam_seasonal.homogenize

echo "== 4/4 Informe de control de calidad =="
"$PY" -m ideam_seasonal.verify --report
