#!/usr/bin/env bash
# Compatibilidad: equivale a `bash kairos.sh qc` (control de calidad de la Fase 2).
#   bash run.sh [--force]
exec bash "$(dirname "${BASH_SOURCE[0]}")/kairos.sh" qc "$@"
