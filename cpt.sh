#!/usr/bin/env bash
# Compatibilidad: equivale a `bash kairos.sh` (mismas opciones). Usa kairos.sh.
exec bash "$(dirname "${BASH_SOURCE[0]}")/kairos.sh" "$@"
