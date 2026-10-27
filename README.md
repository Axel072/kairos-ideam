# ideam-seasonal
## Requisitos

- Python 3.10 o superior
- Cuenta en el [CDS de Copernicus](https://cds.climate.copernicus.eu) con la licencia del
  dataset `seasonal-monthly-single-levels` aceptada
- Archivo `%USERPROFILE%\.cdsapirc` (Windows) o `~/.cdsapirc` con tu token:

  ```
  url: https://cds.climate.copernicus.eu/api
  key: <tu-token>
  ```

## Instalación

Desde la raíz del repositorio:

```
python -m venv venv
venv\Scripts\activate
pip install -e .
```

En Linux/macOS, `source venv/bin/activate`. La activación solo hace falta para correr los
módulos a mano o los tests; `run.sh` no la necesita.

## Uso

Todo el flujo se corre con un solo script, `run.sh`, desde Git Bash (en cualquier carpeta;
usa el `venv` del repositorio sin necesidad de activarlo). Región, años, mes, variable y
centros están en `config/params.yaml`.

```
bash run.sh            # usa la caché de data/raw/
bash run.sh --force    # vuelve a descargar todo
```

El script se detiene en el primer paso que falle (`set -e`) y ejecuta, en orden:

1. **Descarga** de NCEP, DWD, BOM y CHIRPS a `data/raw/` (`ingest`). Si ya existe un archivo
   con el mismo request se reusa.
2. **Verificación** de cada GRIB (`verify <archivo>`): longitudes, unidades y acumulación.
   Si una fuente falla, el flujo se detiene antes de homogeneizar.
3. **Homogeneización** (`homogenize`): reduce NCEP, DWD y BOM a (año, miembro, anticipación),
   pasa todo a mm/día y promedia CHIRPS a la celda de 1° de los modelos. Escribe
   `data/processed/homogeneizado_precipitation_m05_box.nc`, con un grupo por fuente (`ncep`,
   `dwd`, `bom`, `chirps`, `chirps_1deg`).
4. **Informe de control de calidad** (`verify --report`): escribe
   `docs/informe_control_calidad.md`.

El archivo homogeneizado se lee con `xr.open_dataset(ruta, group="ncep")`. `lead` 1 es el
mes de inicialización (mayo).

Tests:

```
python -m pytest
```

## Estructura

```
run.sh                    corre el flujo completo (descarga, verificación, homogeneización, informe)
config/params.yaml        región, variables, años, centros
src/ideam_seasonal/
  ingest.py               descarga (cdsapi y CHIRPS)
  common.py               constantes y helpers compartidos (unidades, lectura de GRIB/CHIRPS)
  verify.py               chequeos e informe
  homogenize.py           reducción de ensambles y dataset homogeneizado
tests/
docs/                     hallazgos.md, informe_control_calidad.md
data/                     no se versiona (raw/ descargas, processed/ homogeneizado)
```

## Notas

- La región de `params.yaml` es una caja de prueba, no Colombia completo.
