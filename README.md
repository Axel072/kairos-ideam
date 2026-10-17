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

En Linux/macOS, `source venv/bin/activate`.

## Uso

Los comandos se corren desde la raíz del repositorio. Región, años, mes, variable y
centros están en `config/params.yaml`.

Descarga (los archivos quedan en `data/raw/`; si ya existen con el mismo request se reusan):

```
python -m ideam_seasonal.ingest ncep
python -m ideam_seasonal.ingest dwd
python -m ideam_seasonal.ingest bom
python -m ideam_seasonal.ingest chirps
```

Verificación de un archivo (longitudes, unidades y acumulación):

```
python -m ideam_seasonal.verify data/raw/ncep_precipitation_m05_box.grib
```

Homogeneización requiere los cuatro grupos de descargas en `data/raw/`. Reduce NCEP, DWD y
BOM a (año, miembro, anticipación), pasa todo a mm/día, y promedia CHIRPS a la celda de 1° de
los modelos. Escribe `data/processed/homogeneizado_precipitation_m05_box.nc`, con un grupo por
fuente (`ncep`, `dwd`, `bom`, `chirps`, `chirps_1deg`):

```
python -m ideam_seasonal.homogenize
```

Se lee con `xr.open_dataset(ruta, group="ncep")`. `lead` 1 es el mes de inicialización
(mayo).

Informe de control de calidad (escribe `docs/informe_control_calidad.md`; necesita el
archivo homogeneizado, así que corre antes el paso anterior):

```
python -m ideam_seasonal.verify --report
```

Tests:

```
python -m pytest
```

## Estructura

```
config/params.yaml        región, variables, años, centros
src/ideam_seasonal/
  ingest.py               descarga (cdsapi y CHIRPS)
  verify.py               chequeos e informe
  homogenize.py           reducción de ensambles y dataset homogeneizado
tests/
docs/                     hallazgos.md, informe_control_calidad.md
data/                     no se versiona (raw/ descargas, processed/ homogeneizado)
```

## Notas

- La región de `params.yaml` es una caja de prueba, no Colombia completo.
