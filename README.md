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
módulos a mano o los tests; `kairos.sh` no la necesita.

## Uso

Todo se corre con un solo comando, `kairos.sh`, desde Git Bash o bash de Linux (en cualquier
carpeta; usa el `venv` del repositorio sin necesidad de activarlo):

```
bash kairos.sh
```

Sin opciones baja del CDS el pronóstico más reciente publicado y el hindcast del mismo mes de
inicio, verifica cada archivo, verifica el predictando (Y) y escribe en `data/cpt/` los
archivos de entrada de CPT: X (hindcast) y Z (pronóstico) de NCEP, DWD, BOM y su promedio,
para lluvia y temperatura, más un `registro_*.md` por variable. Todo lo que no se indique
sale de `config/params.yaml`.

### Opciones

Se pueden combinar; cada una reemplaza el valor de `params.yaml` solo en esa corrida.
`bash kairos.sh --help` las lista todas.

| Opción | Ejemplo | Por defecto |
|---|---|---|
| `-r`, `--region` | `colombia`, `prueba`, `global` o un cuadro `5,-75,-2,-70` (N,W,S,E, longitudes con signo) | `cpt.region` (colombia) |
| `-m`, `--modelo` | `ncep`, `-m ncep -m dwd` o `ncep,dwd` | los tres y su promedio |
| `-v`, `--variable` | `precipitation`, `temperature` | ambas |
| `-i`, `--inicio` | `2026-10` | el más reciente publicado |
| `-t`, `--temporada` | `2-4` (leads; 1 = mes de inicio) | `cpt.season_leads` (2-4) |
| `-a`, `--anios` | `1995-2015` (periodo del hindcast) | el de los centros (1993-2016) |
| `-o`, `--salida` | `otra/carpeta` | `data/cpt` |
| `-f`, `--force` | | usa la caché de `data/raw/` |

Ejemplos:

```
bash kairos.sh -m ncep -v precipitation          # un modelo, una variable
bash kairos.sh --region 5,-75,-2,-70             # cuadro propio
bash kairos.sh --inicio 2026-10 --temporada 2-4  # inicio y temporada concretos
```

Con más de un modelo se escribe también el promedio (peso igual por modelo). Con un cuadro
propio, las descargas se guardan aparte en `data/raw/` (no se mezclan con las de otra región).
Una opción inválida se rechaza antes de descargar nada; si algo falla durante la corrida, el
comando termina con `ERROR: <motivo>` y código de salida 1 sin escribir archivos CPT a medias.

El registro de cada corrida guarda el comando exacto, la elección del inicio, la temporada, la
región, los años del hindcast, el Y usado y los miembros de cada modelo.

### Pronóstico más reciente

Se busca desde el mes actual; solo se retrocede un mes si el CDS responde que ese pronóstico
aún no está publicado. Cualquier otro error (credenciales, licencia, servidor, red) detiene la
corrida con su mensaje original. Con `--inicio` se pide un mes concreto y no se retrocede.

### Valor mensual

Cuando un modelo tiene varios valores para un mes (varios días e inicializaciones de un
ensamble *lagged*, o varios miembros), el valor mensual es su promedio simple (decisión del
cliente). La temporada es el promedio de los valores mensuales de sus leads.

### Predictando (Y)

Lo entrega el IDEAM ya en formato CPT de estaciones; su ruta está en `config/params.yaml`
(`predictand`). Antes de escribir X y Z se verifica: formato, longitudes, que haya estaciones
dentro de la región, que cubra todos los meses de la temporada del hindcast, unidades, que no
venga acumulado y que no supere el 10 % de faltantes. Para verificarlo solo:

```
python -m ideam_seasonal.verify --predictando 11   # 11 = mes de inicio del pronóstico
```

Temperatura todavía no tiene Y: se escriben solo X y Z y el registro lo indica.

### Control de calidad de la Fase 2

```
bash kairos.sh qc            # usa la caché de data/raw/
bash kairos.sh qc --force    # vuelve a descargar todo
```

Sobre la caja de prueba y el mes de `params.yaml`: descarga NCEP, DWD, BOM y CHIRPS,
verifica longitudes, unidades y acumulación de cada GRIB, homogeneiza
(`data/processed/homogeneizado_precipitation_m05_box.nc`, un grupo por fuente; en cada modelo
`pr` guarda todos los miembros y `pr_mensual` el valor mensual) y escribe
`docs/informe_control_calidad.md`. El archivo homogeneizado se lee con
`xr.open_dataset(ruta, group="ncep")`; `lead` 1 es el mes de inicialización.

`run.sh` y `cpt.sh` siguen funcionando por compatibilidad: equivalen a `kairos.sh qc` y
`kairos.sh`.

### Tests

```
python -m pytest
```

## Estructura

```
kairos.sh                 comando único (archivos CPT; `qc` para el control de calidad)
run.sh, cpt.sh            compatibilidad: llaman a kairos.sh
config/params.yaml        región, variables, años, centros, temporada, predictando
src/ideam_seasonal/
  kairos.py               opciones del comando único
  ingest.py               descarga (cdsapi y CHIRPS) y pronóstico más reciente
  common.py               configuración, variables, regiones, nombres de archivo, lectura
  verify.py               chequeos de GRIB y del Y, informe
  homogenize.py           reducción de ensambles, valor mensual y dataset homogeneizado
  cpt.py                  escritura en formato CPT y registro de corrida
tests/
docs/                     predictando (Y) del IDEAM
data/                     no se versiona (raw/ descargas, processed/, cpt/)
```
