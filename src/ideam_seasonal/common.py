"""Constantes y funciones compartidas: configuración, variables, regiones, nombres de archivo."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

CENTRES = ("ncep", "dwd", "bom")

# Raíz del repositorio (src/ideam_seasonal/common.py -> ../../); no depende del directorio
# desde el que se ejecute.
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "params.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Variable:
    """Una variable y cómo pasar cada unidad declarada a la unidad estándar.

    `conversions[unidad] = (escala, desplazamiento)`: estándar = valor * escala + desplazamiento.
    `None` marca una unidad reconocida pero acumulada (hay que desacumular antes).
    """
    key: str
    short: str
    units: str
    long_name: str
    plausible: tuple[float, float]  # rango plausible de la MEDIA, en unidades estándar
    conversions: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    accumulable: bool = False


VARIABLES = {
    "precipitation": Variable(
        key="precipitation", short="pr", units="mm/day",
        long_name="precipitación, tasa media mensual",
        plausible=(0.0, 40.0),
        conversions={
            "kg m**-2 s**-1": (86400.0, 0.0),
            "m s**-1": (86400.0 * 1000.0, 0.0),
            "m": None,
            "mm/day": (1.0, 0.0),
        },
        accumulable=True),
    "temperature": Variable(
        key="temperature", short="tas", units="Celsius",
        long_name="temperatura media del aire a 2 m",
        plausible=(-60.0, 45.0),
        conversions={"K": (1.0, -273.15), "C": (1.0, 0.0), "degC": (1.0, 0.0)}),
}

# Vistas por variable que usa verify.py (derivadas del registro, no se editan aparte).
PRECIP_UNIT_TO_MM_DAY = {u: (c[0] if c else None)
                         for u, c in VARIABLES["precipitation"].conversions.items()}
TEMP_UNIT_TO_C = {u: c[1] for u, c in VARIABLES["temperature"].conversions.items()}


def get_units(da: xr.DataArray) -> str | None:
    return da.attrs.get("units") or da.attrs.get("GRIB_units")


def variable_of(da: xr.DataArray) -> Variable:
    """Variable según la unidad declarada; ValueError si falta o no está en el registro."""
    unit = get_units(da)
    if unit is None:
        raise ValueError("sin atributo units")
    for v in VARIABLES.values():
        if unit in v.conversions:
            return v
    raise ValueError(f"unidad '{unit}' no reconocida; agregarla tras confirmarla")


def conversion(da: xr.DataArray) -> tuple[float, float]:
    """(escala, desplazamiento) a la unidad estándar; ValueError si es desconocida o acumulada."""
    v = variable_of(da)
    unit = get_units(da)
    conv = v.conversions[unit]
    if conv is None:
        raise ValueError(f"unidad '{unit}' es acumulada: desacumular antes de comparar")
    return conv


def apply_conversion(values, conv: tuple[float, float]):
    scale, offset = conv
    if scale != 1.0:
        values = values * scale
    if offset != 0.0:
        values = values + offset
    return values


def to_standard(da: xr.DataArray, values):
    return apply_conversion(values, conversion(da))


def unit_factor(da: xr.DataArray) -> float:
    """Factor a mm/día de una unidad de precipitación; ValueError si falta, no se reconoce o es acumulada."""
    if variable_of(da).key != "precipitation":
        raise ValueError(f"unidad '{get_units(da)}' no es de precipitación")
    return conversion(da)[0]


def is_temperature(da: xr.DataArray) -> bool:
    return get_units(da) in VARIABLES["temperature"].conversions


# ---------------------------------------------------------------------------------------------
# Regiones y nombres de archivo
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    """`name` va en los nombres de archivo; `area` es None para global."""
    name: str
    area: dict | None

    def bounds(self) -> str:
        if self.area is None:
            return "global"
        a = self.area
        return f"N {a['north']}, W {a['west']}, S {a['south']}, E {a['east']}"


def _num(x: float) -> str:
    return f"{x:g}"


def parse_box(text: str) -> dict:
    """'N,W,S,E' (grados, longitudes con signo) -> dict; ValueError si no es un cuadro válido."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"cuadro '{text}': se esperan 4 valores N,W,S,E")
    try:
        n, w, s, e = (float(p) for p in parts)
    except ValueError:
        raise ValueError(f"cuadro '{text}': valores no numéricos") from None
    if not (-90 <= s < n <= 90):
        raise ValueError(f"cuadro '{text}': se requiere -90 <= sur < norte <= 90")
    if not (-180 <= w < e <= 180):
        raise ValueError(f"cuadro '{text}': se requiere -180 <= oeste < este <= 180 "
                         "(longitudes con signo, sin cruzar el antimeridiano)")
    return {"north": n, "west": w, "south": s, "east": e}


def resolve_region(cfg: dict, spec: str | None = None, default: str = "prueba") -> Region:
    """Región por nombre ('prueba', 'global' o una de `cpt.regions`) o por cuadro 'N,W,S,E'."""
    spec = spec or default
    if spec == "global":
        return Region("global", None)
    if spec == "prueba":
        return Region("box", dict(cfg["area"]))
    named = cfg.get("cpt", {}).get("regions", {})
    if spec in named:
        return Region(spec, dict(named[spec]))
    if "," in spec:
        a = parse_box(spec)
        name = f"N{_num(a['north'])}_W{_num(a['west'])}_S{_num(a['south'])}_E{_num(a['east'])}"
        return Region(name, a)
    raise ValueError(f"región '{spec}' desconocida: usa prueba, global, "
                     f"{', '.join(named) or '(ninguna con nombre)'} o un cuadro N,W,S,E")


def raw_path(out_dir: Path, centre: str, variable: str, month: int, region: Region,
             forecast_year: int | None = None) -> Path:
    """GRIB crudo del CDS. `forecast_year` -> pronóstico de ese año; si no, hindcast."""
    tag = f"f{forecast_year}{month:02d}" if forecast_year else f"m{month:02d}"
    return out_dir / f"{centre}_{variable}_{tag}_{region.name}.grib"


def chirps_path(out_dir: Path, year: int | str, region: Region) -> Path:
    """CHIRPS recortado de un año; `year="*"` da el patrón para buscar todos los años."""
    return out_dir / f"chirps_v3_{year}_{region.name}.nc"


def processed_path(out_dir: Path, variable: str, month: int, region: Region) -> Path:
    return out_dir / f"homogeneizado_{variable}_m{month:02d}_{region.name}.nc"


# ---------------------------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------------------------

def open_grib(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def open_chirps(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path, mask_and_scale=False, decode_times=False)
    ds["time"] = pd.to_datetime(ds["time"].values, unit="D", origin="1980-01-01")
    return ds


def predictand_path(cfg: dict, variable: str) -> Path | None:
    """Ruta del Y configurado para la variable (relativa a la raíz del repo), o None."""
    rel = (cfg.get("predictand") or {}).get(variable)
    return ROOT / rel if rel else None


def _cpt_tags(line: str) -> dict[str, str]:
    tags = {}
    for item in line.strip().split(","):
        item = item.strip()
        if item.startswith("cpt:") and "=" in item:
            k, v = item[4:].split("=", 1)
            tags[k.strip()] = v.strip()
    return tags


def read_cpt_grid(path: Path) -> dict[str, xr.DataArray]:
    """Archivo CPT de rejilla -> {cpt:T: DataArray (lat, lon)}, latitudes de norte a sur.

    Lee tanto los archivos que escribe `cpt.write_cpt` como los que escribe CPT (donde las
    etiquetas de un bloque se heredan del anterior si no se repiten).
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out, tags, i = {}, {}, 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("cpt:") and "cpt:T=" in ln:
            tags.update(_cpt_tags(ln))
            nrow, ncol = int(tags["nrow"]), int(tags["ncol"])
            lon = np.array([float(v) for v in lines[i + 1].split("\t")[1:ncol + 1]])
            rows = [r.split("\t") for r in lines[i + 2:i + 2 + nrow]]
            lat = np.array([float(r[0]) for r in rows])
            vals = np.array([[float(v) for v in r[1:ncol + 1]] for r in rows])
            if "missing" in tags:
                vals[vals == float(tags["missing"])] = np.nan
            da = xr.DataArray(vals, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
                              attrs={"units": tags.get("units")})
            out[tags["T"]] = da.sortby("lat", ascending=False)
            i += 2 + nrow
        else:
            i += 1
    return out


def open_cpt_stations(path: Path) -> xr.DataArray:
    """Lee un archivo CPT de estaciones (un campo, serie mensual) -> DataArray (time, station).

    Faltantes (`cpt:missing`) -> NaN. Falla con ValueError si el formato no es el esperado:
    no es de estaciones, tiene más de un campo, fechas no mensuales o filas/columnas que no
    coinciden con `cpt:nrow`/`cpt:ncol`.
    """
    lines = [ln.rstrip("\r\n") for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    starts = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("cpt:field")]
    if len(starts) != 1:
        raise ValueError(f"{Path(path).name}: se espera un solo bloque cpt:field (hay {len(starts)})")
    i = starts[0]
    tags = _cpt_tags(lines[i])
    if tags.get("col") != "station" or tags.get("row") != "T":
        raise ValueError(f"{Path(path).name}: no es un archivo de estaciones "
                         f"(cpt:col={tags.get('col')}, cpt:row={tags.get('row')})")
    nrow, ncol = int(tags["nrow"]), int(tags["ncol"])

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.split("\t")][1:][:ncol]

    ids = cells(lines[i + 1])
    rows = {ln.split("\t", 1)[0].strip(): ln for ln in lines[i + 2:i + 4]}
    if set(rows) != {"cpt:X", "cpt:Y"}:
        raise ValueError(f"{Path(path).name}: faltan las filas cpt:X / cpt:Y de coordenadas")
    lon = np.array(cells(rows["cpt:X"]), dtype=float)
    lat = np.array(cells(rows["cpt:Y"]), dtype=float)
    data = [ln for ln in lines[i + 4:] if ln.strip()]
    if len(data) != nrow or len(ids) != ncol or lon.size != ncol or lat.size != ncol:
        raise ValueError(f"{Path(path).name}: tamaño distinto al declarado (filas {len(data)} vs "
                         f"cpt:nrow={nrow}; columnas {len(ids)} vs cpt:ncol={ncol})")
    dates = [ln.split("\t", 1)[0].strip() for ln in data]
    try:
        time = pd.to_datetime(dates, format="%Y-%m")
    except ValueError:
        raise ValueError(f"{Path(path).name}: fechas no mensuales (YYYY-MM), p. ej. "
                         f"'{dates[0]}'; solo se admiten series mensuales") from None
    values = np.array([cells(ln) for ln in data], dtype=float)
    missing = float(tags["missing"]) if "missing" in tags else None
    if missing is not None:
        values[values == missing] = np.nan
    return xr.DataArray(
        values, dims=("time", "station"),
        coords={"time": time, "station": ids, "lon": ("station", lon), "lat": ("station", lat)},
        attrs={"units": tags.get("units"), "field": tags.get("field"), "missing": missing,
               "source": Path(path).name})
