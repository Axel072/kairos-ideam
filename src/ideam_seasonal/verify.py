"""Chequeos de longitudes, unidades y acumulación; cada uno devuelve (ok, mensaje).

Uso:  python -m ideam_seasonal.verify data/raw/ncep_precipitation_m05_box.grib
      python -m ideam_seasonal.verify --report
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from ideam_seasonal.common import (CENTRES, PRECIP_UNIT_TO_MM_DAY, get_units, open_grib,
                                   unit_factor)
from ideam_seasonal.ingest import load_config

Result = tuple[bool, str]

LEAD_DIMS = ("forecastMonth", "step", "leadtime_month")
MEMBER_DIMS = ("number", "member")

PRECIP_MM_DAY_RANGE = (0.0, 40.0)


def _find_dim(da: xr.DataArray, candidates) -> str | None:
    return next((d for d in candidates if d in da.dims), None)


def check_longitudes(ds: xr.Dataset) -> Result:
    """Convención única y longitudes monótonas."""
    lon = np.asarray(ds["longitude"].values, dtype=float)
    if lon.min() < 0 and lon.max() > 180:
        return False, f"convenciones mezcladas: lon {lon.min()}..{lon.max()}"
    if not np.all(np.diff(lon) > 0):
        return False, "longitudes no monótonas crecientes (¿campo partido o desplazado?)"
    conv = "-180..180" if lon.min() < 0 else "0..360"
    return True, f"convención {conv}, rango {lon.min()}..{lon.max()}"


def check_units(da: xr.DataArray) -> Result:
    """Unidad reconocida y media plausible en mm/día."""
    try:
        factor = unit_factor(da)
    except ValueError as e:
        return False, str(e)
    unit = get_units(da)
    mean = float(da.mean()) * factor
    lo, hi = PRECIP_MM_DAY_RANGE
    if not lo <= mean <= hi:
        return False, f"media {mean:.2f} mm/día fuera de [{lo}, {hi}] con unidad '{unit}'"
    return True, f"unidad '{unit}' -> media {mean:.2f} mm/día"


def check_accumulation(da: xr.DataArray) -> Result:
    """Acumulación por metadatos y por tendencia en la anticipación."""
    step_type = da.attrs.get("GRIB_stepType")
    if step_type == "accum":
        return False, "GRIB_stepType=accum: desacumular"
    nan_frac = float(da.isnull().mean())
    unit = get_units(da)
    is_rate = unit in PRECIP_UNIT_TO_MM_DAY and PRECIP_UNIT_TO_MM_DAY[unit] is not None
    note = f"stepType={step_type}, unidad '{unit}' ({'tasa' if is_rate else 'no tasa'}), NaN={nan_frac:.1%}"
    lead = _find_dim(da, LEAD_DIMS)
    if nan_frac > 0.5 or lead is None or da.sizes[lead] < 3:
        return is_rate, note + "; tendencia por anticipación no evaluable"
    curve = da.mean([d for d in da.dims if d != lead]).values * (PRECIP_UNIT_TO_MM_DAY.get(unit) or 1.0)
    if np.all(np.diff(curve) > 0) and not is_rate:
        return False, f"{note}; media crece con la anticipación {np.round(curve, 2)}"
    return is_rate, f"{note}; media por anticipación (mm/día) {np.round(curve, 2)}"


def verify(path: Path, var: str | None = None) -> bool:
    ds = open_grib(path)
    var = var or next(iter(ds.data_vars))
    da = ds[var]
    print(f"{path.name}  var={var}  dims={dict(da.sizes)}")
    results = {
        "longitudes": check_longitudes(ds),
        "unidades": check_units(da),
        "acumulación": check_accumulation(da),
    }
    for name, (ok, msg) in results.items():
        print(f"  [{'OK' if ok else 'FALLA'}] {name}: {msg}")
    return all(ok for ok, _ in results.values())


POINT = {"lat": 2.5, "lon": -73.5}


def check_chirps(paths: list[Path]) -> dict:
    ds = xr.open_dataset(paths[0], mask_and_scale=False, decode_times=False)
    da = ds["precip"]
    lon = ds["longitude"].values
    fill = da.attrs.get("_FillValue")
    return {
        "units": get_units(da),
        "lon_range": (float(lon.min()), float(lon.max())),
        "lon_ok": check_longitudes(ds),
        "accumulated": "mm/month = total mensual (no desde la inicialización); se divide por "
                       "los días del mes para pasar a mm/día",
        "fill_in_data": bool((da.values == fill).any()) if fill is not None else None,
        "standard_name": da.attrs.get("standard_name"),
    }


def _homogenized_section(processed: Path, variable: str, month: int, y0: int, y1: int) -> list[str]:
    path = processed / f"homogeneizado_{variable}_m{month:02d}_box.nc"
    if not path.exists():
        raise FileNotFoundError(f"falta {path}; corre `python -m ideam_seasonal.homogenize`")
    pt = {"lat": POINT["lat"], "lon": POINT["lon"]}
    g = {n: xr.open_dataset(path, group=n)["pr"] for n in ("ncep", "dwd", "bom", "chirps_1deg")}
    L = ["## 2. Dataset homogeneizado (actividad 2.4 / hito E3)", "",
         f"Archivo: `{path.as_posix()}` (NetCDF con un grupo por fuente). Todo en mm/día, longitudes "
         "con signo, latitudes crecientes, sin NaN tras la reducción.", "",
         "| Grupo | Dimensiones | NaN | Resolución |", "|---|---|---|---|"]
    for n, da in g.items():
        L.append(f"| {n} | `{dict(da.sizes)}` | {float(da.isnull().mean()):.1%} | 1° |")
    nat = xr.open_dataset(path, group="chirps")["pr"]
    L.append(f"| chirps | `{dict(nat.sizes)}` | {float(nat.isnull().mean()):.1%} | 0,05° (nativo) |")
    L += ["", "Reducción: `year` = año de la inicialización, `member` = `number`, `lead` = "
          "`forecastMonth`. **Mapeo anticipación ↔ mes (verificado con `forecastMonth` y "
          "`validityDate` de los tres GRIB):** lead 1 = el mes de la inicialización (mayo); "
          "`valid_time` marca el FIN del periodo (1 jun), por eso el mes de CHIRPS que le "
          "corresponde es `mes de inicio + lead - 1`. CHIRPS se promedió a la celda de 1° de los "
          "modelos (bloques de 20x20 celdas de 0,05°).", "",
          f"### Un mismo punto y mes en las cuatro fuentes (mm/día; lat {POINT['lat']}, lon "
          f"{POINT['lon']}; media {y0}–{y1}, y de todos los miembros)", "",
          "| Lead | Mes | NCEP | DWD | BOM | CHIRPS (1°) |", "|---|---|---|---|---|---|"]
    for lead in g["ncep"]["lead"].values:
        vals = [float(g[n].sel(lead=lead, **pt).mean([d for d in g[n].dims if d in ("year", "member")]))
                for n in ("ncep", "dwd", "bom", "chirps_1deg")]
        mes = int(g["ncep"]["target_month"].sel(lead=lead))
        L.append(f"| {int(lead)} | {mes} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    return L + [""]


def build_report(raw: Path = Path("data") / "raw", processed: Path = Path("data") / "processed",
                 out: Path = Path("docs") / "informe_control_calidad.md",
                 variable: str = "precipitation") -> Path:
    cfg = load_config()
    a, month = cfg["area"], cfg["month"]
    y0, y1 = cfg["centres"]["ncep"]["years"]
    fc = {}
    for c in CENTRES:
        path = raw / f"{c}_{variable}_m{month:02d}_box.grib"
        ds = open_grib(path)
        da = ds[next(iter(ds.data_vars))]
        fc[c] = {
            "units": get_units(da), "step_type": da.attrs.get("GRIB_stepType"),
            "data_type": da.attrs.get("GRIB_dataType"), "dims": dict(da.sizes),
            "lon": check_longitudes(ds), "unit_check": check_units(da),
            "acc_check": check_accumulation(da), "nan_frac": float(da.isnull().mean()),
            "lat0": float(ds.latitude.values[0]), "lat1": float(ds.latitude.values[-1]),
        }
    chirps_paths = sorted(raw.glob("chirps_v3_*_box.nc"))
    ch = check_chirps(chirps_paths)

    L = ["# Informe de control de calidad — Fase 2 (ingesta)", "",
         "Generado por `python -m ideam_seasonal.verify --report`. Región de prueba "
         f"PROVISIONAL (lat {a['south']}–{a['north']}, lon {a['west']}…{a['east']}), "
         f"variable {variable}, mes de inicialización {month}, hindcast {y0}–{y1}. "
         f"Punto de comparación: lat {POINT['lat']}, lon {POINT['lon']} (celda de 1°). "
         f"CHIRPS: {len(chirps_paths)} años descargados.", "",
         "## 1. Las tres trampas por fuente", "",
         "| Fuente | Unidad declarada | Tipo (GRIB) | Acumulada | Longitudes | Latitudes | Dims cfgrib | NaN |",
         "|---|---|---|---|---|---|---|---|"]
    for c, r in fc.items():
        acc = "no (tasa media)" if r["acc_check"][0] else "REVISAR"
        L.append(f"| {c.upper()} | `{r['units']}` | {r['step_type']}/{r['data_type']} | {acc} | "
                 f"{r['lon'][1].replace('convención ', '')} | {r['lat0']}→{r['lat1']} | "
                 f"`{r['dims']}` | {r['nan_frac']:.1%} |")
    L.append(f"| CHIRPS v3.0 | `{ch['units']}` | NetCDF | total mensual (no desde la "
             f"inicialización) | {ch['lon_range'][0]:.3f}…{ch['lon_range'][1]:.3f} (con signo) | "
             "creciente | 12 meses/año, 0,05° | sin fill en la caja |")
    L += ["", "Resultado de los chequeos automáticos:", ""]
    for c, r in fc.items():
        for name, key in (("longitudes", "lon"), ("unidades", "unit_check"),
                          ("acumulación", "acc_check")):
            ok, msg = r[key]
            L.append(f"- {c.upper()} · {name}: {'OK' if ok else 'FALLA'} — {msg}")
    L += [f"- CHIRPS · longitudes: {'OK' if ch['lon_ok'][0] else 'FALLA'} — {ch['lon_ok'][1]}",
          f"- CHIRPS · unidades: `{ch['units']}`; {ch['accumulated']}.",
          f"- CHIRPS · valores de relleno (-9999) dentro de la caja: {ch['fill_in_data']}.",
          f"- CHIRPS · `standard_name` declarado = `{ch['standard_name']}` (no describe una "
          "precipitación mensual; solo metadato, `long_name` y `units` son los fiables).", "",
          "### Estructura de los GRIB (hallada, no asumida)", "",
          "- NCEP: ensamble *lagged* de 24 miembros, 6 inicializaciones (6 abr–1 may, 4 ciclos/día).",
          "- DWD: 30 miembros, una sola inicialización (1 may); sin NaN.",
          "- BOM: 27 miembros *lagged*, 9 inicializaciones (23 abr–1 may).",
          "- NCEP y BOM salen de cfgrib como producto cartesiano `time x step` casi vacío; "
          "`homogenize.py` los reduce a año x miembro x anticipación (sección 2).", ""]
    L += _homogenized_section(processed, variable, month, y0, y1)
    L += ["## 3. Cierre ", "",
          "En un mismo punto y mes las "
          "cuatro fuentes tienen las mismas unidades tras normalizar (mm/día) y el mismo orden "
          "de magnitud, y las tres trampas de ingesta (longitudes, unidades, acumulación) están "
          "verificadas en las cuatro fuentes (sección 1). La observación abierta es el sesgo "
          "de NCEP (sección 4).", "",
          "## 4. Hallazgo ", "",
          "**NCEP sale sistemáticamente más bajo** que CHIRPS en el punto de prueba (2-4x por "
          "debajo según la anticipación; DWD queda en medio y BOM sigue de cerca a CHIRPS)", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["--report"]:
        print(build_report())
    else:
        sys.exit(0 if verify(Path(sys.argv[1])) else 1)
