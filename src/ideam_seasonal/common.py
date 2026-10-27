from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

CENTRES = ("ncep", "dwd", "bom")

PRECIP_UNIT_TO_MM_DAY = {
    "kg m**-2 s**-1": 86400.0,
    "m s**-1": 86400.0 * 1000.0,
    "m": None,
    "mm/day": 1.0,
}


def get_units(da: xr.DataArray) -> str | None:
    return da.attrs.get("units") or da.attrs.get("GRIB_units")


def unit_factor(da: xr.DataArray) -> float:
    """Factor a mm/día de la unidad declarada; ValueError si falta, no se reconoce o es acumulada."""
    unit = get_units(da)
    if unit is None:
        raise ValueError("sin atributo units")
    if unit not in PRECIP_UNIT_TO_MM_DAY:
        raise ValueError(f"unidad '{unit}' no reconocida; agregarla tras confirmarla")
    factor = PRECIP_UNIT_TO_MM_DAY[unit]
    if factor is None:
        raise ValueError(f"unidad '{unit}' es acumulada: desacumular antes de comparar")
    return factor


def open_grib(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def open_chirps(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path, mask_and_scale=False, decode_times=False)
    ds["time"] = pd.to_datetime(ds["time"].values, unit="D", origin="1980-01-01")
    return ds
