import pathlib
from typing import Tuple

import pandas as pd
import xarray as xr


def load_segy_dataset(path: pathlib.Path) -> xr.Dataset:
    ds = xr.open_dataset(
        path,
        dim_byte_fields={"iline": 189, "xline": 193},
        extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
    )
    ds.segysak.scale_coords()

    # Estimate median inline spacing in meters using CDP coordinates
    try:
        dx = ds["cdp_x"].diff("iline")
        dy = ds["cdp_y"].diff("iline")
        spacing = ((dx ** 2 + dy ** 2) ** 0.5)
        inline_spacing_m = float(spacing.median(skipna=True).values)
        ds.attrs["inline_spacing_m"] = inline_spacing_m
    except Exception:
        ds.attrs["inline_spacing_m"] = float("nan")

    # Also record the inline index step (usually 1)
    try:
        inline_step = float(np.median(np.diff(np.unique(ds["iline"].values))))
        ds.attrs["inline_step"] = inline_step
    except Exception:
        ds.attrs["inline_step"] = float("nan")

    return ds


def load_well_locations(path: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(path, sep="\t")
    return df, df["X"], df["Y"], df["Nickname"]


