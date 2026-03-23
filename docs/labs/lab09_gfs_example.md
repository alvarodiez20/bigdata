# GFS Data: Download, Process with Dask, and Visualise

This page provides a fully worked example of the end-to-end workflow you will use in your final project:

1. **Download** GFS GRIB2 files from two public sources (NOMADS filter and AWS S3).
2. **Open lazily** with `cfgrib` + `xarray` backed by `dask`.
3. **Process** a multi-file forecast run using distributed computation.
4. **Visualise** the result over the Iberian Peninsula with `cartopy`.

---

## Prerequisites

Make sure the required dependencies are installed. They are defined under the `lab09` dependency group in `pyproject.toml`.

Run the following command to sync the packages:

```bash
uv sync --group lab09
```

Verify the stack:

```bash
uv run python -c "import cfgrib, s3fs, cartopy; print('OK')"
```

!!! warning "eccodes system library"
    `eccodes` requires the C library to be present. On most systems the pip package bundles it, but if you see `FileNotFoundError` or `Cannot find the ecCodes library`, install the system package first:
    
    **macOS**:
    ```bash
    brew install eccodes
    ```
    
    **Ubuntu/Debian / Windows (WSL)**:
    ```bash
    sudo apt-get install libeccodes-dev
    ```
    
    **Windows (Native)**:
    Native Windows is not officially supported by the `eccodes` pip package. We strongly recommend using **Windows Subsystem for Linux (WSL)** and following the Ubuntu instructions. If you must use native Windows, you will need to use `conda` instead of `uv` to install the C-library bindings:
    ```bash
    conda install -c conda-forge eccodes cfgrib
    ```

---

## Grid Visualisation — GFS vs ECMWF over the Iberian Peninsula

Before downloading any forecast data it is useful to understand how the model grids look in practice. The script below plots the grid nodes of both GFS (0.25°) and ECMWF IFS (0.1°) over the Iberian Peninsula and saves the result to `docs/img/gfs_vs_ecmwf_grid_iberia.png`.

```bash
uv run python src/scripts/plot_gfs_ecmwf_grids.py
```

The script is located at `src/scripts/plot_gfs_ecmwf_grids.py`. It uses `numpy`, `matplotlib`, and `cartopy` — no GRIB data required.

Key points about the script:

- GFS nodes (blue) use a 0.25° regular lat–lon grid — ~2,400 nodes over the Peninsula.
- ECMWF nodes (orange) use a 0.1° regular lat–lon grid — ~14,250 nodes over the same domain (~6× denser in area).
- The ECMWF nodes are subsampled every other point for readability.
- Projection: Lambert Conformal centred at −2.5° E, 40° N to minimise distortion over Spain.

---

## Forecast Error Growth — Lorenz Model

The script below plots RMSE growth curves for Z500, T850, and U10 over a 10-day lead time, comparing ECMWF and GFS using the Lorenz error-growth model. No data download required — the curves are computed analytically from representative verification parameters.

```bash
uv run python src/scripts/plot_lorenz_error_growth.py
```

The script is located at `src/scripts/plot_lorenz_error_growth.py` and saves its output to `docs/img/forecast_error_growth.png`.

---

## Source 1 — NOMADS Filtered Download (recent operational data)

NOAA's **NOMADS** server exposes a HTTP filter interface that lets you request only the variables, pressure levels, and geographic sub-region you need, drastically reducing download size.

Filter URL pattern:

```
https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
  ?file=gfs.tHHz.pgrb2.0p25.fFFF
  &var_TMP=on&var_UGRD=on&var_VGRD=on&var_HGT=on
  &lev_500_mb=on&lev_850_mb=on&lev_2_m_above_ground=on
  &subregion=&leftlon=-15&rightlon=10&toplat=46&bottomlat=34
  &dir=%2Fgfs.YYYYMMDD%2FHH%2Fatmos
```

The download script below fetches six forecast steps (0 to 24 h, every 6 h) for a given run date. Replace `RUN_DATE` and `RUN_HOUR` before running.

```python
"""
download_gfs_nomads.py
----------------------
Downloads a short GFS forecast sequence over the Iberian Peninsula
from the NOAA NOMADS filtered HTTP service.

Usage:
    uv run python src/scripts/download_gfs_nomads.py --date 20240715 --hour 00
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

# ── configuration ────────────────────────────────────────────────────────────
NOMADS_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# Iberian Peninsula bounding box (add a small buffer)
BBOX = dict(leftlon=-15, rightlon=10, toplat=46, bottomlat=34)

VARIABLES = ["TMP", "UGRD", "VGRD", "HGT", "PRMSL", "APCP"]
LEVELS    = ["500_mb", "850_mb", "2_m_above_ground", "mean_sea_level",
             "surface"]

FORECAST_HOURS = [0, 6, 12, 18, 24]    # change to range(0, 121, 6) for 5 days


def build_url(date: str, hour: str, fhour: int) -> str:
    """Return the NOMADS filter URL for a single forecast step."""
    fname = f"gfs.t{hour}z.pgrb2.0p25.f{fhour:03d}"
    params = [
        f"file={fname}",
        *[f"var_{v}=on" for v in VARIABLES],
        *[f"lev_{lv}=on" for lv in LEVELS],
        "subregion=",
        *[f"{k}={v}" for k, v in BBOX.items()],
        f"dir=%2Fgfs.{date}%2F{hour}%2Fatmos",
    ]
    return NOMADS_BASE + "?" + "&".join(params)


def download_file(url: str, dest: Path, retries: int = 3) -> None:
    """Stream-download a URL to *dest*, retrying on transient errors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip (exists) → {dest.name}")
        return
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                        f.write(chunk)
            size_mb = dest.stat().st_size / 1e6
            print(f"  downloaded ({size_mb:.1f} MB) → {dest.name}")
            return
        except requests.RequestException as exc:
            print(f"  attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Run date YYYYMMDD")
    parser.add_argument("--hour", default="00", choices=["00","06","12","18"])
    args = parser.parse_args()

    out_dir = Path("data") / f"gfs_{args.date}_{args.hour}z"
    print(f"Downloading GFS {args.date} {args.hour}Z → {out_dir}/")

    for fh in FORECAST_HOURS:
        url  = build_url(args.date, args.hour, fh)
        dest = out_dir / f"gfs.t{args.hour}z.pgrb2.0p25.f{fh:03d}.grb2"
        download_file(url, dest)

    print(f"\nDone. {len(FORECAST_HOURS)} files in {out_dir}/")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python src/scripts/download_gfs_nomads.py --date 20240715 --hour 00
```

!!! note "NOMADS data retention"
    NOMADS only holds the last **~10 days** of operational data. For historical case studies use the AWS S3 source below.

---

## Source 2 — AWS Open Data (historical archive, no credentials)

NOAA publishes the complete GFS archive on Amazon S3 as part of the **AWS Open Data** programme. Access is free and anonymous.

Bucket: `s3://noaa-gfs-bdp-pds/`
Key pattern: `gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF`

```python
"""
list_gfs_s3.py
--------------
Lists and downloads GFS files from the public AWS S3 bucket.
No AWS account or credentials required.
"""

import s3fs

fs = s3fs.S3FileSystem(anon=True)

# List available run cycles for a given date
date = "20230701"
runs = fs.ls(f"noaa-gfs-bdp-pds/gfs.{date}/")
print("Available run cycles:", [r.split("/")[-1] for r in runs])

# Download a single file (0-h analysis, 0.25° resolution)
src  = f"noaa-gfs-bdp-pds/gfs.{date}/00/atmos/gfs.t00z.pgrb2.0p25.f000"
dest = f"data/gfs_{date}_00z/gfs.t00z.pgrb2.0p25.f000.grb2"

import pathlib
pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
fs.get(src, dest)
print(f"Downloaded → {dest}")
```

For batch downloads of many forecast steps, wrap `fs.get` in a loop or use `fs.get(src_list, dest_dir)` to transfer multiple files in parallel.

---

## Opening GFS Files with cfgrib + xarray + Dask

GRIB2 files contain many independent **messages** (one per variable × level × time). `cfgrib` exposes each consistent group of messages as an `xarray.Dataset`. Use `filter_by_keys` to select only what you need.

### Open a single file

```python
import xarray as xr

ds = xr.open_dataset(
    "data/gfs_20240715_00z/gfs.t00z.pgrb2.0p25.f000.grb2",
    engine="cfgrib",
    filter_by_keys={
        "typeOfLevel": "isobaricInhPa",   # pressure levels
        "shortName": "t",                  # temperature
    }
)
print(ds)
# Dimensions: (isobaricInhPa: 41, latitude: 49, longitude: 101)
# Data variables: t  (isobaricInhPa, latitude, longitude)
```

!!! tip "Discover available variables"
    Use `cfgrib.open_datasets(path)` (without filters) — it returns a list of all
    datasets found in the file, one per consistent set of GRIB messages.
    ```python
    import cfgrib
    datasets = cfgrib.open_datasets("gfs.t00z.pgrb2.0p25.f000.grb2")
    for ds in datasets:
        print(ds)
    ```

### Open multiple forecast steps lazily with Dask

The real power of `xarray` + `dask` is opening an entire forecast run — or multiple days — as a single virtual dataset that fits in RAM because data is only read when explicitly computed.

```python
import glob
from pathlib import Path

import dask
import xarray as xr

# ── 1. Collect all GRIB2 files for this run ───────────────────────────────
run_dir = Path("data/gfs_20240715_00z")
files = sorted(run_dir.glob("*.grb2"))          # f000, f006, f012, f018, f024
print(f"Found {len(files)} files")

# ── 2. Open as a multi-file dataset (lazy, Dask-backed) ───────────────────
#    Each file is one forecast step; xarray concatenates along 'step'.
#    combine="nested" + concat_dim tells xarray the files differ in step.
ds = xr.open_mfdataset(
    files,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    parallel=True,                              # open index files in parallel
    filter_by_keys={
        "typeOfLevel": "isobaricInhPa",
        "shortName": "t",
    },
    backend_kwargs={"errors": "ignore"},
)

print(ds)
# Dimensions: (step: 5, isobaricInhPa: 41, latitude: 49, longitude: 101)
# This printed instantly — no data has been read from disk yet.
```

### Inspect the task graph

```python
t500 = ds["t"].sel(isobaricInhPa=500)   # still lazy
print(t500)                              # shows dask array with shape (5, 49, 101)

# Visualise the Dask task graph (optional)
t500.data.visualize(filename="task_graph.svg", rankdir="LR")
```

### Compute: forecast evolution and mean

```python
import numpy as np

# Convert temperature from Kelvin to Celsius (lazy operation — builds graph)
t500_c = t500 - 273.15

# Mean temperature at 500 hPa over the Iberian Peninsula across all steps
# Still lazy — no computation yet
t500_mean = t500_c.mean(dim="step")

# ── Trigger computation ────────────────────────────────────────────────────
# dask.compute() runs the graph; .compute() on an xarray DataArray is equivalent
t500_mean_computed = t500_mean.compute()   # reads files, executes graph
print(f"Shape: {t500_mean_computed.shape}")    # (49, 101)
print(f"Min: {float(t500_mean_computed.min()):.1f} °C")
print(f"Max: {float(t500_mean_computed.max()):.1f} °C")
```

### Multi-variable processing pipeline

```python
# Open temperature AND geopotential together
def open_run(run_dir: Path, short_name: str) -> xr.DataArray:
    files = sorted(run_dir.glob("*.grb2"))
    ds = xr.open_mfdataset(
        files,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        parallel=True,
        filter_by_keys={"typeOfLevel": "isobaricInhPa", "shortName": short_name},
        backend_kwargs={"errors": "ignore"},
    )
    return ds[short_name]


run_dir = Path("data/gfs_20240715_00z")

temp  = open_run(run_dir, "t")  - 273.15   # °C, lazy
z_geo = open_run(run_dir, "gh")            # geopotential height [m], lazy

# Compute thickness (850–500 hPa layer) for all steps at once
z500 = z_geo.sel(isobaricInhPa=500)
z850 = z_geo.sel(isobaricInhPa=850)
thickness = (z500 - z850).compute()         # single graph execution

print(f"500–850 hPa thickness — mean: {float(thickness.mean()):.0f} m")
```

---

## Visualisation: Iberian Peninsula

### 2-m Temperature at analysis time (f000)

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

# ── Load 2-m temperature from the analysis file ───────────────────────────
ds_sfc = xr.open_dataset(
    "data/gfs_20240715_00z/gfs.t00z.pgrb2.0p25.f000.grb2",
    engine="cfgrib",
    filter_by_keys={
        "typeOfLevel": "heightAboveGround",
        "level": 2,
        "shortName": "2t",
    },
)
t2m = ds_sfc["t2m"] - 273.15   # K → °C

# ── Plot ──────────────────────────────────────────────────────────────────
proj = ccrs.LambertConformal(central_longitude=-4, central_latitude=40)

fig, ax = plt.subplots(figsize=(9, 6),
                        subplot_kw={"projection": proj})

# Geographic extent: Iberian Peninsula + Balearic Islands
ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())

# Natural Earth features
ax.add_feature(cfeature.LAND,       facecolor="white", zorder=0)
ax.add_feature(cfeature.OCEAN,      facecolor="#cce5ff", zorder=0)
ax.add_feature(cfeature.COASTLINE,  linewidth=0.8)
ax.add_feature(cfeature.BORDERS,    linewidth=0.6, linestyle="--")
ax.add_feature(cfeature.RIVERS,     linewidth=0.4, edgecolor="#6baed6")

# Filled contours
cf = ax.contourf(
    t2m.longitude, t2m.latitude, t2m,
    levels=np.arange(10, 42, 2),
    cmap="RdYlBu_r",
    transform=ccrs.PlateCarree(),
    extend="both",
)

# MSLP contours from the same file
ds_mslp = xr.open_dataset(
    "data/gfs_20240715_00z/gfs.t00z.pgrb2.0p25.f000.grb2",
    engine="cfgrib",
    filter_by_keys={"typeOfLevel": "meanSea", "shortName": "prmsl"},
)
mslp = ds_mslp["prmsl"] / 100   # Pa → hPa

cs = ax.contour(
    mslp.longitude, mslp.latitude, mslp,
    levels=np.arange(990, 1030, 4),
    colors="k", linewidths=0.8,
    transform=ccrs.PlateCarree(),
)
ax.clabel(cs, fmt="%d", fontsize=7)

# Colorbar and labels
cbar = fig.colorbar(cf, ax=ax, orientation="vertical",
                    pad=0.02, fraction=0.03)
cbar.set_label("2-m Temperature (°C)", fontsize=10)

gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                   color="gray", alpha=0.5, linestyle="--")
gl.top_labels   = False
gl.right_labels = False

ax.set_title(
    f"GFS — 2-m Temperature and MSLP\n"
    f"Valid: {str(ds_sfc.valid_time.values)[:16]} UTC",
    fontsize=11,
)

plt.tight_layout()
plt.savefig("results/gfs_t2m_iberia.png", dpi=150, bbox_inches="tight")
plt.show()
```

The resulting plot will show a filled temperature colour scale (warm reds over inland Spain, cooler blues along the coast) with black MSLP contours labelled in hPa.

---

### 500 hPa Geopotential Height — Forecast Evolution (all steps)

This plot shows how the 500 hPa pattern evolves across the five forecast steps in a single figure, exploiting the lazy multi-step dataset built earlier.

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Use the lazily opened z_geo from the section above
z500_all = z_geo.sel(isobaricInhPa=500).compute()   # shape: (5, lat, lon)
steps     = z500_all.step.values                      # timedelta64 array

proj = ccrs.LambertConformal(central_longitude=-4, central_latitude=40)
fig, axes = plt.subplots(
    1, len(steps),
    figsize=(4 * len(steps), 4),
    subplot_kw={"projection": proj},
)

levels = np.arange(5400, 5900, 40)   # typical 500 hPa heights in metres

for ax, step, z in zip(axes, steps, z500_all):
    ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle="--")
    ax.add_feature(cfeature.OCEAN,     facecolor="#cce5ff", zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)

    cf = ax.contourf(
        z.longitude, z.latitude, z,
        levels=levels, cmap="viridis",
        transform=ccrs.PlateCarree(), extend="both",
    )
    ax.contour(
        z.longitude, z.latitude, z,
        levels=levels, colors="k", linewidths=0.5,
        transform=ccrs.PlateCarree(),
    )

    hours = int(step / np.timedelta64(1, "h"))
    ax.set_title(f"+{hours:03d} h", fontsize=9)

fig.colorbar(cf, ax=axes, orientation="horizontal",
             pad=0.04, fraction=0.03, label="500 hPa Geopotential Height (m)")

fig.suptitle("GFS — 500 hPa Geopotential Height Forecast\nIberian Peninsula",
             fontsize=12, y=1.02)

plt.tight_layout()
plt.savefig("results/gfs_z500_evolution.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

### Wind Barbs at 850 hPa

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np

# Open u and v components at 850 hPa (analysis step only)
def open_wind(path: str, level: int) -> tuple:
    kw = dict(engine="cfgrib",
              filter_by_keys={"typeOfLevel": "isobaricInhPa", "level": level})
    u = xr.open_dataset(path, **{**kw, "filter_by_keys": {**kw["filter_by_keys"],
                                                            "shortName": "u"}})["u"]
    v = xr.open_dataset(path, **{**kw, "filter_by_keys": {**kw["filter_by_keys"],
                                                            "shortName": "v"}})["v"]
    return u, v


path = "data/gfs_20240715_00z/gfs.t00z.pgrb2.0p25.f000.grb2"
u850, v850 = open_wind(path, 850)

# Speed for background shading
speed = np.sqrt(u850**2 + v850**2)

proj = ccrs.LambertConformal(central_longitude=-4, central_latitude=40)
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={"projection": proj})
ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND,      facecolor="#f0ede0", zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor="#cce5ff", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS,   linewidth=0.6, linestyle="--")

# Background: wind speed
cf = ax.contourf(
    speed.longitude, speed.latitude, speed,
    levels=np.arange(0, 30, 3), cmap="YlOrRd",
    transform=ccrs.PlateCarree(), extend="max",
)

# Wind barbs — subsample to avoid clutter (every 4th point)
skip = 4
ax.barbs(
    u850.longitude.values[::skip],
    u850.latitude.values[::skip],
    u850.values[::skip, ::skip],
    v850.values[::skip, ::skip],
    transform=ccrs.PlateCarree(),
    length=5, linewidth=0.6,
    barbcolor="k", flagcolor="k",
)

cbar = fig.colorbar(cf, ax=ax, orientation="vertical",
                    pad=0.02, fraction=0.03)
cbar.set_label("Wind speed (m s⁻¹)", fontsize=10)

gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                   color="gray", alpha=0.5, linestyle="--")
gl.top_labels   = False
gl.right_labels = False

ax.set_title(
    f"GFS — 850 hPa Wind\n"
    f"Valid: {str(u850.valid_time.values)[:16]} UTC",
    fontsize=11,
)

plt.tight_layout()
plt.savefig("results/gfs_wind850_iberia.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Dask Performance Notes

| Setting | Recommendation |
|---|---|
| **Chunk size** | Target 100–500 MB per chunk. Too-small chunks create scheduler overhead; too-large chunks risk out-of-memory errors. |
| **Scheduler** | For a laptop use the default threaded scheduler. For a cluster or multi-core workstation: `dask.distributed.Client()`. |
| **`parallel=True` in `open_mfdataset`** | Only parallelises opening the index files (fast). Actual data reads are parallelised when `.compute()` is called. |
| **`persist()` vs `compute()`** | On a distributed cluster, `.persist()` keeps the result in distributed RAM for reuse; `.compute()` brings it back to the local process. |
| **Avoid `for` loops over time** | Use `xr.concat`, `xr.open_mfdataset`, or `.rolling()` / `.groupby()` — these build a single graph that dask optimises. |

### Profiling the task graph

```python
from dask.diagnostics import ProgressBar, ResourceProfiler

with ProgressBar(), ResourceProfiler(dt=0.5) as rprof:
    result = t500_mean.compute()

rprof.visualize()   # plots CPU and memory usage over time
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `KeyError: 'cfVarName'` | GRIB message has no CF name mapping | Use `shortName` or `paramId` in `filter_by_keys` instead |
| `ValueError: multiple values for unique key` | Multiple GRIB messages match the filter | Add more keys: `"typeOfLevel"`, `"level"`, `"stepType"` |
| `FileNotFoundError: libeccodes.so` | eccodes C library not found | `brew install eccodes` or `apt-get install libeccodes-dev` |
| `MergeError` in `open_mfdataset` | Files have incompatible coordinates | Add `compat="override"` or `data_vars="minimal"` |
| Memory error during `.compute()` | Chunk too large | Reduce `chunks=` or process one step at a time |
| Blank map / no data plotted | Longitude range mismatch (0–360 vs −180–180) | `ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180).sortby("longitude")` |

---

