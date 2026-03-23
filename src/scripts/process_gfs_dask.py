"""
process_gfs_dask.py
-------------------
Opens multiple GFS forecast steps lazily using xarray and Dask,
computes a forecast mean, and processes multi-variable data.
"""

from pathlib import Path
import xarray as xr
import dask
import matplotlib.pyplot as plt
import numpy as np

run_dir = Path("data/gfs_20260322_00z")
print(f"Data directory: {run_dir}")

# 1. Open a single file (Analysis step f000)
single_file = run_dir / "gfs.t00z.pgrb2.0p25.f000.grb2"
print(f"\n--- 1. Opening single file: {single_file.name} ---")
ds_single = xr.open_dataset(
    single_file,
    engine="cfgrib",
    filter_by_keys={
        "typeOfLevel": "isobaricInhPa",
        "shortName": "t",
    }
)
print("Single file dataset:")
print(ds_single)

# 2. Open multiple forecast steps lazily with Dask (mfdataset)
print("\n--- 2. Opening multiple forecast steps (parallel) ---")
files = sorted(run_dir.glob("*.grb2"))
print(f"Found {len(files)} files: {[f.name for f in files]}")

ds_multi = xr.open_mfdataset(
    files,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    parallel=True,
    filter_by_keys={
        "typeOfLevel": "isobaricInhPa",
        "shortName": "t",
    },
    backend_kwargs={"errors": "ignore"},
)
print("Multi-file dataset (Lazy computation):")
print(ds_multi)

# 3. Compute forecast mean (500 hPa Temperature)
print("\n--- 3. Computing Mean 500 hPa Temperature ---")
t500 = ds_multi["t"].sel(isobaricInhPa=500)
t500_c = t500 - 273.15  # Kelvin to Celsius
t500_mean = t500_c.mean(dim="step")

print("Triggering Dask computation...")
t500_mean_computed = t500_mean.compute()
print(f"Computed shape: {t500_mean_computed.shape}")
print(f"Min temp over region: {float(t500_mean_computed.min()):.1f} °C")
print(f"Max temp over region: {float(t500_mean_computed.max()):.1f} °C")

# 4. Multi-variable processing pipeline
print("\n--- 4. Multi-variable processing (850-500 hPa thickness) ---")
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

z_geo = open_run(run_dir, "gh")  # Geopotential height [m]
z500 = z_geo.sel(isobaricInhPa=500)
z850 = z_geo.sel(isobaricInhPa=850)
thickness = (z500 - z850).compute()

print(f"500-850 hPa average thickness: {float(thickness.mean()):.0f} m")
print("\nProcessing complete!")
