"""
visualize_gfs.py
----------------
Generates the three final GFS visualizations over the Iberian Peninsula:
1. 2-m Temperature and MSLP (Analysis)
2. 500 hPa Geopotential Height Forecast Evolution
3. 850 hPa Wind Barbs (Analysis)
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

run_dir = Path("data/gfs_20260322_00z")
f000_path = run_dir / "gfs.t00z.pgrb2.0p25.f000.grb2"
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------------------
# 1. 2-m Temperature and MSLP
# -------------------------------------------------------------------------
print("Generating Plot 1: 2-m Temperature and MSLP...")

ds_sfc = xr.open_dataset(
    f000_path,
    engine="cfgrib",
    filter_by_keys={
        "typeOfLevel": "heightAboveGround",
        "level": 2,
        "shortName": "2t",
    },
)
t2m = ds_sfc["t2m"] - 273.15   # K → °C

proj = ccrs.LambertConformal(central_longitude=-4, central_latitude=40)
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={"projection": proj})
ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND,       facecolor="white", zorder=0)
ax.add_feature(cfeature.OCEAN,      facecolor="#cce5ff", zorder=0)
ax.add_feature(cfeature.COASTLINE,  linewidth=0.8)
ax.add_feature(cfeature.BORDERS,    linewidth=0.6, linestyle="--")
ax.add_feature(cfeature.RIVERS,     linewidth=0.4, edgecolor="#6baed6")

cf = ax.contourf(
    t2m.longitude, t2m.latitude, t2m,
    levels=np.arange(10, 42, 2),
    cmap="RdYlBu_r",
    transform=ccrs.PlateCarree(),
    extend="both",
)

ds_mslp = xr.open_dataset(
    f000_path,
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

cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, fraction=0.03)
cbar.set_label("2-m Temperature (°C)", fontsize=10)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
gl.top_labels   = False
gl.right_labels = False

ax.set_title(
    f"GFS — 2-m Temperature and MSLP\nValid: {str(ds_sfc.valid_time.values)[:16]} UTC",
    fontsize=11,
)

plt.tight_layout()
out_path_1 = results_dir / "gfs_t2m_iberia.png"
plt.savefig(out_path_1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> Saved: {out_path_1}")


# -------------------------------------------------------------------------
# 2. 500 hPa Geopotential Height — Forecast Evolution
# -------------------------------------------------------------------------
print("Generating Plot 2: 500 hPa Geopotential Height Evolution...")

def open_run(r_dir: Path, short_name: str) -> xr.DataArray:
    files = sorted(r_dir.glob("*.grb2"))
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

z_geo = open_run(run_dir, "gh")
z500_all = z_geo.sel(isobaricInhPa=500).compute()   # shape: (5, lat, lon)
steps     = z500_all.step.values

fig, axes = plt.subplots(
    1, len(steps),
    figsize=(4 * len(steps), 4),
    subplot_kw={"projection": proj},
)

levels = np.arange(5400, 5900, 40)

for ax, step, z in zip(axes, steps, z500_all):
    ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle="--")
    ax.add_feature(cfeature.OCEAN,     facecolor="#cce5ff", zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)

    cf2 = ax.contourf(
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

fig.colorbar(cf2, ax=axes, orientation="horizontal", pad=0.04, fraction=0.03, label="500 hPa Geopotential Height (m)")
fig.suptitle("GFS — 500 hPa Geopotential Height Forecast\nIberian Peninsula", fontsize=12, y=1.02)

out_path_2 = results_dir / "gfs_z500_evolution.png"
plt.savefig(out_path_2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> Saved: {out_path_2}")


# -------------------------------------------------------------------------
# 3. Wind Barbs at 850 hPa
# -------------------------------------------------------------------------
print("Generating Plot 3: 850 hPa Wind Barbs...")

def open_wind(path: str|Path, level: int) -> tuple:
    kw = dict(engine="cfgrib", filter_by_keys={"typeOfLevel": "isobaricInhPa", "level": level})
    u = xr.open_dataset(path, **{**kw, "filter_by_keys": {**kw["filter_by_keys"], "shortName": "u"}})["u"]
    v = xr.open_dataset(path, **{**kw, "filter_by_keys": {**kw["filter_by_keys"], "shortName": "v"}})["v"]
    return u, v

u850, v850 = open_wind(f000_path, 850)
speed = np.sqrt(u850**2 + v850**2)

fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={"projection": proj})
ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND,      facecolor="#f0ede0", zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor="#cce5ff", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS,   linewidth=0.6, linestyle="--")

cf3 = ax.contourf(
    speed.longitude, speed.latitude, speed,
    levels=np.arange(0, 30, 3), cmap="YlOrRd",
    transform=ccrs.PlateCarree(), extend="max",
)

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

cbar = fig.colorbar(cf3, ax=ax, orientation="vertical", pad=0.02, fraction=0.03)
cbar.set_label("Wind speed (m s⁻¹)", fontsize=10)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
gl.top_labels   = False
gl.right_labels = False

ax.set_title(
    f"GFS — 850 hPa Wind\nValid: {str(u850.valid_time.values)[:16]} UTC",
    fontsize=11,
)

plt.tight_layout()
out_path_3 = results_dir / "gfs_wind850_iberia.png"
plt.savefig(out_path_3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> Saved: {out_path_3}")

print("\nAll visualizations complete!")
