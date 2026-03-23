"""
plot_gfs_ecmwf_grids.py
-----------------------
Plots GFS 0.25° and ECMWF 0.1° grid nodes over the Iberian Peninsula
and saves the figure to docs/img/gfs_vs_ecmwf_grid_iberia.png.

Usage:
    uv run python src/scripts/plot_gfs_ecmwf_grids.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# ── Domain centred on the Iberian Peninsula ──────────────────────────────────
LON_MIN, LON_MAX = -10.0, 5.0
LAT_MIN, LAT_MAX = 35.0, 44.5

# ── Build GFS 0.25° grid nodes ────────────────────────────────────────────────
gfs_lons = np.arange(LON_MIN, LON_MAX + 0.25, 0.25)
gfs_lats = np.arange(LAT_MIN, LAT_MAX + 0.25, 0.25)
gfs_lon2d, gfs_lat2d = np.meshgrid(gfs_lons, gfs_lats)

# ── Build ECMWF 0.1° grid nodes (subsample every 2 for readability) ───────────
ecmwf_lons = np.arange(LON_MIN, LON_MAX + 0.1, 0.1)
ecmwf_lats = np.arange(LAT_MIN, LAT_MAX + 0.1, 0.1)
ecmwf_lon2d, ecmwf_lat2d = np.meshgrid(ecmwf_lons[::2], ecmwf_lats[::2])

# ── Projection ────────────────────────────────────────────────────────────────
proj = ccrs.LambertConformal(
    central_longitude=-2.5, central_latitude=40.0,
    standard_parallels=(38.0, 42.0),
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": proj})

configs = [
    ("GFS  —  0.25° regular lat–lon",   gfs_lon2d,   gfs_lat2d,   "steelblue",   18, "0.25°"),
    ("ECMWF IFS  —  0.1° regular lat–lon", ecmwf_lon2d, ecmwf_lat2d, "darkorange",  4,  "0.1°"),
]

for ax, (title, lons, lats, color, size, spacing) in zip(axes, configs):
    ax.set_extent(
        [LON_MIN - 0.5, LON_MAX + 0.5, LAT_MIN - 0.5, LAT_MAX + 0.5],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cce5ff",    zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7,          zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle="--", zorder=1)
    ax.add_feature(cfeature.RIVERS,    linewidth=0.3, edgecolor="dodgerblue", zorder=1)

    ax.scatter(
        lons.ravel(), lats.ravel(),
        s=size, color=color, alpha=0.6,
        transform=ccrs.PlateCarree(), zorder=2,
        label=f"Grid node  (Δ = {spacing})",
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color="grey", alpha=0.5, linestyle=":")
    gl.top_labels   = False
    gl.right_labels = False

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.legend(loc="lower right", fontsize=8)

fig.suptitle(
    "Horizontal grid nodes over the Iberian Peninsula\n"
    "GFS 0.25° vs ECMWF IFS 0.1°  (Lambert Conformal projection)",
    fontsize=12, y=1.02,
)
plt.tight_layout()

out = Path("docs/img/gfs_vs_ecmwf_grid_iberia.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
