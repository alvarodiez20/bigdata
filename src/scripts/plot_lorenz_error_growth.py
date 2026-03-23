"""
plot_lorenz_error_growth.py
---------------------------
Plots RMSE forecast error growth curves for Z500, T850, and U10
using the Lorenz error-growth model, comparing ECMWF and GFS.
Parameters are representative of published WeatherBench 2 /
ECMWF verification scores.

Usage:
    uv run python src/scripts/plot_lorenz_error_growth.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ── Lorenz error-growth model ─────────────────────────────────────────────────
def lorenz_error(t_days, e0, e_inf, lam=0.35):
    """e0: initial RMSE, e_inf: saturation RMSE, lam: Lyapunov exponent (day⁻¹)"""
    factor = np.exp(lam * t_days)
    return e0 * factor / np.sqrt(1 + (e0 / e_inf) ** 2 * (factor ** 2 - 1))


t = np.linspace(0, 10, 200)  # lead time in days

# Representative parameters (WeatherBench 2 / ECMWF verification reports)
params = {
    "ECMWF  Z500  (m)"   : dict(e0=8,   e_inf=780, lam=0.35, color="royalblue",      ls="-"),
    "GFS    Z500  (m)"   : dict(e0=10,  e_inf=780, lam=0.40, color="steelblue",      ls="--"),
    "ECMWF  T850  (K)"   : dict(e0=0.5, e_inf=8.5, lam=0.32, color="tomato",         ls="-"),
    "GFS    T850  (K)"   : dict(e0=0.6, e_inf=8.5, lam=0.37, color="darksalmon",     ls="--"),
    "ECMWF  U10   (m/s)" : dict(e0=0.8, e_inf=9.0, lam=0.45, color="seagreen",       ls="-"),
    "GFS    U10   (m/s)" : dict(e0=1.0, e_inf=9.0, lam=0.50, color="mediumseagreen", ls="--"),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
groups = [
    ("Z500  (m)",   ["ECMWF  Z500  (m)",   "GFS    Z500  (m)"],   axes[0]),
    ("T850  (K)",   ["ECMWF  T850  (K)",   "GFS    T850  (K)"],   axes[1]),
    ("U10   (m/s)", ["ECMWF  U10   (m/s)", "GFS    U10   (m/s)"], axes[2]),
]

for var_name, keys, ax in groups:
    for label in keys:
        p = params[label]
        rmse = lorenz_error(t, p["e0"], p["e_inf"], p["lam"])
        ax.plot(t, rmse, color=p["color"], ls=p["ls"], lw=2, label=label)
        ax.axhline(p["e_inf"], color=p["color"], lw=0.5, ls=":", alpha=0.5)

    ax.axvline(6.5, color="grey", lw=1.0, ls="--", alpha=0.7)
    ax.text(6.6, ax.get_ylim()[0] * 1.05 if ax.get_ylim()[0] > 0 else 0.5,
            "~Day 6–7\nuseful skill\nthreshold",
            fontsize=7, color="grey", va="bottom")

    ax.set_xlabel("Lead time (days)", fontsize=10)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.set_title(var_name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Forecast RMSE growth with lead time\n"
    "Lorenz error-growth model  ·  ECMWF vs GFS  ·  Northern Hemisphere extratropics",
    fontsize=12,
)
plt.tight_layout()

out = Path("docs/img/forecast_error_growth.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
