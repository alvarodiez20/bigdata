"""
download_gfs_nomads.py
----------------------
Downloads a short GFS forecast sequence over the Iberian Peninsula
from the NOAA NOMADS filtered HTTP service.

Usage:
    uv run python scripts/download_gfs_nomads.py --date 20240715 --hour 00
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
