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
