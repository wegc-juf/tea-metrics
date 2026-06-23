from pathlib import Path
import requests

run = "00"      # 00,06,12,18 UTC
date = "20260623"

base = (
    "https://opendata.dwd.de/weather/nwp/icon-eu/grib/"
    f"{run}/t_2m/"
)

outdir = Path("icon_eu_t2m")
outdir.mkdir(exist_ok=True)

for fh in range(0, 121):

    fhr = f"{fh:03d}"

    fname = (
        f"icon-eu_europe_regular-lat-lon_single-level_"
        f"{date}{run}_{fhr}_T_2M.grib2.bz2"
    )

    url = base + fname

    r = requests.get(url, timeout=60)

    if r.status_code == 200:
        with open(outdir / fname, "wb") as f:
            f.write(r.content)

        print("downloaded", fhr)