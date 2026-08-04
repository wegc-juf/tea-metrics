#!/usr/bin/env python

from pathlib import Path
import requests
import bz2
import shutil
import datetime

run = "00"  # 00, 06, 12, 18 UTC

base = (
    "https://opendata.dwd.de/weather/nwp/icon-eu/grib/"
    f"{run}/t_2m/"
)

ICON_PATH = "/data/arsclisys/normal/ICON/"

def download_icon_eu_t2m():
    date = datetime.date.today().strftime("%Y%m%d")
    
    outdir = Path(f"{ICON_PATH}/icon_eu_t2m")
    outdir.mkdir(exist_ok=True)

    for fh in range(0, 121):

        fhr = f"{fh:03d}"

        bz2_name = (
            f"icon-eu_europe_regular-lat-lon_single-level_"
            f"{date}{run}_{fhr}_T_2M.grib2.bz2"
        )

        grib_name = bz2_name[:-4]  # remove .bz2

        url = base + bz2_name
        grib_path = outdir / grib_name

        if grib_path.exists():
            print(f"skipping existing {fhr}")
            continue

        r = requests.get(url, timeout=60)

        if r.status_code != 200:
            print(f"missing: {fhr}")
            continue

        bz2_path = outdir / bz2_name

        # Save compressed file
        with open(bz2_path, "wb") as f:
            f.write(r.content)

        # Decompress
        with bz2.open(bz2_path, "rb") as fin:
            with open(grib_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)

        # Remove compressed archive
        bz2_path.unlink()

        print(f"downloaded + unpacked {fhr}")


def download_icon_t2m(
    date,
    run="00",
    outdir="icon_global_t2m",
    max_hour=None,
):
    """
    Download DWD ICON Global T_2M forecasts.

    Parameters
    ----------
    date : str
        YYYYMMDD

    run : str
        "00", "06", "12", "18"

    outdir : str

    max_hour : int or None
        Forecast horizon.
        Defaults:
            180 for 00/12
            120 for 06/18
    """

    if max_hour is None:
        max_hour = 180 if run in ["00", "12"] else 120

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = (
        "https://opendata.dwd.de/weather/nwp/icon/grib/"
        f"{run}/t_2m/"
    )

    for fh in range(max_hour + 1):

        fhr = f"{fh:03d}"

        fname_bz2 = (
            f"icon_global_icosahedral_single-level_"
            f"{date}{run}_{fhr}_T_2M.grib2.bz2"
        )

        url = base + fname_bz2
        outfile = outdir / fname_bz2[:-4]

        if outfile.exists():
            print(f"skipping existing {fhr}")
            continue

        try:

            print(f"Downloading {url} ...")
            r = requests.get(url, timeout=60)

            if r.status_code != 200:
                print(f"missing: {fhr}")
                continue

            with open(outfile, "wb") as f:
                f.write(bz2.decompress(r.content))

            print(f"downloaded {fhr}")

        except Exception as e:

            print(f"failed {fhr}: {e}")


def download_icon_d2_t2m():
    base = (
        "https://opendata.dwd.de/weather/nwp/icon-d2/grib/"
        f"{run}/t_2m/"
    )
    date = datetime.date.today().strftime("%Y%m%d")
    
    outdir = Path(f"{ICON_PATH}/icon_d2_t2m")
    outdir.mkdir(exist_ok=True)

    for fh in range(0, 49):

        fhr = f"{fh:03d}"

        bz2_name = (
            f"icon-d2_germany_regular-lat-lon_single-level_"
            f"{date}{run}_{fhr}_2d_t_2m.grib2.bz2"
        )

        grib_name = bz2_name[:-4]  # remove .bz2

        url = base + bz2_name
        grib_path = outdir / grib_name

        if grib_path.exists():
            print(f"skipping existing {fhr}")
            continue

        print(f"Downloading {url} ...")
        r = requests.get(url, timeout=60)

        if r.status_code != 200:
            print(f"missing: {fhr}")
            continue

        bz2_path = outdir / bz2_name

        # Save compressed file
        with open(bz2_path, "wb") as f:
            f.write(r.content)

        # Decompress
        with bz2.open(bz2_path, "rb") as fin:
            with open(grib_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)

        # Remove compressed archive
        bz2_path.unlink()

        print(f"downloaded + unpacked {fhr}")


def run_main():
    download_icon_eu_t2m()
    download_icon_d2_t2m()
    # download_icon_t2m(date=datetime.date.today().strftime("%Y%m%d"), run="00", outdir="icon_global_t2m", max_hour=180)
    
    
if __name__ == "__main__":
    run_main()
