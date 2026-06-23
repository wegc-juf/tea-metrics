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


def download_icon_eu_t2m():
    date = datetime.date.today().strftime("%Y%m%d")
    
    outdir = Path("icon_eu_t2m")
    outdir.mkdir(exist_ok=True)

    for fh in range(0, 121):

        fhr = f"{fh:03d}"

        bz2_name = (
            f"icon-eu_europe_regular-lat-lon_single-level_"
            f"{date}{run}_{fhr}_T_2M.grib2.bz2"
        )

        grib_name = bz2_name[:-4]  # remove .bz2

        url = base + bz2_name

        r = requests.get(url, timeout=60)

        if r.status_code != 200:
            print(f"missing: {fhr}")
            continue

        bz2_path = outdir / bz2_name
        grib_path = outdir / grib_name

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
    
if __name__ == "__main__":
    run_main()