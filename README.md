# TEAmetrics repository

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17210239.svg)](https://doi.org/10.5281/zenodo.17210239)

This repository provides the software to compute the threshold-exceedance-amount (TEA) metrics as demonstrated in 
the paper _A new class of climate hazard metrics and its demonstration: revealing a ten-fold increase of extreme heat over Europe_ 
(https://doi.org/10.1016/j.wace.2026.100855). A detailed TEA metrics methods description is part of the 
Supplementary Information of the paper.

## Installation
Start by creating a new `virtualenv` for your project e.g. by using `virtualenvwrapper` (https://virtualenvwrapper.readthedocs.io/en/latest/):
```bash
mkvirtualenv <project_name>
```

Then, install the package using `pip`:
```bash
pip install https://wegenernet.org/downloads/teametrics/teametrics-0.7.1-py3-none-any.whl
```

## Usage

### 1) Download of input datasets
To calculate TEA metrics, you need to download and prepare your input data first.

Download input datasets (eg. ERA5, ERA5-Land, or SPARTACUS) if necessary. For ERA5 data, you can use the following 
scripts
as a starting point:
- [`download_ERA5.py`](https://github.com/wegc-juf/tea-metrics/blob/main/src/teametrics/utils/ERA5/download_ERA5.py) - for downloading ERA5 
  data from the Copernicus Climate Data Store (CDS).
- [`download_ERA5-Land.py`](https://github.com/wegc-juf/tea-metrics/blob/main/src/teametrics/utils/ERA5/download_ERA5-Land.py) - for downloading 
  ERA5-Land data from the Copernicus Climate Data Store (CDS).
- [`other download scripts`](https://github.com/wegc-juf/tea-metrics/tree/main/src/teametrics/utils)

### 2) Preparation of input datasets
For calculation of daily TEA metrics, the input data must be aggregated to daily data.
To prepare the input datasets (ERA5, ERA5-Land, or SPARTACUS) run one of the following data prep scripts:

- `prep_ERA5 --inpath INPATH --outpath OUTPATH` -- for preparing ERA5 data (aggregates hourly data to
  daily data).
- `prep_ERA5Land --inpath INPATH --outpath OUTPATH --orog-file PATH_TO_OROG_FILE` -- for preparing 
  ERA5-Land data
  (aggregates hourly data to daily data).
- `prep_ERA5Heat --inpath INPATH --outpath OUTPATH --orog-file PATH_TO_OROG_FILE` -- for preparing
  ERA5-Heat data
  (aggregates hourly data to daily data).
- `regrid_SPARTACUS_to_WEGNext --config-file CONFIG_FILE` -- only needed for SPARTACUS data for regridding 
  SPARTACUS to a regular 1 km x 1 km
  grid which is congruent with the 1 km x 1 km WEGN grid within FBR. Attention: run twice, once for regular data
  and once for orography data.

### 3) Creation of mask file (optional)
In case you want to define your own GeoRegion (GR) mask, you can create a mask file using the script
`create_region_masks --config-file CONFIG_FILE`\
This script allows you to create a mask file for your GR based on a shapefile or coordinates.
The configuration options (_run-control file_) for the script are documented in [`CFG-PARAMS-doc.md`](https://github.com/wegc-juf/tea-metrics/blob/main/docs/CFG-PARAMS-doc.md) \
(For WEGC users: input data filepaths are listed in [`create_region_masks.md`](https://github.com/wegc-juf/tea-metrics/blob/main/docs/create_region_masks.md))

### 4) Calculation of TEA metrics
After preparing all the necessary input and mask data, run `calc_tea --config-file CONFIG_FILE`.

- A minimal example config can be found in [`TEA_CFG_minimal.yaml`](https://github.com/wegc-juf/tea-metrics/blob/main/src/teametrics/config/TEA_CFG_minimal.yaml)
- Template config files are [`TEA_CFG_template.yaml`](https://github.com/wegc-juf/tea-metrics/blob/main/src/teametrics/config/TEA_CFG_template.yaml) for
gridded data and [`TEA_CFG_template_station.yaml`](https://github.com/wegc-juf/tea-metrics/blob/main/src/teametrics/config/TEA_CFG_template_station.yaml) for station data
- Templates for recreating the results of the TEAmetrics introduction paper can be found in
the `teametrics/config/paper_data/` folder
- The configuration options (_run-control file_) for the script are documented in [`CFG-PARAMS-doc.md`](https://github.com/wegc-juf/tea-metrics/blob/main/docs/CFG-PARAMS-doc.md)

### 5) Using the TEA metrics classes TEAIndicators and TEAAgr (optional)
In case you want a more fine-grained control over the TEA metrics calculations, you can use the classes
`teametrics.TEA.TEAIndicators` for normal GeoRegions, and \
`teametrics.TEA_AGR.TEAAgr` for Aggregate GeoRegions.

A simple example can be found in the script [`tea_example`](https://github.com/wegc-juf/tea-metrics/blob/main/src/teametrics/TEA_example.py).

[//]: # (Source code documentation for the classes can be found in TODO: add source code doc link - use auto doc tools.)

## Support (best-effort basis)
Just open an issue on the [GitHub repository](https://github.com/wegc-juf/tea-metrics) or contact the authors 
directly (see contacts below).

## Contacts 
- **Jürgen Fuchsberger** (Lead Software Developer): juergen.fuchsberger@uni-graz.at
- **Gottfried Kirchengast** (Project and Methodology Lead): gottfried.kirchengast@uni-graz.at
- **Stephanie Haas** (Software Co-Developer): stephanie.haas@uni-graz.at

## License
This project is licensed under the Gnu General Public License v3.0 (GPL-3.0). See the LICENSE file for details.

## Suggested Citations
If you use this code, please cite:

**This repository:**
Fuchsberger, J., Kirchengast, G., and Haas, S. J. (2026). TEAmetrics software for _A new class of climate hazard 
metrics and its demonstration: revealing a ten-fold increase of extreme heat over Europe_ and other applications. 
Version 0.6. Zenodo. 
https://doi.org/10.5281/zenodo.17210239

**The related paper:**
Kirchengast, G., S. J. Haas, and J. Fuchsberger (2026).
A new class of climate hazard metrics and its demonstration: revealing a ten-fold increase of extreme heat over Europe.
_Weather Clim. Extremes_ 51, 100855. 
https://doi.org/10.1016/j.wace.2026.100855

## TEA metrics Run Control File (RCF) specifications

The **TEA metrics Run Control File (RCF) specifications** define and frame the scope of applicability of the software and are the basis for the config parameters file ([`CFG-PARAMS-doc.md`](https://github.com/wegc-juf/tea-metrics/blob/main/docs/CFG-PARAMS-doc.md)) that informs the yaml input files feeding the TEAmetrics computation scripts (`teametrics/config/` yaml files). Compiled and maintained by Gottfried Kirchengast. For a detailed methods description of the TEA metrics computations see the Supplementary materials PDF file of the related introduction paper at https://doi.org/10.1016/j.wace.2026.100855.

[**TEAmetrics Run Control File (RCF) specifications v8-3Sep2025**](https://github.com/wegc-juf/tea-metrics/blob/main/docs/TEAmetrics_RCFspecs_v8-3Sep2025.md) --- Structured in **seven definition 
groups**, preceded by 
an optional Project&TEArun Id group with specs for orderly filepath management and identification of ensembles of 
TEAmetrics compute runs. **The following definitions apply** and are **summarized group by group below** 
(structured per group in self-explanatory style; this version covering specs up to TEAmetrics v1.0):

[**1. Input Datsets Definition**](https://github.com/wegc-juf/tea-metrics/blob/main/docs/TEAmetrics_RCFspecs_v8-3Sep2025.md?ref_type=heads#input-datasets-def)  
[**2. Key Variable Defintions**](https://github.com/wegc-juf/tea-metrics/blob/main/docs/TEAmetrics_RCFspecs_v8-3Sep2025.md?ref_type=heads#key-variable-def)    
[**3. GeoRegions Definition**](https://github.com/wegc-juf/tea-metrics/blob/main/docs/TEAmetrics_RCFspecs_v8-3Sep2025.md?ref_type=heads#georegions-def)      
[**4. Aggregate GeoRegions Definition**](https://github.com/wegc-juf/tea-metrics/blob/main/docs/TEAmetrics_RCFspecs_v8-3Sep2025.md?ref_type=heads#aggregate-georegions-def)      
[**5. Time Domain Definitions**](https://github.com/wegc-juf/tea-metrics/blob/main/docs/TEAmetrics_RCFspecs_v8-3Sep2025.md?ref_type=heads#time-domain-def)      
[**6. Threshold Map and Exceedance Definitions**](https://github.com/wegc-juf/tea-metrics/blob/main/docs/TEAmetrics_RCFspecs_v8-3Sep2025.md?ref_type=heads#threshold-map-and-exceedance-def)      
[**7. Natural Variability Estimation Definitions**](https://github.com/wegc-juf/tea-metrics/blob/main/docs/TEAmetrics_RCFspecs_v8-3Sep2025.md?ref_type=heads#natural-variability-estimation-def)      

For the complete specification file see [**TEAmetrics_RCFspecs_v8-3Sep2025.md**](https://github.com/wegc-juf/tea-metrics/blob/main/docs/TEAmetrics_RCFspecs_v8-3Sep2025.md).

## Changelog
See [Releases](https://github.com/wegc-juf/tea-metrics/releases) and [Tags](https://github.com/wegc-juf/tea-metrics/tags) for a detailed changelog.

