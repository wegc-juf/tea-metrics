# TEAmetrics Copilot Instructions

## Commands

This is a Poetry project requiring Python 3.11 or newer. Install the locked runtime and development dependencies with:

```bash
poetry install
```

Run the test suite:

```bash
poetry run pytest
```

Run one test file or one test case:

```bash
poetry run pytest tests/test_regrid_SPARTACUS_transforms.py
poetry run pytest tests/test_crs_propagation.py::test_crs_propagates_through_ctp_decadal_and_amplification
```

The project has no configured linter. Build distributable artifacts with:

```bash
poetry build
```

The shell integration check in `tests/test_scripts.sh` assumes it is invoked from `tests/` and exercises the SPARTACUS transform tests, the minimal `calc_tea` run, CLI help, and the example:

```bash
(cd tests && bash test_scripts.sh)
```

## Architecture

- `src/teametrics/TEA.py` contains `TEAIndicators`, the core xarray-based calculation engine. Its calculation stages are daily basis variables, climatic-time-period (CTP) indicators, decadal indicators, then amplification factors; each stage has matching NetCDF save/load methods.
- `src/teametrics/TEA_AGR.py` provides `TEAAgr`, a `TEAIndicators` subclass for Aggregate GeoRegions. `calc_TEA.py` selects it whenever a run-control configuration contains `agr`; it computes a grid of GeoRegions before aggregate means and spreads.
- `src/teametrics/calc_TEA.py` is the CLI orchestration layer behind `calc_tea`. For gridded data it loads or generates a threshold, processes daily data in ten-year chunks, writes daily and CTP outputs, then reloads CTP data for decadal and amplification calculations.
- `src/teametrics/common/config.py` is the configuration contract. Every CLI loads a YAML section named after its script (for example, `calc_TEA`), applies defaults, expands supported variables, validates types/paths/choices, and derives fields such as `param_str`.
- `src/teametrics/common/general_functions.py` handles input discovery, dataset loading, percentile thresholds, and provenance metadata. `src/teametrics/utils/` holds data preparation, mask creation, and dataset-specific regridding workflows.
- The public console scripts are declared in `pyproject.toml`: `calc_tea`, `create_region_masks`, `tea_example`, `prep_ERA5`, `prep_ERA5Land`, and `regrid_SPARTACUS_to_WEGNext`. Use `src/teametrics/config/TEA_CFG_minimal.yaml` as the smallest end-to-end `calc_tea` configuration; `docs/CFG-PARAMS-doc.md` is the parameter reference.

## Repository conventions

- Treat YAML run-control files as a public interface. Keep options inside the appropriate script section, preserve the shared-anchor-and-merge structure used by the bundled configs, and update `config.py` type/default/choice handling plus `docs/CFG-PARAMS-doc.md` when adding an option. Unknown options intentionally fail validation.
- `$script_path` is the only supported configuration placeholder and expands to the installed `teametrics` package directory. Use it for paths to packaged example data.
- TEA workflows cache and reuse NetCDF intermediates beneath `outpath`: `daily_basis_variables/`, `ctp_indicator_variables/`, and decadal/amplification output paths. Respect the `recalc_*` flags rather than changing output naming or silently recomputing stages.
- Preserve xarray coordinates, dimensions, attributes, and CRS through every transformation. `TEAIndicators.find_dim_names` accepts common pairs (`x/y`, `X/Y`, `lon/lat`, and longitude/latitude variants); mask and area-grid data carry CRS in `attrs["coordinate_sys"]`, which must propagate to derived datasets and data variables.
- SPARTACUS CRS transforms use conventional `(x, y)` order. When creating `pyproj.Transformer` instances for these transforms, use `always_xy=True`; the transform regression tests protect against authority-axis-order swaps.
- NetCDF result metadata is part of the output contract. Route CLI-produced datasets through `create_tea_history` or `create_history_from_cfg` so history, source, and dynamically versioned package metadata are retained.
- `pyproject.toml` uses dynamic Git-based versioning and persists the resolved version in `src/teametrics/__init__.py` and `README.md` during packaging. Do not manually normalize those generated version strings.
