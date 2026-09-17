# Austrian Population Downscaling

`downscale_population` converts annual regional population stocks to the 1 km Statistics Austria grid. The reference grid is only an allocation-weight surface; its total is never used as a population total.

For each source region `r`, cell `i`, and year `y`:

```text
population(i, y) = source_total(r, y) * reference_weight(i, r) / sum(reference_weight(:, r))
```

The grid-cell weight is the reference population count times the fraction of the cell lying in the source region. This conserves every source-region total, apart from floating-point precision.

## Products

- `population_nuts3_ardeco.nc`: ARDECO NUTS-3 stocks allocated with grid weights.
- `population_municipality_statat.nc`: Statistics Austria municipality stocks allocated with grid weights.
- `population_joint_best_guess.nc`: the municipality-constrained product. Statistics Austria is authoritative because it is official and more detailed; ARDECO remains available independently as the NUTS-3 product.

The supplied Statistics Austria population-stock CSVs are retrospectively harmonized to the **2025 municipality territorial status**, including historical years. Use the matching 2025 `OGDEXT_GEM_1` WFS layer and its `g_id` GKZ field for this source:

```text
https://www.statistik.gv.at/gs-open/GEODATA/ows?service=WFS&version=2.0.0&request=GetFeature&typeNames=GEODATA:STATISTIK_AUSTRIA_GEM_20250101&outputFormat=application/json
```

Example:

```bash
downscale_population \
  --grid /data/users/alm/StatAT_data/OGDEXT_RASTER_1_STATISTIK_AUSTRIA_L001000_LAEA/Austria_population_1km_2019.nc \
  --ardeco /data/arsclisys/normal/ARDECO/ARDECO_SNPTN_NUTS2024_NUTS3_population_1January_total.csv \
  --statat-directory /data/arsclisys/normal/StatAT/population_stock \
  --nuts3-boundaries /data/users/alm/ACCREU/ARDECO/ARDECO_shapefile_NUTS3_3035.shp \
  --municipality-boundary-template 'https://www.statistik.gv.at/gs-open/GEODATA/ows?service=WFS&version=2.0.0&request=GetFeature&typeNames=GEODATA:STATISTIK_AUSTRIA_GEM_20250101&outputFormat=application/json' \
  --output-directory /path/to/output
```

For reproducible offline runs, download the 2025 layer and pass its local path. A `{year}` template such as `/data/boundaries/gemeinden_{year}.gpkg` is accepted only if its population totals use the corresponding territorial status; it must not be paired with the supplied retrospective CSVs.
