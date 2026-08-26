# TEA-indicators Project TODO List

## General Issues (High Priority)

- [ ] Check why TEX results differ from old ones according to gki

## General Issues
- [ ] Check NetCDF compression performance, compare without compression but rounding enabled

## Older Issues
- [ ] Optimize TEA calculations for x y grids (xarray method) - TEA_AGR.py:153
- [ ] Optimize code performance - TEA_AGR.py:612
- [ ] Take values directly from file - overview-plots/plot_main-parameter.py:173,268
- [ ] Calculate also for annual data - calc_TEA.py:106
- [ ] Use this also for non-AGR and test - calc_TEA.py:201,268
- [ ] Move TODO to test routine - calc_TEA.py:442, calc_decadal_indicators.py:147
- [ ] Make more dynamic with x and y names in CFG - TEA.py:256
- [ ] Add fine-grained integration of exceedance magnitudes based on highest available resolution - TEA.py:1031
- [ ] Try to drop non-exceedance days before resampling, then fill up timeseries again - TEA.py:1073
- [ ] Try using this way of calculation for equation 10 - TEA.py:1239
- [ ] Optimize code in TEA._calc_spread_estimators - TEA.py:1937
- [ ] Check again when SI infos are ready - TEA.py:2334,2342
- [ ] Choose between SEA and FBR - SPCS_P24h_7to7_95_template.yaml:3
- [ ] Add path to original SPARTACUS data - SPCS_P24h_7to7_95_template.yaml:43
- [ ] Choose between AUT, SEA, and FBR - SPCS_Tx99_template.yaml:3
- [ ] Add path to original SPARTACUS data - SPCS_Tx99_template.yaml:43
- [ ] Choose between ERA5 and ERA5Land - ERA5_Tx99_template.yaml:3
- [ ] Add path to ERA5 or ERA5Land data - ERA5_Tx99_template.yaml:21
- [ ] Set to true for hourly data - ERA5_Tx99_template.yaml:39
- [ ] Choose between SEA and FBR - ERA5_P24h_7to7_95_template.yaml:3
- [ ] Add path to ERA5 or ERA5Land data - ERA5_P24h_7to7_95_template.yaml:21
