#!/bin/bash
pytest ../tests/test_regrid_SPARTACUS_transforms.py &&
calc_tea -cf ../src/teametrics/config/TEA_CFG_minimal.yaml && prep_ERA5 --help && prep_ERA5Land --help && create_region_masks --help &&
regrid_SPARTACUS_to_WEGNext --help && tea_example --no-gui
