#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for the coordinate transform helper functions used in
regrid_SPARTACUS.py.

These functions convert grid coordinates between CRSs (UTM33N, EPSG:3416
"Austria BMN M34", EPSG:3035 "ETRS89-LAEA Europe"). Several of these CRSs
have a non-conventional (authority-declared) axis order (e.g. EPSG:3035 is
officially Northing-then-Easting, not Easting-then-Northing). If a
pyproj.Transformer is created without always_xy=True, and the code assumes
conventional (x, y) order, coordinates get silently scrambled (e.g. huge
negative y-values). These tests perform forward+inverse round trips and
check the recovered coordinates match the originals, which would catch
this class of bug immediately.
"""

import numpy as np
import pytest
import pyproj
import xarray as xr

from teametrics.utils.SPARTACUS.regrid_SPARTACUS import (
    assign_target_crs,
    epsg3035_to_epsg3416_grid,
    epsg3416_to_epsg3035_grid,
    epsg3416_to_utm_grid,
    get_crs_from_dataset,
    utm_to_epsg3416_grid,
)

# Sample coordinates roughly within Austria for each CRS, used as round-trip
# anchors. Exact values are not important, only that they are realistic and
# that round trips recover them closely.
UTM_X = np.array([500000.0, 501000.0, 502000.0])
UTM_Y = np.array([5200000.0, 5201000.0])

EPSG3416_X = np.array([480000.0, 481000.0, 482000.0])
EPSG3416_Y = np.array([350000.0, 351000.0])


def test_utm_to_epsg3416_roundtrip():
    """UTM33N -> EPSG:3416 -> UTM33N should recover original coordinates."""
    new_x, new_y = utm_to_epsg3416_grid(UTM_X, UTM_Y)

    # sanity: EPSG:3416 (Austria BMN M34) x,y should be in a plausible range
    assert new_x.min() > 100000 and new_x.max() < 900000
    assert new_y.min() > 100000 and new_y.max() < 600000

    # Round trip individual points (not axis vectors): EPSG:3416 and UTM33N
    # are rotated relative to each other, so recombining row/column vectors
    # via meshgrid would not reproduce the original points exactly. Testing
    # single-point pairs avoids this and still catches axis-order bugs.
    for i in range(len(UTM_X)):
        for j in range(len(UTM_Y)):
            back_x, back_y = epsg3416_to_utm_grid(np.array([new_x[j, i]]),
                                                    np.array([new_y[j, i]]))
            np.testing.assert_allclose(back_x[0, 0], UTM_X[i], atol=1.0)
            np.testing.assert_allclose(back_y[0, 0], UTM_Y[j], atol=1.0)


def test_epsg3416_to_epsg3035_roundtrip():
    """EPSG:3416 -> EPSG:3035 -> EPSG:3416 should recover original coordinates."""
    new_x, new_y = epsg3416_to_epsg3035_grid(EPSG3416_X, EPSG3416_Y)

    # sanity: EPSG:3035 (ETRS89-LAEA Europe) values for Austria are
    # roughly x ~ 4.5-4.8M, y ~ 2.5-2.8M. A sign/axis-order bug produces
    # values far outside this range (e.g. large negative numbers).
    assert new_x.min() > 4000000 and new_x.max() < 5000000
    assert new_y.min() > 2000000 and new_y.max() < 3000000

    # Round trip individual points (not axis vectors): EPSG:3416 and
    # EPSG:3035 are rotated relative to each other, so recombining row/
    # column vectors via meshgrid would not reproduce the original points
    # exactly. Testing single-point pairs avoids this and still catches
    # axis-order bugs.
    for i in range(len(EPSG3416_X)):
        for j in range(len(EPSG3416_Y)):
            back_x, back_y = epsg3035_to_epsg3416_grid(np.array([new_x[j, i]]),
                                                         np.array([new_y[j, i]]))
            np.testing.assert_allclose(back_x[0, 0], EPSG3416_X[i], atol=1.0)
            np.testing.assert_allclose(back_y[0, 0], EPSG3416_Y[j], atol=1.0)


def test_epsg3035_to_epsg3416_produces_positive_y():
    """
    Regression test for a specific bug where epsg3035_to_epsg3416_grid
    produced large negative y-values due to pyproj respecting EPSG:3035's
    authority axis order (Northing, Easting) instead of conventional (x, y).
    """
    # Realistic EPSG:3035 coordinates for Austria
    x3035 = np.array([4654000.0, 4655000.0, 4656000.0])
    y3035 = np.array([2668000.0, 2669000.0])

    new_x, new_y = epsg3035_to_epsg3416_grid(x3035, y3035)

    assert np.all(new_y > 0), 'EPSG:3416 y-values must be positive for Austria'
    assert new_x.min() > 100000 and new_x.max() < 900000
    assert new_y.min() > 100000 and new_y.max() < 600000


def test_target_crs_is_read_from_statat_file(tmp_path):
    """The configured target file supplies the target CRS metadata."""
    crs = pyproj.CRS.from_epsg(3035)
    statat_file = tmp_path / 'statat.nc'
    xr.Dataset(
        {
            'crs': xr.DataArray(0, attrs={'crs_wkt': crs.to_wkt()}),
        }
    ).to_netcdf(statat_file)

    assert get_crs_from_dataset(statat_file).to_epsg() == 3035


def test_assign_target_crs_replaces_source_mapping():
    """Output metadata must describe the target, not source EPSG:3416."""
    target_crs = pyproj.CRS.from_epsg(3035)
    ds = xr.Dataset(
        {
            'RR': (('y', 'x'), np.zeros((2, 2)), {
                'grid_mapping': 'lambert_conformal_conic',
                'coordinates': 'lambert_conformal_conic',
                'esri_pe_string': 'source CRS',
            }),
            'lambert_conformal_conic': xr.DataArray(
                0, attrs={
                    'grid_mapping_name': 'lambert_conformal_conic',
                    'crs_wkt': pyproj.CRS.from_epsg(3416).to_wkt(),
                }
            ),
        },
        coords={'x': [4285500.0, 4286500.0], 'y': [2891500.0, 2890500.0]},
    )
    target = {
        'epsg': '3035',
        'grid_mapping': 'ETRS89_LAEA',
        'crs': target_crs,
    }

    result = assign_target_crs(ds, target)

    assert 'lambert_conformal_conic' not in result
    assert result.RR.attrs['grid_mapping'] == 'ETRS89_LAEA'
    assert 'coordinates' not in result.RR.attrs
    assert 'esri_pe_string' not in result.RR.attrs
    assert result.ETRS89_LAEA.attrs['grid_mapping_name'] == 'lambert_azimuthal_equal_area'
    assert result.ETRS89_LAEA.attrs['code'] == 'EPSG:3035'
    assert result.ETRS89_LAEA.attrs['geotransform'] == '4285000 1000 0 2892000 0 -1000'
    assert result.x.attrs['standard_name'] == 'projection_x_coordinate'
    assert result.y.attrs['standard_name'] == 'projection_y_coordinate'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
