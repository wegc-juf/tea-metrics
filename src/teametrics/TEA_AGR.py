"""
Threshold Exceedance Amount (TEA) indicators Class implementation for aggregated georegions (AGR)
Based on: https://doi.org/10.1016/j.wace.2026.100855
Equation numbers refer to Supplementary Notes therin
"""
import warnings
import time
import os
from typing import Any

import xarray as xr
import numpy as np
from tqdm import trange
from xarray import Dataset

from .common.var_attrs import get_attrs
from .common.TEA_logger import logger
from .TEA import TEAIndicators, DEFAULT_MIN_DURATION
from .TEA import DEBUG


class TEAAgr(TEAIndicators):
    """
    Class for Threshold Exceedance Amount (TEA) indicators for aggregated georegions (AGR)

    as defined in https://doi.org/10.1016/j.wace.2026.100855 and
    Methods as defined in
    Kirchengast, G., Haas, S. J. & Fuchsberger, J. Compound event metrics detect and explain ten-fold
    increase of extreme heat over Europe—Supplementary Note: Detailed methods description for
    computing threshold-exceedance-amount (TEA) indicators. Supplementary Information (SI) to
    Preprint – April 2025. 40 pp. Wegener Center, University of Graz, Graz, Austria, 2025.
    """
    def __init__(self, input_data=None, threshold=None, mask=None, full_mask=None, min_area=0.0001,
                 gr_grid_res=0.5, land_sea_mask=None, gr_grid_mask=None, gr_grid_areas=None,
                 land_frac_min=0.25, cell_size_y=2, agr_frac_min=0.25, min_duration=DEFAULT_MIN_DURATION,
                 **kwargs):
        """
        initialize TEA object

        Args:
            input_data: input data grid
            threshold: threshold grid
            mask: mask grid (needed if data should be masked out by e.g. country borders)
            full_mask: region mask before altitude filtering, used for A_AGR_full
            min_area: minimum area for valid grid cells in areals (100 km^2). Default: 0.0001 areals or 10 km^2
            gr_grid_res: resolution for grid of GeoRegions (in degrees)
            land_sea_mask: land-sea mask if needed
            land_frac_min: minimum fraction of land below 1500m. Default: 0.5
            cell_size_y: size of GR grid cell in latitudinal/y direction. Default: 2
            gr_grid_mask: mask for GR grid (will be automatically generated if not provided)
            gr_grid_areas: areas for GR grid (will be automatically generated if not provided)
        """
        super().__init__(input_data=input_data, threshold=threshold,
                         mask=mask, min_area=min_area, apply_mask=False, **kwargs)
        self.full_mask = mask if full_mask is None else full_mask
        if self.area_grid is not None and not isinstance(self.area_grid, int):
            ref_grid = self.area_grid
        elif self.input_data is not None:
            ref_grid = self.input_data
        elif self.mask is not None:
            ref_grid = self.mask
        else:
            ref_grid = None

        if ref_grid is not None:
            self._y_resolution_in = round(abs(ref_grid[self.ydim].values[0] - ref_grid[self.ydim].values[1]), 4)
        else:
            self._y_resolution_in = None
        self.gr_grid_res = gr_grid_res
        self.gr_grid_mask = None
        self.land_sea_mask = land_sea_mask
        self.land_frac_min = land_frac_min
        self.cell_size_y = cell_size_y

        self.gr_grid_mask = gr_grid_mask
        self.gr_grid_areas = gr_grid_areas
        self.gr_grid_mask_full = None
        
        self.a_agr_full = None
        self.areas_full = None
        self.agr_area_weights_full = None
        self.a_agr_domain = None
        self.areas_domain = None
        self.a_agr_elig = None
        self.areas_elig = None
        self.agr_area_weights = None
        self.agr_frac_min = agr_frac_min
        
        self.min_duration = min_duration

        # filter input data to valid cells
        if self.land_sea_mask is not None and self.input_data is not None:
            self.land_sea_mask = self._align_spatial_grid(self.land_sea_mask, self.input_data)
            self.input_data = self.input_data.where(self.land_sea_mask > 0)

    @staticmethod
    def _align_spatial_grid(source, target):
        """Align a spatial grid to target coordinates, using nearest centers if needed."""
        aligned = source
        for dim in (TEAIndicators.find_dim_names(target)):
            if dim not in aligned.dims:
                raise ValueError(f'Cannot align spatial grid: dimension {dim!r} is missing.')
            target_coords = target[dim].values
            source_coords = np.asarray(aligned[dim].values, dtype=float)
            if np.isin(target_coords, source_coords).all():
                aligned = aligned.sel({dim: target_coords})
                continue

            if source_coords.size < 2:
                raise ValueError(f'Cannot align spatial grid along {dim!r}: fewer than two source coordinates.')
            tolerance = np.max(np.abs(np.diff(source_coords))) / 2 + np.finfo(float).eps
            aligned = aligned.reindex({dim: target_coords}, method='nearest', tolerance=tolerance)
        return aligned

    def calc_daily_basis_vars(self, grid=True, gr=False):
        """
        calculate all daily basis variables for grid of GeoRegions
        """
        super().calc_daily_basis_vars(grid=grid, gr=gr)

    def select_sub_gr(self, ycoord, xcoord):
        """
        select data of GeoRegion sub-cell and weight edges
        Args:
            ycoord: center value of cell for ycoord
            xcoord: center value of cell for xcoord

        Returns:
            cell_data: data of cell
            cell_static: static data of cell
        """

        daily_grid = self.get_daily_results(grid=True, gr=False)
        # This is the local TEA calculation footprint centered on this GRG
        # point. It is controlled by cell_size_y and can overlap neighboring
        # footprints; AGR weighting later uses separate grg_grid_spacing tiles.
        y_indices, y_weights = self._get_axis_overlap(
            daily_grid[self.ydim].values, ycoord, self.cell_size_y,
            latitude=self.ydim.lower() in {'lat', 'latitude'})
        if self._is_geographic_grid():
            cell_size_x = self.cell_size_y / np.cos(np.deg2rad(ycoord))
        else:
            cell_size_x = self.cell_size_y
        x_indices, x_weights = self._get_axis_overlap(
            daily_grid[self.xdim].values, xcoord, cell_size_x)

        if len(y_indices) == 0 or len(x_indices) == 0:
            return None

        y_coords = daily_grid[self.ydim].isel({self.ydim: y_indices})
        x_coords = daily_grid[self.xdim].isel({self.xdim: x_indices})
        overlap_weights = xr.DataArray(
            np.outer(y_weights, x_weights),
            coords={self.ydim: y_coords, self.xdim: x_coords},
            dims=(self.ydim, self.xdim),
        )
        indexers = {self.ydim: y_indices, self.xdim: x_indices}

        aligned_area_grid = self._align_spatial_grid(self.area_grid, daily_grid)
        cell_area_grid = aligned_area_grid.isel(indexers) * overlap_weights
        cell_area_grid.attrs = dict(aligned_area_grid.attrs)

        if self.land_frac_min > 0:
            aligned_lsm = self._align_spatial_grid(self.land_sea_mask, daily_grid)
            cell_lsm = aligned_lsm.isel(indexers)
            # Weight the land fraction by the part of each pixel covered by
            # this GRG cell and by the pixel's physical area.
            land_area = cell_area_grid * cell_lsm
            land_frac = land_area.sum(skipna=True) / cell_area_grid.sum(skipna=True)
            if land_frac < self.land_frac_min:
                return None

        # A GR variable is already reduced over the parent domain and cannot be
        # reused for this cell. The child recalculates all GR variables below.
        cell_data = daily_grid.isel(indexers)
        
        # select threshold grid for cell
        if isinstance(self.threshold_grid, xr.DataArray):
            aligned_threshold_grid = self._align_spatial_grid(self.threshold_grid, daily_grid)
            cell_threshold_grid = aligned_threshold_grid.isel(indexers)
        else:
            cell_threshold_grid = self.threshold_grid
        if self.population_grid is not None:
            aligned_population_grid = self._align_spatial_grid(self.population_grid, daily_grid)
            cell_population_grid = aligned_population_grid.isel(indexers) * overlap_weights
            cell_population_grid.attrs = dict(aligned_population_grid.attrs)
        else:
            cell_population_grid = None
        
        if len(cell_area_grid[self.ydim]) == 0 or len(cell_area_grid[self.xdim]) == 0:
            return None

        # TODO: optimize for x/y grids (xarray method); either stack the selected
        # cells or calculate individual child TEA objects.
        tea_sub_gr = TEAIndicators(area_grid=cell_area_grid, min_area=self._min_area, unit=self.unit, ctp=self.CTP,
                                   threshold=cell_threshold_grid, population_grid=cell_population_grid)

        def replace_daily_variable(name, values, attrs=None):
            if attrs is None:
                attrs = dict(cell_data[name].attrs) if name in cell_data else {}
            cell_data[name] = values
            cell_data[name].attrs = attrs

        # The cached daily fields are extensive over each full source pixel.
        # Rebuild their area-dependent forms from the fractionally weighted
        # pixel areas before the child calculates GR-level aggregates.
        replace_daily_variable('area_grid', cell_area_grid,
                               dict(self.area_grid.attrs))
        if 'DTEC' in cell_data:
            dtea = cell_data.DTEC * cell_area_grid
            replace_daily_variable('DTEA', dtea)
            if 'DTEM' in cell_data:
                replace_daily_variable('DTEMA', cell_data.DTEM * dtea)

        # Population is extensive too. The child helper selects annual
        # population by each day's year before deriving DTEP and DTEMP.
        if cell_population_grid is not None:
            replace_daily_variable('population_grid', cell_population_grid,
                                   dict(cell_population_grid.attrs))
            if 'DTEC' in cell_data:
                population_by_day = tea_sub_gr._population_for_time(cell_data.time)
                dtep = cell_data.DTEC * population_by_day / 10000
                replace_daily_variable('DTEP', dtep)
                if 'DTEM' in cell_data:
                    replace_daily_variable('DTEMP', cell_data.DTEM * dtep)

        tea_sub_gr.set_daily_results(cell_data)
        return tea_sub_gr

    @staticmethod
    def _get_axis_overlap(coordinates, center, cell_size, latitude=False):
        """Return source-pixel indices and footprint-overlap fractions.

        Coordinates are source-raster pixel centers. The returned fraction for
        each pixel is the fraction of its footprint covered by the requested
        cell interval; latitude fractions use sine latitude for spherical area.
        """
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # Pixel edges are halfway between neighboring coordinate centers.
        edges = np.empty(coordinates.size + 1, dtype=float)
        if coordinates.size == 1:
            edges[:] = [coordinates[0] - cell_size / 2, coordinates[0] + cell_size / 2]
        else:
            edges[1:-1] = (coordinates[:-1] + coordinates[1:]) / 2
            edges[0] = coordinates[0] + (coordinates[0] - edges[1])
            edges[-1] = coordinates[-1] + (coordinates[-1] - edges[-2])

        pixel_low = np.minimum(edges[:-1], edges[1:])
        pixel_high = np.maximum(edges[:-1], edges[1:])
        cell_low, cell_high = center - cell_size / 2, center + cell_size / 2
        overlap_low = np.maximum(pixel_low, cell_low)
        overlap_high = np.minimum(pixel_high, cell_high)
        has_overlap = overlap_high > overlap_low

        if latitude:
            pixel_span = np.sin(np.deg2rad(pixel_high)) - np.sin(np.deg2rad(pixel_low))
            overlap_span = (np.sin(np.deg2rad(overlap_high))
                            - np.sin(np.deg2rad(overlap_low)))
        else:
            pixel_span = pixel_high - pixel_low
            overlap_span = overlap_high - overlap_low
        fractions = np.divide(overlap_span, pixel_span, out=np.zeros_like(overlap_span),
                              where=pixel_span > 0)
        indices = np.flatnonzero(has_overlap)
        return indices, fractions[indices]

    def _is_geographic_grid(self):
        """Return whether the active spatial dimensions identify lon/lat coordinates."""
        return (self.xdim.lower() in {'lon', 'longitude'}
                and self.ydim.lower() in {'lat', 'latitude'})

    def set_ctp_results(self, xcoord, ycoord, ctp_results):
        """
        set CTP variables for point
        Args:
            xcoord: x coordinate value
            ycoord: y coordinate value
            ctp_results: CTP GR data for point
        """
        # remove GR from variable names
        ctp_results = ctp_results.rename({var: var.replace('_GR', '') for var in ctp_results.data_vars})

        if self.ctp_results is None or not len(self.ctp_results.data_vars):
            data_vars = [var for var in ctp_results.data_vars]
            var_dict = {}
            xcoords, ycoords = self._get_xy()
            for var in data_vars:
                if 'time' in ctp_results[var].dims:
                    var_dict[var] = (['time', self.ydim, self.xdim], np.nan * np.ones((len(ctp_results.time),
                                                                                       len(ycoords),
                                                                                       len(xcoords))))
                elif len(ctp_results[var].dims) == 0:
                    var_dict[var] = ([self.ydim, self.xdim], np.nan * np.ones((len(ycoords), len(xcoords))))
                else:
                    raise ValueError(f'Unsupported variable dimensions for variable {var}: {ctp_results[var].dims}')
            self.ctp_results = xr.Dataset(coords={'time': ctp_results.time,
                                                  self.xdim: xcoords,
                                                  self.ydim: ycoords},
                                          data_vars=var_dict,
                                          attrs=ctp_results.attrs)

        self.ctp_results.loc[{self.ydim: ycoord, self.xdim: xcoord}] = ctp_results

        # set attributes for variables
        for var in ctp_results.data_vars:
            if 'attrs' not in self.ctp_results[var]:
                attrs = ctp_results[var].attrs
                new_attrs = get_attrs(vname=var)
                attrs['long_name'] = new_attrs['long_name']
                self.ctp_results[var].attrs = attrs

    def get_ctp_results(self, grid=True, gr=True):
        """
        get CTP results for grid of GeoRegions
        """
        return self.ctp_results

    def save_ctp_results(self, filepath, variables=None, save_tiff=False):
        """
        save all CTP results to filepath
        Args:
            filepath: path to save the results.
            variables: list of specific variable names to save as GeoTIFF. If None, saves all raster variables.
            save_tiff: if True, save results as GeoTIFF in addition to NetCDF. Default: False
        """
        with warnings.catch_warnings():
            # ignore warnings due to nan multiplication
            warnings.simplefilter("ignore")
            try:
                if save_tiff:
                    self._save_geotiff(filepath, variables, dataset=self.ctp_results, split_by_year=False)
                self._to_netcdf(self.ctp_results, filepath)
            except PermissionError as err:
                if not DEBUG:
                    raise err
                logger.info(f"Permission denied for {filepath}. Trying to save file to /tmp/ instead.")
                filename = os.path.basename(filepath)
                filepath = os.path.join('/tmp/', filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                self._to_netcdf(self.ctp_results, filepath)

    def _apply_gr_grid_mask(self):
        """
        apply GR grid mask to CTP results
        """
        if self.ctp_results is not None:
            self.ctp_results = self.ctp_results.where(self.gr_grid_mask > 0)

    def calc_annual_ctp_indicators(self, ctp=None, drop_daily_results=False):
        """
        calculate annual CTP indicators for all GeoRegions in GeoRegion grid
        Args:
            ctp: Climatic Time Period (CTP) to resample to
                allowed values: 'annual', 'seasonal', 'WAS', 'ESS', 'JJA', 'DJF', 'EWS', 'monthly'
                'WAS': warm season (April to October)
                'ESS': extended summer season (May to September)
                'JJA': summer season (June to August)
                'DJF': winter season (December to February)
                'EWS': extended winter season (November to March)
            drop_daily_results: if True, drop daily results after calculation

        Returns:

        """
        if ctp is not None:
            self._set_ctp(ctp)

        self._calc_annual_gr_grid()
        self._apply_gr_grid_mask()

        if drop_daily_results:
            self.daily_results.close()
            del self._daily_results_filtered
            del self.daily_results

    def _calc_annual_gr_grid(self):
        """
        calculate annual CTP TEA indicators for all GeoRegions in GeoRegion grid
        Args:

        Returns:

        """
        xcoords, ycoords = self._get_xy()

        valid_cells_found = False
        for i_y in trange(len(ycoords), desc='Processing AGR cells'):
            ycoord = ycoords[i_y]
            valid_cells_found |= self._calc_tea_ctp_at_ycoord(ycoord, xcoords=xcoords)
        if not valid_cells_found:
            logger.error('No valid cells found for annual CTP calculation. Try to decrease the land_frac_min '
                         'parameter or check the region definition. ')

    def _crop_to_shp(self):
        """
        crop GeoRegion grid data to spatial extent of AGR shape

        Returns:

        """
        if ((self.gr_grid_mask[self.xdim] != self._ref_mean[self.xdim]).all() or (self.gr_grid_mask[self.ydim] !=
                self._ref_mean[self.ydim]).all()):
            raise ValueError('GR grid mask and reference mean have different coordinates. Cannot crop to shape.')
        self._crop_to_gr_mask_extents()
        # Keep only centers with mask coverage; fractional boundary weighting
        # is applied once when calc_a_agr builds agr_area_weights.
        self._ref_mean = self._ref_mean.where(self.gr_grid_mask > 0)
        self._cc_mean = self._cc_mean.where(self.gr_grid_mask > 0)
        self.decadal_results = self.decadal_results.where(self.gr_grid_mask > 0)
        self.amplification_factors = self.amplification_factors.where(self.gr_grid_mask > 0)
        if self.ctp_results is not None:
            self.ctp_results = self.ctp_results.where(self.gr_grid_mask > 0)

    def _crop_to_gr_mask_extents(self):
        """
        crop GeoRegion grid data to spatial extent of aggregated GeoRegion mask

        Returns:

        """
        mask_tmp = self.mask
        self.mask = self.gr_grid_mask
        self._crop_to_mask_extents()
        self.mask = mask_tmp

    def _crop_to_rect(self, x_range, y_range):
        """
        crop GeoRegion grid data to spatial extent of aggregated GeoRegion
        Args:
            x_range: x coordinate range (min, max)
            y_range: y coordinate range (min, max)

        Returns:

        """
        if y_range is None:
            y_range = (self.gr_grid_areas[self.ydim].min(), self.gr_grid_areas[self.ydim].max())
        if x_range is None:
            x_range = (self.gr_grid_areas[self.xdim].min(), self.gr_grid_areas[self.xdim].max())

        # use correct slice order for lat/lon and xy grids
        if self.gr_grid_areas[self.ydim][0] > self.gr_grid_areas[self.ydim][-1]:
            y_slice = slice(max(y_range), min(y_range))
        else:
            y_slice = slice(min(y_range), max(y_range))
        x_slice = slice(x_range[0], x_range[1])

        self.gr_grid_areas = self.gr_grid_areas.sel({self.ydim: y_slice, self.xdim: x_slice})
        self._ref_mean = self._ref_mean.sel({self.ydim: y_slice, self.xdim: x_slice})
        self._cc_mean = self._cc_mean.sel({self.ydim: y_slice, self.xdim: x_slice})
        self.decadal_results = self.decadal_results.sel({self.ydim: y_slice, self.xdim: x_slice})
        self.amplification_factors = self.amplification_factors.sel({self.ydim: y_slice, self.xdim: x_slice})
        if self.ctp_results:
            self.ctp_results = self.ctp_results.sel({self.ydim: y_slice, self.xdim: x_slice})

    def _drop_agr_values_and_spreads(self):
        """
        drop old AGR values and spreads
        """
        agr_vars = [var for var in self.decadal_results.data_vars if 'AGR' in var or 'supp' in var or 'slow' in var]
        self.decadal_results = self.decadal_results.drop_vars(agr_vars)
        
        agr_vars = [var for var in self.ctp_results.data_vars if 'AGR' in var or 'supp' in var or 'slow' in var]
        self.ctp_results = self.ctp_results.drop_vars(agr_vars)
        
        agr_vars = [var for var in self.amplification_factors.data_vars if 'AGR' in var or 'supp' in var or 'slow' in
                    var]
        self.amplification_factors = self.amplification_factors.drop_vars(agr_vars)
        
        agr_vars = [var for var in self._ref_mean.data_vars if 'AGR' in var or 'supp' in var or 'slow' in var]
        self._ref_mean = self._ref_mean.drop_vars(agr_vars)
        
        agr_vars = [var for var in self._cc_mean.data_vars if 'AGR' in var or 'supp' in var or 'slow' in var]
        self._cc_mean = self._cc_mean.drop_vars(agr_vars)
        
        # also drop ref and CC
        unwanted_vars = [var for var in self.decadal_results if '_ref' in var or '_CC' in var]
        self.decadal_results = self.decadal_results.drop_vars(unwanted_vars)
        # also drop ref and CC
        unwanted_vars = [var for var in self._ref_mean if '_ref' in var or '_CC' in var]
        self._ref_mean = self._ref_mean.drop_vars(unwanted_vars)
        # also drop ref and CC
        unwanted_vars = [var for var in self._cc_mean if '_ref' in var or '_CC' in var]
        self._cc_mean = self._cc_mean.drop_vars(unwanted_vars)

    def _calc_weighted_perc(self, data, areas, annual=False):
        """
        calculate weighted percentiles
        :param data: data
        :param areas: area sizes
        :param annual: annual data
        :return:
        """
        # remove values where data is nan
        areas_3d = areas.expand_dims({'time': data.time}) if 'time' in data.dims else areas
        areas_3d = areas_3d.where(~np.isnan(data))

        wgts = areas_3d / areas_3d.sum(dim=(self.ydim, self.xdim))

        if 'time' not in data.dims:
            pval005, pval095 = TEAAgr._calc_weighted_perc_single(data.values, wgts.values)
            return pval005, pval095

        perc005 = xr.DataArray(data=np.ones(len(data.time)) * np.nan,
                               coords={'time': (['time'], data.time.data)})
        perc095 = xr.DataArray(data=np.ones(len(data.time)) * np.nan,
                               coords={'time': (['time'], data.time.data)})

        for iyr, yr in enumerate(data.time):
            if iyr < 5 and not annual:
                continue
            avals = data[iyr, :, :].values
            wgts_iyr = wgts[iyr, :, :].values

            pval005, pval095 = TEAAgr._calc_weighted_perc_single(avals, wgts_iyr)

            perc005[iyr] = pval005
            perc095[iyr] = pval095

        return perc005, perc095

    @staticmethod
    def _calc_weighted_perc_single(avals, wgts):
        """
        calculate weighted percentiles for a single timestep
        Args:
            avals: data
            wgts: weights

        Returns:
            pval005, pval095: 5th and 95th percentile

        """
        # stack arrays along a new dimension and reshape to a 2D array
        # (rows contain two values to sort together)

        combined = np.stack((avals, wgts), axis=-1)
        combined_reshaped = combined.reshape(-1, 2)
        # sort combined array by avals
        combined_sorted = combined_reshaped[combined_reshaped[:, 0].argsort()]
        # split the sorted array back into separate ones
        avals_ordered = combined_sorted[:, 0]
        wgts_ordered = combined_sorted[:, 1]
        # calculate cumsum of weights to find percentiles
        wgts_cumsum = np.nancumsum(wgts_ordered)
        wgts_diff_005 = np.abs(wgts_cumsum - 0.05)
        wgts_diff_095 = np.abs(wgts_cumsum - 0.95)
        # find index of wgts_cumsum that is closest to 0.05 (but smaller or equal than 0.05)
        try:
            p005_index = np.nanargmin(wgts_diff_005)
            if wgts_cumsum[p005_index] > 0.05 and p005_index > 0:
                while wgts_cumsum[p005_index] > 0.05:
                    p005_index -= 1
            pval005 = avals_ordered[p005_index]
        except ValueError:
            pval005 = np.nan
        # find index of wgts_cumsum that is closest to 0.95 (but greater or equal than 0.95)
        try:
            p095_index = np.nanargmin(wgts_diff_095)
            if wgts_cumsum[p095_index] < 0.95:
                while wgts_cumsum[p095_index] < 0.95 and p095_index < len(wgts_cumsum) - 1:
                    p095_index += 1
            pval095 = avals_ordered[p095_index]
        except ValueError:
            pval095 = np.nan
        return pval005, pval095
    
    def calc_a_agr(self):
        """
        Calculate AGR areas
        """
        self.agr_area_weights = self._get_agr_area_weights()
        full_coverage = self.gr_grid_mask_full
        if full_coverage is None:
            full_coverage = self.gr_grid_mask
        self.agr_area_weights_full = self._get_agr_area_weights(full_coverage)
        self._calc_a_agr_elig()
        self._calc_a_agr_domain()
        self._calc_a_agr_full()

    def _get_agr_area_weights(self, coverage_mask=None):
        """Apply a GRG-tile coverage fraction once to reusable tile areas.

        ``gr_grid_areas`` remains the full, mask-independent spacing-tile area.
        The supplied coverage grid (altitude-filtered by default) is applied
        here. Filtered weights are used by AGR means, domain/eligible areas,
        spreads, and percentiles; full-region weights determine A_AGR_full.
        Do not multiply by a coverage grid again downstream.
        """
        if self.gr_grid_areas is None or self.gr_grid_mask is None:
            raise ValueError('GRG tile areas and mask are required for AGR area weights.')
        if coverage_mask is None:
            coverage_mask = self.gr_grid_mask
        if coverage_mask is None:
            raise ValueError('A GRG coverage mask is required for AGR area weights.')
        coverage = coverage_mask.reindex({
            self.ydim: self.gr_grid_areas[self.ydim].values,
            self.xdim: self.gr_grid_areas[self.xdim].values,
        }).fillna(0)
        coverage = coverage.clip(min=0, max=1)
        area_weights = (self.gr_grid_areas * coverage).where(coverage > 0)
        area_weights.attrs = dict(self.gr_grid_areas.attrs)
        return area_weights
    
    def _calc_a_agr_elig(self):
        """
        calculate eligible area for AGR calculation (A_AGR_elig, equation TBD)
        
        Returns:

        """
        # agr_area_weights already includes the boundary coverage fraction.
        self.areas_elig = self.agr_area_weights.where(self._ref_mean.ED > 0)
        self.a_agr_elig = self.areas_elig.sum()
        
    def _calc_a_agr_domain(self):
        """
        calculate domain area for AGR calculation (A_AGR_domain, equation TBD)
        
        Returns:

        """
        self.areas_domain = self.agr_area_weights.where(~np.isnan(self._ref_mean.threshold_avg))
        self.a_agr_domain = self.areas_domain.sum()

    def _calc_a_agr_full(self):
        """
        calculate full area for AGR calculation (A_AGR_full, equation TBD)
        
        Returns:

        """
        self.areas_full = self.agr_area_weights_full.copy()
        self.a_agr_full = self.areas_full.sum()

    def calc_agr_vars(self, y_range=None, x_range=None, spreads=True, crop_to_shp=False, calc_annual=False):
        """
        calculate AGR variables

        Args:
            x_range: x coordinate range (min, max). Default: Full region
            y_range: y coordinate range (min, max). Default: Full region
            spreads: if True, calculate spreads and percentiles for AGR variables
            crop_to_shp: if True, crop data to shape of aggregated GeoRegion (default: False)
            calc_annual: if True, calculate vars also for annual data
        """
        min_duration_avg = self.min_duration / 10
        xt_p_agr = None
        x_p_agr = None
        x_p_spreads = None
        xt_p_ref_agr = None
        
        # filter data to spatial extent of aggregated GeoRegion
        if y_range is not None or x_range is not None:
            self._crop_to_rect(x_range=x_range, y_range=y_range)
        elif crop_to_shp:
            self._crop_to_shp()

        # drop old AGR values and spreads
        self._drop_agr_values_and_spreads()

        # calculate area weights (equation 34_0)
        if self.gr_grid_areas is None:
            raise ValueError('No GR area grid provided. Please provide a valid GR area grid.')
        
        # calculate areas for AGR calculation, full area, domain area, and eligible area
        self.calc_a_agr()
        
        # check if there are enough valid grid cells for AGR calculation
        ratio = self.a_agr_elig / self.a_agr_domain
        if ratio <= self.agr_frac_min:
            self.areas_elig = self.areas_elig * np.nan
        
        # calc X_Ref^AGR and X_s^AGR (equation 34_1 and equation 34_2)
        x_ref_agr = self.calc_area_weighted_mean(self.areas_elig, self._ref_mean)
        xt_s_agr = self.calc_area_weighted_mean(self.areas_elig, self.decadal_results)
        if calc_annual:
            xt_p_agr = self.calc_area_weighted_mean(self.areas_elig, self.ctp_results)
        
        # calc Xt_ref_agr (equation 34_3)
        xt_ref_agr = self._calc_gmean_decadal(start_year=self.ref_period[0], end_year=self.ref_period[1], data=xt_s_agr)
        if calc_annual:
            xt_p_ref_agr = xt_p_agr.sel(
                time=slice(f'{self.ref_period[0]}-01-01', f'{self.ref_period[1]}-12-31')).mean(dim='time')

        # calculate X_s_AGR (equation 34_4)
        x_s_agr = (x_ref_agr / xt_ref_agr) * xt_s_agr
        if calc_annual:
            x_p_agr = (x_ref_agr / xt_ref_agr) * xt_p_agr

        # calculate compound variables (equation 35)
        x_s_agr = self._calc_compound_vars(x_s_agr)
        if calc_annual:
            x_p_agr = self._calc_compound_vars(x_p_agr)
        x_ref_agr = self._calc_compound_vars(x_ref_agr)

        # set values to nan for first and last 5/4 years for all variables with dimension 'time' to avoid edge
        # effects of decadal averaging
        for var in x_s_agr.data_vars:
            ds = x_s_agr[var]
            if 'time' in ds.dims:
                ds[dict(time=slice(0, 5))] = np.nan
                ds[dict(time=slice(-4, None))] = np.nan

        # calculate spread estimates (equation 38)
        if spreads:
            x_s_spreads = self._calc_agr_spread(data=self.decadal_results, ref=x_s_agr)
            if calc_annual:
                x_p_spreads = self._calc_agr_spread(data=self.ctp_results, ref=x_p_agr)
        else:
            x_s_spreads = xr.Dataset()
            x_p_spreads = xr.Dataset()

        # calculate CC period averages (equation 36)
        x_cc_agr = self._calc_gmean_decadal(start_year=self.cc_period[0], end_year=self.cc_period[1], data=x_s_agr)

        # calculate amplification factors (equation 37)
        self._apply_min_duration(ds=x_ref_agr, min_duration=min_duration_avg, duration_data=x_ref_agr)
        af_agr = x_s_agr / x_ref_agr
        af_cc_agr = x_cc_agr / x_ref_agr

        # rename variables
        rename_dict = {var: f'{var}_AGR_AF' for var in af_agr.data_vars}
        af_agr = af_agr.rename(rename_dict)
        rename_dict = {var: f'{var}_AGR_AF_CC' for var in af_cc_agr.data_vars}
        af_cc_agr = af_cc_agr.rename(rename_dict)

        if spreads:
            # calculate AF spread estimates (equation 38)
            af_spreads = self._calc_agr_spread(data=self.amplification_factors, ref=af_agr)
            af_spreads = af_spreads.rename({var: var.replace('AF_AGR', 'AGR_AF') for var in af_spreads.data_vars})

            # calculate spread estimates for CC period (equation 39)
            # # select only variables containing 'CC' in name of self.amplification_factors
            af_cc = self.amplification_factors[[var for var in self.amplification_factors.data_vars if 'CC' in var]]
            af_cc_spreads = self._calc_agr_spread(data=af_cc, ref=af_cc_agr)
            af_cc_spreads = af_cc_spreads.rename({var: var.replace('AF_CC_AGR', 'AGR_AF_CC') for var in
                                                  af_cc_spreads.data_vars})
            x_cc_spreads = self._calc_agr_spread(data=self._cc_mean, ref=x_cc_agr)

            # calculate spread estimates for reference period (equation 40)
            x_ref_spreads = self._calc_agr_spread(data=self._ref_mean, ref=x_ref_agr)
        else:
            af_spreads = xr.Dataset()
            af_cc_spreads = xr.Dataset()
            x_cc_spreads = xr.Dataset()
            x_ref_spreads = xr.Dataset()

        # calculate error estimates for AGR mean (equation 42TODEFINE)
        N_dof = self._get_N_dof()
        
        if spreads:
            # add p5 and p95 values (equation 41TODEFINE)
            # TODO: optimize this
            if calc_annual:
                self._calc_agr_percentiles(data=self.ctp_results, annual=True)
            self._calc_agr_percentiles(data=self.decadal_results)
            self._calc_agr_percentiles(data=self._ref_mean)
            self._calc_agr_percentiles(data=self._cc_mean)
            self._calc_agr_percentiles(data=self.amplification_factors)
        
        x_cc_agr, x_cc_spreads, x_ref_agr, x_ref_spreads, x_s_agr, x_p_agr = self._rename_AGR_vars(
            x_cc_agr, x_cc_spreads, x_ref_agr, x_ref_spreads, x_s_agr, x_p_agr, calc_annual)
        
        # add attributes
        self._add_attributes(af_agr)
        self._add_attributes(af_cc_agr)
        self._add_attributes(x_cc_agr)
        self._add_attributes(x_ref_agr)
        self._add_attributes(x_s_agr)
        if calc_annual:
            self._add_attributes(x_p_agr)

        # merge
        x_s_agr = xr.merge([x_s_agr, x_s_spreads])
        af_agr = xr.merge([af_agr, af_cc_agr, af_spreads, af_cc_spreads])
        af_agr = self._duplicate_vars(af_agr)
        if calc_annual:
            x_p_agr = xr.merge([x_p_agr, x_p_spreads])

        # add number of degrees of freedom for AGR mean
        x_s_agr['N_dof_AGR'] = N_dof
        x_s_agr['N_dof_AGR'].attrs = {'long_name': 'Number of degrees of freedom for AGR mean'}
        af_agr['N_dof_AGR'] = N_dof
        af_agr['N_dof_AGR'].attrs = {'long_name': 'Number of degrees of freedom for AGR mean'}
        if calc_annual:
            x_p_agr['N_dof_AGR'] = N_dof

        # put together to final output
        self.decadal_results = xr.merge([x_s_agr, x_ref_spreads, x_cc_spreads, x_ref_agr,
                                         x_cc_agr, self.decadal_results],
                                        compat='override')
        if calc_annual:
            self.ctp_results = xr.merge([x_p_agr, self.ctp_results], compat='override')
        self.amplification_factors = xr.merge([af_agr, self.amplification_factors], compat='override')
    
    def _rename_AGR_vars(self, x_cc_agr: Dataset, x_cc_spreads: Dataset, x_ref_agr: Dataset, x_ref_spreads: Dataset,
                         x_s_agr: Dataset, x_p_agr: Dataset, calc_annual=False) -> tuple[
                            Dataset, Dataset, Dataset, Dataset, Dataset, Dataset]:
        # rename variables
        rename_dict = {var: f'{var}_AGR' for var in x_s_agr.data_vars}
        x_s_agr = x_s_agr.rename(rename_dict)
        rename_dict = {var: f'{var}_AGR_ref' for var in x_ref_agr.data_vars}
        x_ref_agr = x_ref_agr.rename(rename_dict)
        rename_dict = {var: f'{var}_AGR_CC' for var in x_cc_agr.data_vars}
        x_cc_agr = x_cc_agr.rename(rename_dict)
        if calc_annual:
            rename_dict = {var: f'{var}_AGR' for var in x_p_agr.data_vars}
            x_p_agr = x_p_agr.rename(rename_dict)
        rename_dict = {var: var.replace('AGR', 'AGR_CC') for var in x_cc_spreads.data_vars}
        x_cc_spreads = x_cc_spreads.rename(rename_dict)
        rename_dict = {}
        for var in self._cc_mean.data_vars:
            if 'AGR' in var:
                rename_dict[var] = var.replace('AGR', 'AGR_CC')
            else:
                rename_dict[var] = f"{var}_CC"
        self._cc_mean = self._cc_mean.rename(rename_dict)
        rename_dict = {var: var.replace('AGR', 'AGR_ref') for var in x_ref_spreads.data_vars}
        x_ref_spreads = x_ref_spreads.rename(rename_dict)
        rename_dict = {}
        for var in self._ref_mean.data_vars:
            if 'AGR' in var:
                rename_dict[var] = var.replace('AGR', 'AGR_ref')
            else:
                rename_dict[var] = f"{var}_ref"
        self._ref_mean = self._ref_mean.rename(rename_dict)
        return x_cc_agr, x_cc_spreads, x_ref_agr, x_ref_spreads, x_s_agr, x_p_agr
    
    # noinspection PyPep8Naming
    def _get_N_dof(self) -> float | Any:
        A_AGR = self.a_agr_full
        # calculate error estimates for AGR mean (equation 42TODEFINE)
        r_earth = 6371
        u_earth = 2 * np.pi * r_earth
        # This is the nominal local GR calculation footprint, not the
        # spacing-tile area weights in gr_grid_areas.
        if self._is_geographic_grid():
            A_GR_full = (u_earth / 360 * self.cell_size_y) ** 2 / 100
        else:
            A_GR_full = (self.cell_size_y / 1000) ** 2 / 100
        N_dof = max(A_AGR / A_GR_full, 1)
        return N_dof
    
    def _add_attributes(self, dataset):
        """
        set correct attributes for all vars of dataset
        Args:
            dataset: xarray dataset

        Returns:

        """
        for vvar in dataset.data_vars:
            dataset[vvar].attrs = get_attrs(vname=vvar, data_unit=self.unit)
    
    def calc_area_weighted_mean(self, area, data) -> Any:
        """
        calculate area weighted mean according to equation 34_1
        Args:
            area: area data
            data: data

        Returns:
            result: area weighted mean

        """
        awgts = area / area.sum(dim=(self.ydim, self.xdim))
        result = (awgts * data).sum(dim=(self.ydim, self.xdim))
        # set result to nan where all data is nan
        result = result.where(~np.isnan(data).all(dim=(self.ydim, self.xdim)))
        return result
    
    def _calc_agr_spread(self, data, ref):
        """
        calculate spread estimates of grid cell values around AGR mean (equation 38)
        Args:
            data: data for spread estimation
            ref: reference data for spread estimation
        Returns:
            xarray dataset with upper and lower spread estimates

        """
        ref = ref.rename({var: var.replace('_AGR', '') for var in ref.data_vars})

        # Equations 38/39 use the same in-region, once-mask-weighted areas as
        # the AGR means; raw gr_grid_areas would overweight boundary tiles.
        areas = self.agr_area_weights

        # equation 38_1 and 38_4
        c_upp = xr.where(data >= ref, 1, 0)

        # equation 38_2 and 38_5
        c_upp_sum = (c_upp * areas).sum(dim=(self.ydim, self.xdim))
        # replace 0 values with nan to avoid division by zero
        c_upp_sum = c_upp_sum.where(c_upp_sum > 0)
        s_upp = np.sqrt(1 / c_upp_sum * (c_upp * areas * (data - ref) ** 2).sum(dim=(self.ydim, self.xdim)))

        # equation 38_3 and 38_6
        c_low_sum = ((1 - c_upp) * areas).sum(dim=(self.ydim, self.xdim))
        # replace 0 values with nan to avoid division by zero
        c_low_sum = c_low_sum.where(c_upp_sum > 0)
        s_low = np.sqrt(1 / c_low_sum * ((1 - c_upp) * areas * (data - ref) ** 2).sum(dim=(self.ydim, self.xdim)))

        # get attributes for variables and rename them
        for vvar in s_upp.data_vars:
            s_upp[vvar].attrs = get_attrs(vname=f'{vvar}_AGR', spread='upper')
        for vvar in s_low.data_vars:
            s_low[vvar].attrs = get_attrs(vname=f'{vvar}_AGR', spread='lower')
        rename_dict_up = {var: f'{var}_AGR_supp' for var in s_upp.data_vars}
        rename_dict_low = {var: f'{var}_AGR_slow' for var in s_low.data_vars}
        s_upp = s_upp.rename(rename_dict_up)
        s_low = s_low.rename(rename_dict_low)

        return xr.merge([s_upp, s_low], compat='no_conflicts')

    def _calc_agr_percentiles(self, data, annual=False):
        """
        calculate spread estimates of grid cell values around AGR mean (equation 38)
        Args:
            data: data for spread estimation
            annual: set True for annual data (default: False)
            
        Returns:
            xarray dataset with upper and lower spread estimates

        """
        logger.info(f'Calculating 5th and 95th percentiles')
        areas = self.areas_elig

        for var in data.data_vars:
            # calculate 5th and 95th percentiles for each variable

            # standard method (gives only slightly different results to weighted method)
            # p5_std = data[var].quantile(0.05, dim=(self.ydim, self.xdim))
            # p95_std = data[var].quantile(0.95, dim=(self.ydim, self.xdim))

            p5, p95 = self._calc_weighted_perc(data[var], areas, annual)

            if 'AF' in var:
                var = var.replace('AF', 'AGR_AF')
                
            elif 'ref' in var:
                var = var.replace('ref', 'AGR_ref')
            elif 'CC' in var:
                var = var.replace('CC', 'AGR_CC')
            else:
                var = f'{var}_AGR'
            data[f'{var}_p05'] = p5
            data[f'{var}_p95'] = p95
            data[f'{var}_p05'].attrs = get_attrs(vname=f'{var}', spread='5th percentile')
            data[f'{var}_p95'].attrs = get_attrs(vname=f'{var}', spread='95th percentile')

    @staticmethod
    def _grid_centers_from_source_axis(coordinates, grid_spacing):
        """Build a GR lattice anchored to the native source-coordinate phase."""
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.size < 2:
            raise ValueError('At least two source coordinates are required to build the GR grid.')
        if grid_spacing <= 0:
            raise ValueError('GR grid spacing must be positive.')

        steps = np.abs(np.diff(coordinates))
        source_spacing = float(np.median(steps))
        tolerance = max(source_spacing, grid_spacing) * 1e-8
        if not np.allclose(steps, source_spacing, rtol=1e-5, atol=tolerance):
            raise ValueError('GR grid generation requires regular source coordinates.')

        phase = float(np.remainder(coordinates[0], source_spacing))
        if np.isclose(phase, 0.0, atol=tolerance) or np.isclose(phase, source_spacing, atol=tolerance):
            phase = 0.0
        lower, upper = float(coordinates.min()), float(coordinates.max())
        first_index = int(np.ceil((lower - phase - tolerance) / grid_spacing))
        last_index = int(np.floor((upper - phase + tolerance) / grid_spacing))
        if first_index > last_index:
            return np.array([], dtype=float)

        centers = phase + np.arange(first_index, last_index + 1) * grid_spacing
        centers[np.isclose(centers, lower, atol=tolerance)] = lower
        centers[np.isclose(centers, upper, atol=tolerance)] = upper
        source_order = np.argsort(coordinates)
        source_sorted = coordinates[source_order]
        nearest = np.clip(np.searchsorted(source_sorted, centers), 0, source_sorted.size - 1)
        previous = np.clip(nearest - 1, 0, source_sorted.size - 1)
        choose_previous = (np.abs(source_sorted[previous] - centers)
                           < np.abs(source_sorted[nearest] - centers))
        nearest[choose_previous] = previous[choose_previous]
        snap = np.abs(source_sorted[nearest] - centers) <= tolerance
        centers[snap] = source_sorted[nearest[snap]]
        return centers

    def _get_xy(self):
        """Get GR center coordinates from a cell-size-independent lattice."""
        if self.gr_grid_mask is not None:
            xcoords = self.gr_grid_mask[self.xdim].values
            ycoords = self.gr_grid_mask[self.ydim].values
            return xcoords, ycoords
        
        if self.ydim == 'y':
            raise ValueError('GR grid file not available for x/y grid - cannot get x and y coordinates for GR grid. '
                             'Please generate GR grid first.')
        
        if self.input_data is not None:
            ref_grid = self.input_data
        elif self.area_grid is not None:
            ref_grid = self.area_grid
        else:
            raise ValueError('No input data or area grid provided. Please provide a valid input data grid or area '
                             'grid.')

        xcoord = np.asarray(ref_grid[self.xdim].values, dtype=float)
        ycoord = np.asarray(ref_grid[self.ydim].values, dtype=float)
        xcoords = self._grid_centers_from_source_axis(xcoord, self.gr_grid_res)
        ycoords = self._grid_centers_from_source_axis(ycoord, self.gr_grid_res)
        if xcoord[0] > xcoord[-1]:
            xcoords = xcoords[::-1]
        if ycoord[0] > ycoord[-1]:
            ycoords = ycoords[::-1]
        if len(ycoords) == 0 or len(xcoords) == 0:
            raise ValueError('Not enough valid cells found for the configured GR grid spacing; check the input '
                             'data grid and static files.')
        
        return xcoords, ycoords

    def generate_gr_grid_mask(self):
        """
        generate mask for grid of GeoRegions
        """
        logger.info(f'Generating GR grid mask with resolution {self.gr_grid_res} degrees')
        # grg_res defines each AGR weighting tile around a GRG center. This is
        # separate from cell_size_y, which defines the local calculation
        # footprint used by select_sub_gr and may overlap adjacent centers.
        grg_res = self.gr_grid_res
        self.gr_grid_mask = None
        mask_orig = self.mask
        full_mask_orig = self._align_spatial_grid(self.full_mask, mask_orig)
        area_orig = self._align_spatial_grid(self.area_grid, mask_orig)
        res_orig = self._y_resolution_in
        # gr_grid_areas stores full spacing-tile areas; gr_grid_mask stores
        # altitude-filtered regional coverage; gr_grid_mask_full stores
        # pre-altitude regional coverage. AGR uses each for its respective area.
        if self.xdim == 'x':
            # try to aggregate using coarsen (vectorized)
            factor = int(round(grg_res / res_orig)) if res_orig is not None else None
            if factor is None or factor < 1:
                raise ValueError('Invalid grid resolution ratio for cartesian grid')
            coarsen_dims = {self.ydim: factor, self.xdim: factor}
            total_area = area_orig.coarsen(coarsen_dims, boundary='trim').sum(skipna=True)
            gr_grid_mask = self._coverage_fraction_on_projected_grid(
                mask_orig, area_orig, total_area, coarsen_dims, 'mask')
            gr_grid_mask_full = self._coverage_fraction_on_projected_grid(
                full_mask_orig, area_orig, total_area, coarsen_dims, 'mask_full')
            gr_grid_areas = total_area.rename('area_grid')
        else:
            xcoords, ycoords = self._get_xy()
            gr_grid_areas = xr.DataArray(data=np.ones((len(ycoords), len(xcoords))) * np.nan,
                                         coords={self.ydim: ycoords, self.xdim: xcoords},
                                         dims=(self.ydim, self.xdim), name='area_grid')
            gr_grid_mask = self._coverage_fraction_on_geographic_grid(
                mask_orig, area_orig, gr_grid_areas, grg_res, 'mask')
            gr_grid_mask_full = self._coverage_fraction_on_geographic_grid(
                full_mask_orig, area_orig, gr_grid_areas, grg_res, 'mask_full')

        self.gr_grid_mask = gr_grid_mask
        self.gr_grid_mask_full = gr_grid_mask_full
        self.gr_grid_areas = gr_grid_areas

    def _coverage_fraction_on_projected_grid(self, mask, area_grid, total_area, coarsen_dims, name):
        mask_fraction = mask.fillna(0).clip(min=0, max=1)
        in_region_area = (area_grid * mask_fraction).coarsen(
            coarsen_dims, boundary='trim').sum(skipna=True)
        return (in_region_area / total_area).where(total_area > 0).rename(name)

    def _coverage_fraction_on_geographic_grid(self, mask, area_grid, gr_grid_areas, grg_res, name):
        xcoords = gr_grid_areas[self.xdim].values
        ycoords = gr_grid_areas[self.ydim].values
        gr_grid_mask = xr.full_like(gr_grid_areas, np.nan).rename(name)
        for ycoord in ycoords:
            y_indices, y_weights = self._get_axis_overlap(
                mask[self.ydim].values, ycoord, grg_res,
                latitude=self.ydim.lower() in {'lat', 'latitude'})
            if len(y_indices) == 0:
                continue
            for xcoord in xcoords:
                x_indices, x_weights = self._get_axis_overlap(
                    mask[self.xdim].values, xcoord, grg_res)
                if len(x_indices) == 0:
                    continue

                indexers = {self.ydim: y_indices, self.xdim: x_indices}
                overlap_weights = xr.DataArray(
                    np.outer(y_weights, x_weights),
                    coords={
                        self.ydim: mask[self.ydim].isel({self.ydim: y_indices}),
                        self.xdim: mask[self.xdim].isel({self.xdim: x_indices}),
                    },
                    dims=(self.ydim, self.xdim),
                )
                cell_area = area_grid.isel(indexers) * overlap_weights
                total_area = cell_area.sum(skipna=True)
                valid_area = (cell_area * mask.isel(indexers).fillna(0).clip(min=0, max=1)).sum(skipna=True)
                if not np.isfinite(total_area).item() or total_area <= 0:
                    continue
                gr_grid_areas.loc[{self.ydim: ycoord, self.xdim: xcoord}] = total_area.item()
                gr_grid_mask.loc[{self.ydim: ycoord, self.xdim: xcoord}] = (valid_area / total_area).item()
        return gr_grid_mask

    def _calc_tea_ctp_at_ycoord(self, ycoord, xcoords=None):
        """
        calculate TEA indicators of GeoRegion grid cell for all x coords of a y coord
        Args:
            ycoord: y coordinate value
            xcoords: x coordinate values (default: get automatically)

        Returns:
            valid_cells_found: True if at least one valid cell was found, False otherwise

        """
        if xcoords is None:
            xcoords, ycoords = self._get_xy()

        valid_cells_found = False
        # step through all x coordinate values
        for i_x, xcoord in enumerate(xcoords):
            # this comment is necessary to suppress an unnecessary PyCharm warning
            # noinspection PyTypeChecker
            start_time = time.time()
            tea_sub = self.select_sub_gr(ycoord=ycoord, xcoord=xcoord)
            if tea_sub is None:
                continue

            valid_cells_found = True

            # calculate daily basis variables
            tea_sub.calc_daily_basis_vars(grid=False, gr=True)

            # calculate CTP indicators
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered in multiply')
                tea_sub.calc_annual_ctp_indicators(drop_daily_results=True)

                # set agr_results for position
                ctp_results = tea_sub.get_ctp_results(gr=True, grid=False).compute()

            self.set_ctp_results(xcoord, ycoord, ctp_results)
            end_time = time.time()
            logger.debug(f'{self.xdim} {xcoord}, {self.ydim} {ycoord} processed in {end_time - start_time} seconds')
        return valid_cells_found
