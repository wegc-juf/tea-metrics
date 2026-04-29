"""
Threshold Exceedance Amount (TEA) indicators Class implementation for aggregated georegions (AGR)
Based on: https://doi.org/10.1016/j.wace.2026.100855
Equation numbers refer to Supplementary Notes therin
"""
import warnings
import time
from typing import Any

import xarray as xr
import numpy as np
from tqdm import trange
from xarray import Dataset

from .common.var_attrs import get_attrs
from .common.TEA_logger import logger
from .TEA import TEAIndicators


class TEAAgr(TEAIndicators):
    """
    Class for Threshold Exceedance Amount (TEA) indicators for aggregated georegions (AGR)

    as defined in https://doi.org/10.48550/arXiv.2504.18964 and
    Methods as defined in
    Kirchengast, G., Haas, S. J. & Fuchsberger, J. Compound event metrics detect and explain ten-fold
    increase of extreme heat over Europe—Supplementary Note: Detailed methods description for
    computing threshold-exceedance-amount (TEA) indicators. Supplementary Information (SI) to
    Preprint – April 2025. 40 pp. Wegener Center, University of Graz, Graz, Austria, 2025.
    """
    def __init__(self, input_data=None, threshold=None, mask=None, min_area=0.0001,
                 gr_grid_res=0.5, land_sea_mask=None, gr_grid_mask=None, gr_grid_areas=None,
                 land_frac_min=0.25, cell_size_y=2, **kwargs):
        """
        initialize TEA object

        Args:
            input_data: input data grid
            threshold: threshold grid
            mask: mask grid (needed if data should be masked out by e.g. country borders)
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
        self.gr_grid_areas = None
        self.land_sea_mask = land_sea_mask
        self.land_frac_min = land_frac_min
        self.cell_size_y = cell_size_y

        self.gr_grid_mask = gr_grid_mask
        self.gr_grid_areas = gr_grid_areas

        # filter input data to valid cells
        if self.land_sea_mask is not None and self.input_data is not None:
            self.input_data = self.input_data.where(self.land_sea_mask > 0)

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

        y_res = self._y_resolution_in
        y_off = self.cell_size_y / 2
        
        if self.ydim == 'lat':
            lon_off_exact = 1 / np.cos(np.deg2rad(ycoord)) * y_off
            size_exact = lon_off_exact * y_off

            x_off = np.round(lon_off_exact * 4, 0) / 4.
            size_real = x_off * y_off
            area_frac = size_real / size_exact
        else:
            x_off = y_off
            area_frac = 1
            
        # take care of y coordinate order (different for lat/lon and y/x grids)
        if self.daily_results[self.ydim][0] > self.daily_results[self.ydim][-1]:
            slice_y = slice(ycoord + y_off, ycoord - y_off + y_res)
        else:
            slice_y = slice(ycoord - y_off + y_res, ycoord + y_off)
        slice_x = slice(xcoord - x_off, xcoord + x_off - y_res)
        
        if self.land_frac_min > 0:
            # get land-sea mask
            cell_lsm = self.land_sea_mask.sel({self.ydim: slice_y, self.xdim: slice_x})

            # calculate fraction covered by valid cells (land below 1500 m)
            land_frac = cell_lsm.sum().values / np.size(cell_lsm)
            if land_frac < self.land_frac_min:
                return None

        # select data for cell
        cell_data = self.daily_results.sel({self.ydim: slice_y, self.xdim: slice_x})
        # select static data for cell
        cell_area_grid = self.area_grid.sel({self.ydim: slice_y, self.xdim: slice_x})

        # compensate rounding errors
        cell_data['DTEA'] = cell_data['DTEA'] / area_frac
        cell_area_grid = cell_area_grid / area_frac
        
        # select threshold grid for cell
        cell_threshold_grid = self.threshold_grid.sel({self.ydim: slice_y, self.xdim: slice_x})
        
        if len(cell_area_grid[self.ydim]) == 0 or len(cell_area_grid[self.xdim]) == 0:
            return None

        # TODO: optimize for x y grids (xarray method)
        # two options: either return data itself and stack to xarray then calculate TEA or return individual TEA objects
        tea_sub_gr = TEAIndicators(area_grid=cell_area_grid, min_area=self._min_area, unit=self.unit, ctp=self.CTP,
                                   threshold=cell_threshold_grid)
        tea_sub_gr.set_daily_results(cell_data)
        return tea_sub_gr

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

    def save_ctp_results(self, filepath):
        """
        save all CTP results to filepath
        """
        with warnings.catch_warnings():
            # ignore warnings due to nan multiplication
            warnings.simplefilter("ignore")
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
        self.gr_grid_areas = self.gr_grid_areas.where(self.gr_grid_mask > 0)
        self._crop_to_gr_mask_extents()
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

        # calc X_Ref^AGR and X_s^AGR (equation 34_1 and equation 34_2)
        x_ref_agr = self.calc_area_weighted_mean(self.gr_grid_areas, self._ref_mean)
        xt_s_agr = self.calc_area_weighted_mean(self.gr_grid_areas, self.decadal_results)
        if calc_annual:
            xt_p_agr = self.calc_area_weighted_mean(self.gr_grid_areas, self.ctp_results)

        # calc Xt_ref_agr (equation 34_3)
        xt_ref_agr = self._calc_gmean_decadal(start_year=self.ref_period[0], end_year=self.ref_period[1], data=xt_s_agr)
        if calc_annual:
            # TODO: check if geometric mean should be used here, in case of arithmetic mean the fraction below seems
            #  to be always 1 probably because of linear behavior
            xt_p_ref_agr = xt_p_agr.sel(
                time=slice(f'{self.ref_period[0]}-01-01', f'{self.ref_period[1]}-12-31')).mean(dim='time')

        # calculate X_s_AGR (equation 34_4)
        x_s_agr = (x_ref_agr / xt_ref_agr) * xt_s_agr
        if calc_annual:
            # TODO: check if we should use xp_ref_agr and xt_p_ref_agr here when description in SI is ready
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
        A_AGR = self.gr_grid_areas.sum()
        N_dof = self._get_N_dof(A_AGR)
        
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
    def _get_N_dof(self, A_AGR) -> float | Any:
        # calculate error estimates for AGR mean (equation 42TODEFINE)
        r_earth = 6371
        u_earth = 2 * np.pi * r_earth
        # # size of grid cell in 100 km^2
        if self.xdim == 'lon':
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
        area = area.where(~np.isnan(data.threshold_avg))
        awgts = area / area.sum(dim=(self.ydim, self.xdim))
        result = (awgts * data).sum(dim=(self.ydim, self.xdim))
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

        areas = self.gr_grid_areas

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
        c_low_sum = c_low_sum.where(c_low_sum > 0, c_low_sum)
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
        areas = self.gr_grid_areas

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

    def _get_xy(self, margin=None):
        """
        get x and y coords for GeoRegion grid
        """
        if self.gr_grid_mask is not None:
            xcoords = self.gr_grid_mask[self.xdim].values
            ycoords = self.gr_grid_mask[self.ydim].values
            return xcoords, ycoords
        
        if self.ydim == 'y':
            raise ValueError('GR grid file not available for x/y grid - cannot get x and y coordinates for GR grid. '
                             'Please generate GR grid first.')
        
        if margin is None:
            margin = self.cell_size_y / 2

        if self.input_data is not None:
            ref_grid = self.input_data
        elif self.area_grid is not None:
            ref_grid = self.area_grid
        else:
            raise ValueError('No input data or area grid provided. Please provide a valid input data grid or area '
                             'grid.')

        xcoord = ref_grid[self.xdim]
        ycoord = ref_grid[self.ydim]

        if ycoord[0] > ycoord[-1]:
            ycoords = np.arange(ycoord.max().values - margin,
                                ycoord.min().values - self.gr_grid_res + margin,
                                -self.gr_grid_res)
        else:
            ycoords = np.arange(ycoord.min().values + margin,
                                ycoord.max().values + self.gr_grid_res - margin,
                                self.gr_grid_res)

        # compute x margin taking y coord of first grid row
        y_0 = ycoords[0] if len(ycoords) > 0 else ycoord.max().values
        margin_lon = 1 / np.cos(np.deg2rad(y_0)) * margin
        # round margin_lon to resolution of gr grid
        margin_lon = np.round(np.round(margin_lon / self.gr_grid_res, 0) * self.gr_grid_res, 2)
        
        xcoords = np.arange(xcoord.min().values + margin_lon,
                            xcoord.max().values + self.gr_grid_res - margin_lon,
                            self.gr_grid_res)
        if len(ycoords) == 0 or len(xcoords) == 0:
            raise ValueError(f'Not enough valid cells found for margin {margin} - check size of input data grid and '
                             f'static files')
        
        return xcoords, ycoords

    def generate_gr_grid_mask(self):
        """
        generate mask for grid of GeoRegions
        """
        logger.info(f'Generating GR grid mask with resolution {self.gr_grid_res} degrees')
        grg_res = self.gr_grid_res
        mask_orig = self.mask
        area_orig = self.area_grid
        res_orig = self._y_resolution_in

        if self.xdim == 'x':
            # try to aggregate using coarsen (vectorized)
            factor = int(round(grg_res / res_orig)) if res_orig is not None else None
            if factor is None or factor < 1:
                raise ValueError('Invalid grid resolution ratio for cartesian grid')
            # mask_orig is expected to have dims (y, x)
            mask_valid = (mask_orig > 0).astype(int)
            # sum valid cells and area per block
            valid_count = mask_valid.coarsen({self.ydim: factor, self.xdim: factor}, boundary='trim').sum()
            cell_area_sum = area_orig.coarsen({self.ydim: factor, self.xdim: factor}, boundary='trim').sum()
            total_cells = factor * factor
            vcell_frac = valid_count / total_cells
            cell_area_sum = cell_area_sum.where(vcell_frac > 0)  # set area to nan for cells with no valid original
            
            gr_grid_mask = vcell_frac.rename('mask')
            gr_grid_areas = cell_area_sum.rename('area_grid')
        else:
            xcoords, ycoords = self._get_xy()
            gr_grid_mask = xr.DataArray(data=np.ones((len(ycoords), len(xcoords))) * np.nan,
                                        coords={self.ydim: ([self.ydim], ycoords), self.xdim: ([self.xdim], xcoords)},
                                        dims={self.ydim: ([self.ydim], ycoords), self.xdim: ([self.xdim], xcoords)})
            gr_grid_mask = gr_grid_mask.rename('mask')

            gr_grid_areas = xr.DataArray(data=np.ones((len(ycoords), len(xcoords))) * np.nan,
                                         coords={self.ydim: ([self.ydim], ycoords), self.xdim: ([self.xdim], xcoords)},
                                         dims={self.ydim: ([self.ydim], ycoords), self.xdim: ([self.xdim], xcoords)})
            gr_grid_areas = gr_grid_areas.rename('area_grid')

            for llat in gr_grid_mask.lat:
                for llon in gr_grid_mask.lon:
                    cell_orig = mask_orig.sel(lat=slice(llat, llat - grg_res + res_orig),
                                              lon=slice(llon, llon + grg_res - res_orig))
                    cell_area = area_orig.sel(lat=slice(llat, llat - grg_res + res_orig),
                                              lon=slice(llon, llon + grg_res - res_orig))
                    valid_cells = cell_orig.sum()
                    if valid_cells == 0:
                        continue
                    vcell_frac = valid_cells / cell_orig.size
                    gr_grid_mask.loc[llat, llon] = vcell_frac.values
                    gr_grid_areas.loc[llat, llon] = cell_area.sum().values

        self.gr_grid_mask = gr_grid_mask
        self.gr_grid_areas = gr_grid_areas

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
