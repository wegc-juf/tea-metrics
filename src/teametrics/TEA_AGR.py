"""
Threshold Exceedance Amount (TEA) indicators Class implementation for aggregated georegions (AGR)
Based on: https://doi.org/10.48550/arXiv.2504.18964
# TODO: change doi when final version is available
Equation numbers refer to Supplementary Notes
"""
import warnings
import time

import xarray as xr
import numpy as np
from tqdm import trange

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
                 land_frac_min=0.5, cell_size_lat=2, **kwargs):
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
            cell_size_lat: size of GR grid cell in latitudinal direction (in degrees). Default: 2
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
            self._lat_resolution_in = round(abs(ref_grid.lat.values[0] - ref_grid.lat.values[1]), 4)
        else:
            self._lat_resolution_in = None
        self.gr_grid_res = gr_grid_res
        self.gr_grid_mask = None
        self.gr_grid_areas = None
        self.land_sea_mask = land_sea_mask
        self.land_frac_min = land_frac_min
        self.cell_size_lat = cell_size_lat

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

    def select_sub_gr(self, lat, lon):
        """
        select data of GeoRegion sub-cell and weight edges
        Args:
            lat: center latitude of cell (in degrees)
            lon: center longitude of cell (in degrees)

        Returns:
            cell_data: data of cell
            cell_static: static data of cell
        """

        lat_res = self._lat_resolution_in
        lat_off = self.cell_size_lat / 2
        lon_off_exact = 1 / np.cos(np.deg2rad(lat)) * lat_off
        size_exact = lon_off_exact * lat_off

        lon_off = np.round(lon_off_exact * 4, 0) / 4.
        size_real = lon_off * lat_off
        area_frac = size_real / size_exact

        if self.land_frac_min > 0:
            # get land-sea mask
            cell_lsm = self.land_sea_mask.sel(lat=slice(lat + lat_off, lat - lat_off + lat_res),
                                              lon=slice(lon - lon_off, lon + lon_off - lat_res))

            # calculate fraction covered by valid cells (land below 1500 m)
            land_frac = cell_lsm.sum().values / np.size(cell_lsm)
            if land_frac < self.land_frac_min:
                return None

        # select data for cell
        cell_data = self.daily_results.sel(lat=slice(lat + lat_off, lat - lat_off + lat_res),
                                           lon=slice(lon - lon_off, lon + lon_off - lat_res))
        # compensate rounding errors
        cell_data['DTEA'] = cell_data['DTEA'] / area_frac

        # select static data for cell
        cell_area_grid = self.area_grid.sel(lat=slice(lat + lat_off, lat - lat_off + lat_res),
                                            lon=slice(lon - lon_off, lon + lon_off - lat_res))
        cell_area_grid = cell_area_grid / area_frac

        if len(cell_area_grid.lat) == 0:
            raise ValueError('No valid cell found, check why this happens')

        # TODO: two options: either return data itself and stack to xarray then calculate TEA or return individual TEA
        # objects
        tea_sub_gr = TEAIndicators(area_grid=cell_area_grid, min_area=self._min_area, unit=self.unit, ctp=self.CTP)
        tea_sub_gr.set_daily_results(cell_data)
        return tea_sub_gr

    def set_ctp_results(self, lat, lon, ctp_results):
        """
        set CTP variables for point
        Args:
            lat: latitude
            lon: longitude
            ctp_results: CTP GR data for point
        """
        # remove GR from variable names
        ctp_results = ctp_results.rename({var: var.replace('_GR', '') for var in ctp_results.data_vars})

        if self.ctp_results is None or not len(self.ctp_results.data_vars):
            data_vars = [var for var in ctp_results.data_vars]
            var_dict = {}
            lats, lons = self._get_lats_lons()
            for var in data_vars:
                var_dict[var] = (['time', 'lat', 'lon'], np.nan * np.ones((len(ctp_results.time),
                                                                           len(lats),
                                                                           len(lons))))
            self.ctp_results = xr.Dataset(coords=dict(time=ctp_results.time,
                                                      lon=lons,
                                                      lat=lats),
                                          data_vars=var_dict,
                                          attrs=ctp_results.attrs)

        self.ctp_results.loc[dict(lat=lat, lon=lon)] = ctp_results

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
            self.ctp_results.to_netcdf(filepath)

    def _apply_gr_grid_mask(self):
        """
        apply GR grid mask to CTP results
        """
        if self.ctp_results is not None:
            self.ctp_results = self.ctp_results.where(self.gr_grid_mask > 0)

    def calc_annual_ctp_indicators(self, ctp=None, drop_daily_results=False, lats=None, lons=None):
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
            lats: Latitudes (default: get automatically)
            lons: Longitudes (default: get automatically)

        Returns:

        """
        if ctp is not None:
            self._set_ctp(ctp)

        self._calc_annual_gr_grid(lats=lats, lons=lons)
        self._apply_gr_grid_mask()

        if drop_daily_results:
            self.daily_results.close()
            del self._daily_results_filtered
            del self.daily_results

    def _calc_annual_gr_grid(self, lats=None, lons=None):
        """
        calculate annual CTP TEA indicators for all GeoRegions in GeoRegion grid
        Args:
            lats: Latitudes (default: get automatically)
            lons: Longitudes (default: get automatically)

        Returns:

        """
        if lats is None:
            lats, lons = self._get_lats_lons()

        valid_cells_found = False
        for ilat in trange(len(lats), desc='Processing AGR cells'):
            lat = lats[ilat]
            valid_cells_found |= self._calc_tea_ctp_lat(lat, lons=lons)
        if not valid_cells_found:
            logger.error('No valid cells found for annual CTP calculation. Try to decrease the land_frac_min '
                         'parameter or check the region definition. ')

    def _crop_to_shp(self):
        """
        crop GeoRegion grid data to spatial extent of AGR shape

        Returns:

        """
        self.gr_grid_areas = self.gr_grid_areas.where(self.gr_grid_mask > 0)
        self._ref_mean = self._ref_mean.where(self.gr_grid_mask > 0)
        self.decadal_results = self.decadal_results.where(self.gr_grid_mask > 0)
        self.amplification_factors = self.amplification_factors.where(self.gr_grid_mask > 0)
        self._crop_to_gr_mask_extents()

    def _crop_to_gr_mask_extents(self):
        """
        crop GeoRegion grid data to spatial extent of aggregated GeoRegion mask

        Returns:

        """
        mask_tmp = self.mask
        self.mask = self.gr_grid_mask
        self._crop_to_mask_extents()
        self.mask = mask_tmp

    def _crop_to_rect(self, lat_range, lon_range):
        """
        crop GeoRegion grid data to spatial extent of aggregated GeoRegion
        Args:
            lat_range: Latitude range (min, max)
            lon_range: Longitude range (min, max)

        Returns:

        """
        if lat_range is None:
            lat_range = (self.gr_grid_areas.lat.min(), self.gr_grid_areas.lat.max())
        if lon_range is None:
            lon_range = (self.gr_grid_areas.lon.min(), self.gr_grid_areas.lon.max())

        self.gr_grid_areas = self.gr_grid_areas.sel(lat=slice(lat_range[1], lat_range[0]),
                                                    lon=slice(lon_range[0], lon_range[1]))
        self._ref_mean = self._ref_mean.sel(lat=slice(lat_range[1], lat_range[0]),
                                            lon=slice(lon_range[0], lon_range[1]))
        self.decadal_results = self.decadal_results.sel(lat=slice(lat_range[1], lat_range[0]),
                                                        lon=slice(lon_range[0], lon_range[1]))
        self.amplification_factors = self.amplification_factors.sel(lat=slice(lat_range[1], lat_range[0]),
                                                                    lon=slice(lon_range[0], lon_range[1]))

    def _drop_agr_values_and_spreads(self):
        """
        drop old AGR values and spreads
        """
        agr_vars = [var for var in self.decadal_results.data_vars if 'AGR' in var or 'supp' in var or 'slow' in var]
        self.decadal_results = self.decadal_results.drop_vars(agr_vars)
        agr_vars = [var for var in self.amplification_factors.data_vars if 'AGR' in var or 'supp' in var or 'slow' in
                    var]
        self.amplification_factors = self.amplification_factors.drop_vars(agr_vars)
        agr_vars = [var for var in self._ref_mean.data_vars if 'AGR' in var or 'supp' in var or 'slow' in var]
        self._ref_mean = self._ref_mean.drop_vars(agr_vars)
        agr_vars = [var for var in self._cc_mean.data_vars if 'AGR' in var or 'supp' in var or 'slow' in var]
        self._cc_mean = self._cc_mean.drop_vars(agr_vars)

    @staticmethod
    def _calc_weighted_perc(data, areas):
        """
        calculate weighted percentiles
        :param data: data
        :param areas: area sizes
        :return:
        """
        # remove values where data is nan
        areas_3d = xr.full_like(data, 1) * areas
        areas_3d = areas_3d.where(np.isfinite(data))

        wgts = areas_3d / areas_3d.sum(dim=('lat', 'lon'))

        if 'time' not in data.dims:
            pval005, pval095 = TEAAgr._calc_weighted_perc_single(data.values, wgts.values)
            return pval005, pval095

        perc005 = xr.DataArray(data=np.ones(len(data.time)) * np.nan,
                               coords={'time': (['time'], data.time.data)})
        perc095 = xr.DataArray(data=np.ones(len(data.time)) * np.nan,
                               coords={'time': (['time'], data.time.data)})

        for iyr, yr in enumerate(data.time):
            if iyr < 5:
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
        wgts_cumsum = np.cumsum(wgts_ordered)
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
                while wgts_cumsum[p095_index] < 0.95:
                    p095_index += 1
            pval095 = avals_ordered[p095_index]
        except ValueError:
            pval095 = np.nan
        return pval005, pval095

    def calc_agr_vars(self, lat_range=None, lon_range=None, spreads=True, crop_to_shp=False):
        """
        calculate AGR variables

        Args:
            lat_range: Latitude range (min, max). Default: Full region
            lon_range: Longitude range (min, max). Default: Full region
            spreads: if True, calculate spreads and percentiles for AGR variables
            crop_to_shp: if True, crop data to shape of aggregated GeoRegion (default: False)
        """
        # filter data to spatial extent of aggregated GeoRegion
        if lat_range is not None or lon_range is not None:
            self._crop_to_rect(lat_range=lat_range, lon_range=lon_range)
        elif crop_to_shp:
            self._crop_to_shp()

        # drop old AGR values and spreads
        self._drop_agr_values_and_spreads()

        # calculate area weights (equation 34_0)
        if self.gr_grid_areas is None:
            raise ValueError('No GR area grid provided. Please provide a valid GR area grid.')
        A_AGR = self.gr_grid_areas.sum()
        awgts = self.gr_grid_areas / A_AGR

        # calc X_Ref^AGR and X_s^AGR (equation 34_1 and equation 34_2)
        x_ref_agr = (awgts * self._ref_mean).sum()
        xt_s_agr = (awgts * self.decadal_results).sum(dim=('lat', 'lon'))

        # calc Xt_ref_db and Xt_ref_agr (equation 34_3)
        xt_ref_agr = self._calc_gmean_decadal(start_year=self.ref_period[0], end_year=self.ref_period[1], data=xt_s_agr)

        # calculate X_s_AGR (equation 34_4)
        x_s_agr = (x_ref_agr / xt_ref_agr) * xt_s_agr

        # calculate compound variables (equation 35)
        x_s_agr = self._calc_compound_vars(x_s_agr)
        x_ref_agr = self._calc_compound_vars(x_ref_agr)

        # set values to nan for first and last 5/4 years
        x_s_agr[dict(time=slice(0, 5))] = np.nan
        x_s_agr[dict(time=slice(-4, None))] = np.nan

        # calculate spread estimates (equation 38)
        if spreads:
            x_s_spreads = self._calc_agr_spread(data=self.decadal_results, ref=x_s_agr)
        else:
            x_s_spreads = xr.Dataset()

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
            # calculate spread estimates (equation 38)
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
        r_earth = 6371
        u_earth = 2 * np.pi * r_earth
        # # size of grid cell in 100 km^2
        A_GR_full = (u_earth / 360 * self.cell_size_lat) ** 2 / 100
        N_dof = int(A_AGR / A_GR_full)

        if spreads:
            # add p5 and p95 values (equation 41TODEFINE)
            # TODO: optimize this
            self._calc_agr_percentiles(data=self.decadal_results)
            self._calc_agr_percentiles(data=self.amplification_factors)

        # add attributes
        for vvar in af_agr.data_vars:
            af_agr[vvar].attrs = get_attrs(vname=vvar, data_unit=self.unit)
        for vvar in af_cc_agr.data_vars:
            af_cc_agr[vvar].attrs = get_attrs(vname=vvar, data_unit=self.unit)

        # join af_agr and af_cc_agr
        af_agr = xr.merge([af_agr, af_cc_agr, af_spreads, af_cc_spreads])
        af_agr = self._duplicate_vars(af_agr)

        # rename variables
        rename_dict = {var: f'{var}_AGR' for var in x_s_agr.data_vars}
        x_s_agr = x_s_agr.rename(rename_dict)
        rename_dict = {var: var.replace('AGR', 'AGR_CC') for var in x_cc_spreads.data_vars}
        x_cc_spreads = x_cc_spreads.rename(rename_dict)
        rename_dict = {var: var.replace('AGR', 'AGR_ref') for var in x_ref_spreads.data_vars}
        x_ref_spreads = x_ref_spreads.rename(rename_dict)

        # add attributes
        for vvar in x_s_agr.data_vars:
            x_s_agr[vvar].attrs = get_attrs(vname=vvar, data_unit=self.unit)
        x_s_agr = xr.merge([x_s_agr, x_s_spreads])

        # add number of degrees of freedom for AGR mean
        x_s_agr['N_dof_AGR'] = N_dof
        x_s_agr['N_dof_AGR'].attrs = {'long_name': 'Number of degrees of freedom for AGR mean'}
        af_agr['N_dof_AGR'] = N_dof
        af_agr['N_dof_AGR'].attrs = {'long_name': 'Number of degrees of freedom for AGR mean'}

        self.decadal_results = xr.merge([x_s_agr, x_ref_spreads, x_cc_spreads, self.decadal_results], compat='override')
        self.amplification_factors = xr.merge([af_agr, self.amplification_factors], compat='override')
        # self.amplification_factors = xr.merge([self.amplification_factors, af_agr], compat='override')

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
        c_upp_sum = (c_upp * areas).sum(dim=('lat', 'lon'))
        # replace 0 values with nan to avoid division by zero
        c_upp_sum = c_upp_sum.where(c_upp_sum > 0)
        s_upp = np.sqrt(1 / c_upp_sum * (c_upp * areas * (data - ref) ** 2).sum(dim=('lat', 'lon')))

        # equation 38_3 and 38_6
        c_low_sum = ((1 - c_upp) * areas).sum(dim=('lat', 'lon'))
        # replace 0 values with nan to avoid division by zero
        c_low_sum = c_low_sum.where(c_low_sum > 0, c_low_sum)
        s_low = np.sqrt(1 / c_low_sum * ((1 - c_upp) * areas * (data - ref) ** 2).sum(dim=('lat', 'lon')))

        # get attributes for variables and rename them
        for vvar in s_upp.data_vars:
            s_upp[vvar].attrs = get_attrs(vname=f'{vvar}_AGR', spread='upper')
        for vvar in s_low.data_vars:
            s_low[vvar].attrs = get_attrs(vname=f'{vvar}_AGR', spread='lower')
        rename_dict_up = {var: f'{var}_AGR_supp' for var in s_upp.data_vars}
        rename_dict_low = {var: f'{var}_AGR_slow' for var in s_low.data_vars}
        s_upp = s_upp.rename(rename_dict_up)
        s_low = s_low.rename(rename_dict_low)

        return xr.merge([s_upp, s_low])

    def _calc_agr_percentiles(self, data):
        """
        calculate spread estimates of grid cell values around AGR mean (equation 38)
        Args:
            data: data for spread estimation
            ref: reference data for spread estimation
        Returns:
            xarray dataset with upper and lower spread estimates

        """
        logger.info(f'Calculating 5th and 95th percentiles')
        areas = self.gr_grid_areas

        for var in data.data_vars:
            # calculate 5th and 95th percentiles for each variable

            # standard method (gives only slightly different results to weighted method)
            # p5_std = data[var].quantile(0.05, dim=('lat', 'lon'))
            # p95_std = data[var].quantile(0.95, dim=('lat', 'lon'))

            p5, p95 = self._calc_weighted_perc(data[var], areas)

            if 'AF' in var:
                var = var.replace('AF', 'AGR_AF')
            else:
                var = f'{var}_AGR'
            data[f'{var}_p05'] = p5
            data[f'{var}_p95'] = p95
            data[f'{var}_p05'].attrs = get_attrs(vname=f'{var}', spread='5th percentile')
            data[f'{var}_p95'].attrs = get_attrs(vname=f'{var}', spread='95th percentile')

    def _get_lats_lons(self, margin=None):
        """
        get latitudes and longitudes for GeoRegion grid
        """
        if margin is None:
            margin = self.cell_size_lat / 2

        if self.input_data is not None:
            ref_grid = self.input_data
        elif self.area_grid is not None:
            ref_grid = self.area_grid
        else:
            raise ValueError('No input data or area grid provided. Please provide a valid input data grid or area '
                             'grid.')

        lats = np.arange(ref_grid.lat.max() - margin,
                         ref_grid.lat.min() - self.gr_grid_res + margin,
                         -self.gr_grid_res)
        margin_lon = 1 / np.cos(np.deg2rad(lats[0])) * margin
        # round margin_lon to resolution of gr grid
        margin_lon = np.round(np.round(margin_lon / self.gr_grid_res, 0) * self.gr_grid_res, 2)
        lons = np.arange(ref_grid.lon.min() + margin_lon,
                         ref_grid.lon.max() + self.gr_grid_res - margin_lon,
                         self.gr_grid_res)
        if len(lats) == 0 or len(lons) == 0:
            raise ValueError(f'Not enough valid cells found for margin {margin} - check size of input data grid and '
                             f'static files')
        return lats, lons

    def generate_gr_grid_mask(self):
        """
        generate mask for grid of GeoRegions
        """
        logger.info(f'Generating GR grid mask with resolution {self.gr_grid_res} degrees')
        grg_res = self.gr_grid_res
        lats, lons = self._get_lats_lons()
        mask_orig = self.mask
        area_orig = self.area_grid
        res_orig = self._lat_resolution_in

        gr_grid_mask = xr.DataArray(data=np.ones((len(lats), len(lons))) * np.nan,
                                    coords={'lat': (['lat'], lats), 'lon': (['lon'], lons)},
                                    dims={'lat': (['lat'], lats), 'lon': (['lon'], lons)})
        gr_grid_mask = gr_grid_mask.rename('mask')

        gr_grid_areas = xr.DataArray(data=np.ones((len(lats), len(lons))) * np.nan,
                                     coords={'lat': (['lat'], lats), 'lon': (['lon'], lons)},
                                     dims={'lat': (['lat'], lats), 'lon': (['lon'], lons)})
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

    def _calc_tea_ctp_lat(self, lat, lons=None):
        """
        calculate TEA indicators of GeoRegion grid cell for all longitudes of a latitude
        Args:
            lat: Latitude
            lons: Longitudes (default: get automatically)

        Returns:
            valid_cells_found: True if at least one valid cell was found, False otherwise

        """
        if lons is None:
            lats, lons = self._get_lats_lons()

        valid_cells_found = False
        # step through all longitudes
        for ilon, lon in enumerate(lons):
            # this comment is necessary to suppress an unnecessary PyCharm warning for lon
            # noinspection PyTypeChecker
            start_time = time.time()
            tea_sub = self.select_sub_gr(lat=lat, lon=lon)
            if tea_sub is None:
                continue

            valid_cells_found = True

            # calculate daily basis variables
            tea_sub.calc_daily_basis_vars(grid=False, gr=True)

            # calculate CTP indicators
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered in multiply')
                tea_sub.calc_annual_ctp_indicators(drop_daily_results=True)

                # set agr_results for lat and lon
                ctp_results = tea_sub.get_ctp_results(gr=True, grid=False).compute()

            self.set_ctp_results(lat, lon, ctp_results)
            end_time = time.time()
            logger.debug(f'Lat {lat}, lon {lon} processed in {end_time - start_time} seconds')
        return valid_cells_found
