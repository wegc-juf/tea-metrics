"""
Plot threshold map
"""

import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfea
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import seaborn as sns
import xarray as xr

from config import load_opts


def _getopts():
    """
    get command line arguments

    Returns:
        opts: command line parameters
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')

    # parser.add_argument('--version', '-v',
    #                     action='version',
    #                     version=TEA_VERSION,
    #                     help='show version and exit')

    myopts = parser.parse_args()

    return myopts

def plot_eur(opts):
    """
    plot EUR map of threshold grid
    Args:
        opts: CLI parameter

    Returns:

    """
    thr = xr.open_dataset(f'{opts.statpath}static_{opts.param_str}_EUR_{opts.dataset}.nc')

    thr = thr.threshold
    thr = thr.sel(lat=slice(72, 35), lon=slice(-10, 40))

    # move longitude half a pixel eastward
    pix_size = thr.lon[1] - thr.lon[0]
    thr = thr.assign_coords(lon=thr.lon + pix_size / 2)

    levels = np.arange(10, 42.5, 2.5)
    cmap = sns.color_palette('Reds', len(levels))

    fig = plt.figure(figsize=(10, 7))
    proj = ccrs.LambertConformal(central_longitude=13.5, central_latitude=53.5, cutoff=30)
    axs = plt.axes(projection=proj)
    im = thr.plot.imshow(ax=axs, transform=ccrs.PlateCarree(), colors=cmap, vmin=10, vmax=40,
                         levels=levels, cbar_kwargs={'label': f'Ref-{opts.param_str} ({opts.unit})',
                                                     'ticks': np.arange(10, 45, 5)})
    axs.set_title('')

    axs.add_feature(cfea.BORDERS)
    axs.coastlines()

    gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='black', linestyle=':',
                       x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-10, 0, 10, 20, 30, 40])
    gl.ylocator = mticker.FixedLocator([35, 45, 55, 65, 75])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    axs.set_extent([-10, 40, 30, 75])
    axs.text(0.02, 0.97, f'{opts.dataset}-{opts.param_str}-Ref{opts.ref_period[0]}-{opts.ref_period[1]}',
             horizontalalignment='left', verticalalignment='center', transform=axs.transAxes,
             backgroundcolor='whitesmoke', fontsize=10)

    plt.savefig(f'{opts.outpath}/plots/threshold-map_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                f'_{opts.start}to{opts.end}.png', bbox_inches='tight', dpi=300)


def plot_single_country(opts):
    """
    plot threshold map for ERA5, ERA5-Land, ERA5-Heat, and EOBS
    Returns:

    """
    thr = xr.open_dataset(f'{opts.statpath}static_{opts.param_str}_{opts.region}_{opts.dataset}.nc')
    cntry = xr.open_dataset(f'{opts.maskpath}{opts.mask_sub}{opts.region}_masks_{opts.dataset}.nc')

    thr = thr.where(cntry.nw_mask == 1)

    cntry_coords = cntry.where(cntry.nw_mask == 1, drop=True)
    cen_lon = cntry_coords.lon.min() + (cntry_coords.lon.max() - cntry_coords.lon.min()) / 2
    cen_lat = cntry_coords.lat.min() + (cntry_coords.lat.max() - cntry_coords.lat.min()) / 2

    fig = plt.figure(figsize=(5, 3))
    proj = ccrs.LambertConformal(central_longitude=cen_lon.values, central_latitude=cen_lat.values)
    axs = plt.axes(projection=proj)
    axs.contourf(cntry.lon, cntry.lat, cntry.nw_mask, colors='gainsboro', transform=ccrs.PlateCarree())
    vals = axs.contourf(thr.lon, thr.lat, thr.threshold,
                        levels=np.arange(np.floor(thr.threshold.min().values),
                                         np.ceil(thr.threshold.max().values), 1),
                        transform=ccrs.PlateCarree(), cmap='Reds')
    axs.add_feature(cfea.BORDERS)
    axs.coastlines()

    cb = plt.colorbar(vals, pad=0.03, shrink=0.84, label=f'Ref-{opts.param_str} ({opts.unit})')

    gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='black', linestyle=':',
                       x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.rotate_labels = False

    lon_min, lon_max = np.floor(cntry_coords.lon.min().values / 2) * 2, np.ceil(cntry_coords.lon.max().values / 2) * 2
    lat_min, lat_max = np.floor(cntry_coords.lat.min().values / 2) * 2, np.ceil(cntry_coords.lat.max().values / 2) * 2
    axs.set_extent([lon_min, lon_max, lat_min, lat_max])

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95)

    axs.set_title(f'{opts.dataset}-{opts.param_str}-Ref{opts.ref_period[0]}-{opts.ref_period[1]}', fontsize=12)

    plt.savefig(f'{opts.outpath}/plots/threshold-map_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                f'_{opts.start}to{opts.end}.png', bbox_inches='tight', dpi=300)


def plot_spartacus():
    """
    ExtDataFig 1 c & d
    :return:
    """
    param = ['Tx', 'P24h_7to7']
    # TODO: adjust labels
    props = {'Tx': {'levels': np.arange(19, 34, 1), 'cb_lbl': 'Ref-p99ANN Temperature (°C)',
                    'cmap': 'Reds', 'ext': 'neither', 'pstr': 'Tx99.0p'},
             'P24h_7to7': {'levels': np.arange(18, 44, 2),
                           'cb_lbl': 'Ref-p95WAS Precipitation (mm)', 'cmap': 'Blues',
                           'ext': 'neither', 'pstr': 'P24h_7to7_95.0p'}}

    for par in param:
        # TODO: adjust input
        thr = xr.open_dataset(STATIC_PATH / f'static_{props[par]["pstr"]}_SEA_SPARTACUS.nc')
        thr = thr.threshold

        fig, axs = plt.subplots(1, 1, figsize=(4.5, 3))
        perc = axs.contourf(thr, cmap=props[par]['cmap'], levels=props[par]['levels'],
                            extend=props[par]['ext'])

        divider = make_axes_locatable(axs)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(perc, cax=cax, orientation='vertical', extend=props[par]['ext'],
                          shrink=0.83, ticks=props[par]['levels'][::2])
        cb.set_label(props[par]['cb_lbl'])

        # TODO: adjust labels
        if par == 'Tx':
            title = 'SPCUS-TMax-Ref1961-1990'
        else:
            title = 'SPCUS-P24H-Ref1961-1990'
        axs.set_title(title, fontsize=12)

        axs.axis('off')

        # TODO: adjust outname
        plt.savefig(f'./threshold.png', bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == '__main__':
    cmd_opts = _getopts()
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)
    plt_outpath = f'{opts.outpath}/plots'
    if not os.path.exists(plt_outpath):
        os.makedirs(plt_outpath)

    if opts.region == 'EUR': # or opts.agr == 'EUR'
        plot_eur(opts=opts)
    elif opts.dataset != 'SPARTACUS':
        plot_single_country(opts=opts)
    else:
        plot_spartacus()
