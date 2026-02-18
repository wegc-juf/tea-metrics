"""
Plot threshold map
"""

from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfea
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.patches as pat
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
from shapely import geometry
import xarray as xr

STATIC_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/')
MASKS_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/')


def plot_eur():
    """
    ExtDataFig 1a
    :return:
    """
    # TODO: adjust input
    thr = xr.open_dataset(STATIC_PATH / 'static_Tx99.0p_EUR_ERA5.nc')

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
    # TODO: adjust labels
    im = thr.plot.imshow(ax=axs, transform=ccrs.PlateCarree(), colors=cmap, vmin=10, vmax=40,
                         levels=levels, cbar_kwargs={'label': 'Ref-p99ANN Temperature (°C)',
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
    # TODO: adjust labels
    axs.text(0.02, 0.97, 'ERA5-TMax-Ref1961-1990', horizontalalignment='left',
             verticalalignment='center', transform=axs.transAxes, backgroundcolor='whitesmoke',
             fontsize=10)
    # TODO: adjust outname
    plt.savefig('./ExtDataFig1a.png', bbox_inches='tight', dpi=300)


def plot_era5land():
    """
    ExtDataFig 1 b
    :return:
    """
    # TODO: adjust input
    thr = xr.open_dataset(STATIC_PATH / 'static_Tx99.0p_AUT_ERA5Land.nc')
    aut = xr.open_dataset(MASKS_PATH / 'AUT_masks_ERA5Land.nc')

    thr = thr.where(aut.nw_mask == 1)

    fig = plt.figure(figsize=(5, 3))
    proj = ccrs.LambertConformal(central_longitude=13.5, central_latitude=53.5, cutoff=30)
    axs = plt.axes(projection=proj)
    axs.contourf(aut.lon, aut.lat, aut.nw_mask, colors='gainsboro', transform=ccrs.PlateCarree())
    vals = axs.contourf(thr.lon, thr.lat, thr.threshold, levels=np.arange(19, 34, 1),
                        transform=ccrs.PlateCarree(), cmap='Reds')
    axs.add_feature(cfea.BORDERS)
    axs.coastlines()

    # TODO: adjust labels
    cb = plt.colorbar(vals, pad=0.03, shrink=0.84, label='Ref-p99ANN Temperature (°C)')

    gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='black', linestyle=':',
                       x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.rotate_labels = False


    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95)

    fig.text(0.06, 0.205, 'Data at z > 1500m excluded.',
             horizontalalignment='left', verticalalignment='center',
             backgroundcolor='gainsboro',
             fontsize=6)

    # TODO: adjust labels
    axs.text(0.02, 0.93, 'ERA5L-TMax-Ref1961-1990', horizontalalignment='left',
             verticalalignment='center', transform=axs.transAxes, backgroundcolor='whitesmoke',
             fontsize=10)
    # TODO: adjust outname
    plt.savefig('./threshold.png', bbox_inches='tight', dpi=300)


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
    plot_eur()
    plot_era5land()
    plot_spartacus()
