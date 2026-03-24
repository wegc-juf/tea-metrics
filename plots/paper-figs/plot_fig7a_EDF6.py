"""
Plot Figure 7a and Extended Data Figure 6
"""

from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfea
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from shapely import geometry
import xarray as xr

INPUT_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/results/')
STATIC_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/static/')


def create_cmap_tex():
    cmax = 35
    cmap = plt.cm.Reds
    cmaplist = [cmap(i) for i in range(cmap.N)]
    col_idx = np.arange(np.floor(256 / (cmax / 2.5)), 256, np.floor((256 / (cmax / 2.5))))
    cmaplist = [col for icol, col in enumerate(cmaplist) if icol in col_idx]
    cmaplist = [element for element in cmaplist for _ in range(5)]
    cmaplist[0] = (0.619, 0.792, 0.870, 1.0)
    cmaplist[1] = (0.619, 0.792, 0.870, 1.0)
    cmaplist = cmaplist[:-10]
    cmax = 30
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap',
                                                        cmaplist, cmax * 2)

    return cmap


def create_cmap():
    cbar_step = 0.25
    cmax = 4

    ncolors = int(np.floor((cmax - 0.5) / cbar_step))
    color_steps = int(np.floor(256 / ncolors))

    cmap = plt.cm.Reds
    cmaplist = [cmap(i) for i in range(cmap.N)]
    col_idx = np.arange(color_steps, 256, color_steps)
    cmaplist = [col for icol, col in enumerate(cmaplist) if icol in col_idx]

    cmap_lt1 = plt.cm.Blues_r
    norm_vals = np.linspace(0, 1, int(1 / cbar_step))
    for iblue in range(int(0.5 / cbar_step)):
        cmaplist[iblue] = cmap_lt1(norm_vals[iblue])

    cmaplist = cmaplist[:-2]
    ncolors = ncolors - 2

    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap',
                                                        cmaplist, ncolors)

    return cmap


def scale_figsize(figwidth, figheight, figdpi):
    """
    scale figsize to certain dpi
    :param figwidth: width in inch
    :param figheight: height in inch
    :param figdpi: desired dpi
    :return:
    """
    # factor for making fonts larger
    enlarge_fonts = 1.4

    scaling_factor = 2 / enlarge_fonts

    width = figwidth * scaling_factor
    height = figheight * scaling_factor
    dpi = figdpi / scaling_factor

    return width, height, dpi


def plot_props(vvar):
    props = {'TEX_AF_CC': {'cb_lbl': 'ERA5-TMax-p99ANN-TEX'
                                     + r'$_\mathrm{CC}$ amplification '
                                       r'($\mathcal{A}_\mathrm{CC}^\mathrm{T}$)',
                           'title': 'Total Events Extremity (TEX) amplification | Heat'},
             'EF_AF_CC': {'cb_lbl': 'ERA5-TMax-p99ANN-EF'
                                    + r'$_\mathrm{CC}$ amplification '
                                      r'($\mathcal{A}_\mathrm{CC}^\mathrm{F}$)',
                          'title': 'Event Frequency (EF) amplification | Heat'},
             'ED_avg_AF_CC': {'cb_lbl': 'ERA5-TMax-p99ANN-ED'
                                        + r'$_\mathrm{CC}$ amplification '
                                          r'($\mathcal{A}_\mathrm{CC}^\mathrm{D}$)',
                              'title': 'Event Duration (ED) amplification | Heat'},
             'EM_avg_AF_CC': {'cb_lbl': 'ERA5-TMax-p99ANN-EM'
                                        + r'$_\mathrm{CC}$ amplification '
                                          r'($\mathcal{A}_\mathrm{CC}^\mathrm{M}$)',
                              'title': 'Exceedance Magnitude (EM) amplification | Heat'},
             'EA_avg_AF_CC': {'cb_lbl': 'ERA5-TMax-p99ANN-EA'
                                        + r'$_\mathrm{CC}$ amplification '
                                          r'($\mathcal{A}_\mathrm{CC}^\mathrm{A}$)',
                              'title': 'Exceedance Area (EA) amplification | Heat'}}

    return props[vvar]


def get_lims(reg):
    """
    set limits of target region
    :param reg: SAF, SCN, or IBE
    :return: lims for lat and lon and center of target region
    """

    if reg == 'SAF':
        center = [15.5, 47]
    elif reg == 'SCN':
        center = [26, 62]
    else:
        center = [-6, 38]

    lat_lim = [center[1] - 1, center[1] + 1]
    lon_lim = [center[0] - (1 / np.cos(np.deg2rad(center[1]))),
               center[0] + (1 / np.cos(np.deg2rad(center[1])))]

    return lat_lim, lon_lim, center


def add_clutter(axs):
    # add small sub-region boxes ans labels
    for reg in ['SAF', 'IBE', 'SCN']:
        lat_lim, lon_lim, center = get_lims(reg=reg)
        geom = geometry.box(minx=lon_lim[0], maxx=lon_lim[1], miny=lat_lim[0], maxy=lat_lim[1])
        axs.add_geometries([geom], crs=ccrs.PlateCarree(),
                           edgecolor='black', facecolor='None', linewidth=1.5)
        axs.scatter(center[0], center[1], marker='x', color='black', s=15,
                    transform=ccrs.PlateCarree())

    axs.text(0.595, 0.74, 'SCN', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=10, rotation=10)
    axs.text(0.53, 0.33, 'SAF', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=10)
    axs.text(0.109, 0.192, 'IBE', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=10, rotation=-10)

    # add EUR box
    coords = []
    for ilon in np.arange(-11, 40.5, 0.5):
        coords.append((ilon, 35))
    for ilat in np.arange(35, 71.5, 0.5):
        coords.append((40, ilat))
    for ilon in np.arange(-11, 40.5, 0.5)[::-1]:
        coords.append((ilon, 71))
    for ilat in np.arange(35, 71.5, 0.5)[::-1]:
        coords.append((-11, ilat))

    geom_eur = geometry.Polygon(coords)
    axs.add_geometries([geom_eur], crs=ccrs.PlateCarree(),
                       edgecolor='black', facecolor='None', linewidth=1.5)
    axs.text(0.245, 0.91, 'EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=14, rotation=-10)

    # add horizontal lines
    for ieur in [45, 55, 70]:
        hl = []
        for ilon in np.arange(-11, 40.5, 0.5):
            hl.append((ilon, ieur))
        geom_hl = geometry.LineString(hl)
        axs.add_geometries([geom_hl], crs=ccrs.PlateCarree(), edgecolor='black',
                           facecolor='None', linewidth=1, linestyle='--')

    axs.text(0.09, 0.31, 'S-EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=12, rotation=-10)
    axs.text(0.12, 0.42, 'C-EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=12, rotation=-10)
    axs.text(0.21, 0.78, 'N-EUR', horizontalalignment='left', verticalalignment='center',
             transform=axs.transAxes, fontsize=12, rotation=-10)


def check_affected_area(data):

    gt8 = data.sel(lat=slice(70, 35), lon=slice(-11, 40)).where(data > 8)
    gt10 = data.sel(lat=slice(55, 45), lon=slice(-11, 40)).where(data > 10)

    areas = xr.open_dataarray(STATIC_PATH / 'area_grid_0p25_EUR_ERA5.nc')
    areas = areas.sel(lon=slice(-11, 40), lat=slice(70, 35))
    eur_area = areas.sum()
    ceur_area = areas.sel(lat=slice(55.25, 45)).sum()
    neur = areas.sel(lat=slice(70, 55.5)).sum()
    seur = areas.sel(lat=slice(44.75, 35)).sum()

    # percentage of EUR with AF > 8
    pct_gt8 = areas.where(gt8.notnull()).sum()/eur_area * 100

    # percentage of C-EUR with AF > 10
    pct_gt10 = areas.where(gt10.notnull()).sum()/ceur_area * 100


def run():
    data = xr.open_dataset(INPUT_PATH / 'dec_indicator_variables' /
                           'amplification/AF_Tx99.0p_AGR-EUR_annual_ERA5_1961to2024.nc')
    data = data.sel(lat=slice(72, 35), lon=slice(-11, 40))
    vkeep = ['EF_AF_CC', 'ED_avg_AF_CC', 'EM_avg_AF_CC', 'EA_avg_AF_CC', 'TEX_AF_CC']
    vdrop = [vvar for vvar in data.data_vars if vvar not in vkeep]
    data = data.drop_vars(vdrop)

    # move longitude half a pixel eastward
    pix_size = data.lon[1] - data.lon[0]
    data = data.assign_coords(lon=data.lon + pix_size / 2)

    check_affected_area(data=data['TEX_AF_CC'])

    cmap_tex = create_cmap_tex()
    cmax_tex = 30

    cmap = create_cmap()
    cmax = 3.5

    for vvar in data.data_vars:
        props = plot_props(vvar=vvar)
        fw, fh, dpi = scale_figsize(figwidth=10, figheight=7, figdpi=300)
        fig = plt.figure(figsize=(fw, fh), dpi=dpi)
        proj = ccrs.LambertConformal(central_longitude=13.5, central_latitude=53.5, cutoff=30)
        axs = plt.axes(projection=proj)
        if vvar == 'TEX_AF_CC':
            im = data[vvar].plot.imshow(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap_tex,
                                        vmin=0, vmax=cmax_tex, add_colorbar=False)
            outname = 'Figure7a'
            cx = cmax_tex
            dc = 5
        else:
            im = data[vvar].plot.imshow(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap,
                                        vmin=0.5, vmax=cmax, add_colorbar=False)
            cx = cmax
            dc = 0.5
            vstr = vvar.split('_')[0]
            outname = f'ExtDataFig6_{vstr}'
        ext = 'neither'
        if data[vvar].max() > cx:
            ext = 'max'
        cb_ticks = list(np.arange(0, cx + dc, dc))
        cb_ticks.insert(1, 1)
        cb = plt.colorbar(im, pad=0.03, ticks=cb_ticks, extend=ext)
        cb.set_label(label=props['cb_lbl'], fontsize=14)
        cb.ax.tick_params(labelsize=12)


        # add borders, gridlines, etc.
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
        axs.set_extent([-10, 40, 30, 72])
        axs.tick_params(axis='both', which='major', labelsize=12)

        # add additional boxes and labels
        add_clutter(axs=axs)

        plt.title(props['title'], fontsize=16)
        plt.savefig(f'./{outname}.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    run()
