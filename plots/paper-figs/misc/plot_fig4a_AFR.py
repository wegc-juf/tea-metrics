import cartopy.crs as ccrs
import cartopy.feature as cfea
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from shapely import geometry
import xarray as xr


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


def run():
    data = xr.open_dataset('/data/users/hst/TEA/TEA/africa_data/dec_indicator_variables/'
                           'amplification/AF_Tx99.0p_AGR_0p5_cell2-AFR_annual_ERA5_1961to2024.nc')
    # data = data.sel(lat=slice(72, 35), lon=slice(-11, 40))
    vkeep = ['EF_AF_CC', 'ED_avg_AF_CC', 'EM_avg_AF_CC', 'EA_avg_AF_CC', 'TEX_AF_CC']
    vdrop = [vvar for vvar in data.data_vars if vvar not in vkeep]
    data = data.drop_vars(vdrop)

    cmap_tex = create_cmap_tex()
    cmax_tex = 30

    cmap = create_cmap()
    cmax = 3.5

    for vvar in data.data_vars:
        props = plot_props(vvar=vvar)
        fw, fh, dpi = scale_figsize(figwidth=10, figheight=7, figdpi=300)
        fig = plt.figure(figsize=(fw, fh), dpi=dpi)
        proj = ccrs.Mollweide(central_longitude=20)
        axs = plt.axes(projection=proj)
        if vvar == 'TEX_AF_CC':
            im = data[vvar].plot.imshow(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap_tex,
                                        vmin=0, vmax=cmax_tex, add_colorbar=False)
            outname = 'Figure4a_AFR'
            cx = cmax_tex
            dc = 5
        else:
            im = data[vvar].plot.imshow(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap,
                                        vmin=0.5, vmax=cmax, add_colorbar=False)
            cx = cmax
            dc = 0.5
            vstr = vvar.split('_')[0]
            outname = f'ExtDataFig7_{vstr}_AFR'
        ext = 'neither'
        if data[vvar].max() > cx:
            ext = 'max'
        cb_ticks = list(np.arange(0, cx + dc, dc))
        cb_ticks.insert(1, 1)
        cb = plt.colorbar(im, pad=0.03, ticks=cb_ticks, extend=ext)
        cb.set_label(label=props['cb_lbl'], fontsize=12)
        cb.ax.tick_params(labelsize=10)

        # add borders, gridlines, etc.
        axs.add_feature(cfea.BORDERS)
        axs.coastlines()
        gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='black', linestyle=':',
                           x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        axs.set_extent([-16, 55, -40, 40], crs=ccrs.PlateCarree())
        axs.tick_params(axis='both', which='major', labelsize=12)

        plt.title(props['title'], fontsize=16)
        plt.savefig(f'/nas/home/hst/work/TEA/plots/misc/africa/{outname}.png',
                    dpi=300, bbox_inches='tight')
        # plt.show()
        pass

if __name__ == '__main__':
    run()
