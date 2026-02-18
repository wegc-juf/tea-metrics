"""
Plot TEX map
"""

import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfea
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
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


def create_cmap_tex():
    """
    create custom color map for TEX plot
    Returns:

    """
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
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmax * 2)

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


def plot_eur_tex(opts):
    """
    Plot map over EUR with TEX AF values
    Args:
        opts: CLI parameter

    Returns:

    """

    # data = xr.open_dataset('INPUT_PATH' / 'dec_indicator_variables' /
    #                        'amplification/AF_Tx99.0p_AGR-EUR_annual_ERA5_1961to2024.nc')
    if opts.agr:
        gr_str = f'AGR-{opts.agr}'
    else:
        gr_str = opts.region
    data = xr.open_dataset(f'{opts.outpath}dec_indicator_variables/amplification/'
                           f'AF_{opts.param_str}_{gr_str}_{opts.period}_{opts.dataset}_{opts.start}to{opts.end}.nc')
    data = data.sel(lat=slice(72, 35), lon=slice(-11, 40))
    data = data['TEX_AF_CC']

    # move longitude half a pixel eastward
    pix_size = data.lon[1] - data.lon[0]
    data = data.assign_coords(lon=data.lon + pix_size / 2)

    cmap_tex = create_cmap_tex()
    cmax_tex = 30

    fw, fh, dpi = scale_figsize(figwidth=10, figheight=7, figdpi=300)
    fig = plt.figure(figsize=(fw, fh), dpi=dpi)
    proj = ccrs.LambertConformal(central_longitude=13.5, central_latitude=53.5, cutoff=30)
    axs = plt.axes(projection=proj)
    im = data.plot.imshow(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap_tex,
                          vmin=0, vmax=cmax_tex, add_colorbar=False)
    cx = cmax_tex
    dc = 5
    ext = 'neither'
    if data.max() > cx:
        ext = 'max'
    cb_ticks = list(np.arange(0, cx + dc, dc))
    cb_ticks.insert(1, 1)
    cb = plt.colorbar(im, pad=0.03, ticks=cb_ticks, extend=ext)
    cb.set_label(label=f'{opts.dataset}-{opts.param_str}-{opts.period}-TEX'
                       + r'$_\mathrm{CC}$ amplification '
                         r'($\mathcal{A}_\mathrm{CC}^\mathrm{T}$)', fontsize=14)
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
    plt.title('Total Events Extremity (TEX) amplification', fontsize=16)

    # check and create output path
    dbv_outpath = f'{opts.outpath}/plots'
    if not os.path.exists(dbv_outpath):
        os.makedirs(dbv_outpath)
    plt.savefig(f'{opts.outpath}/plots/TEX-map_{opts.param_str}_{gr_str}_{opts.period}_{opts.dataset}'
                           f'_{opts.start}to{opts.end}.png', dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    cmd_opts = _getopts()
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)
    plot_eur_tex(opts)
