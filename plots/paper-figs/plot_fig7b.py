"""
Plot Figure 7b
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gmean
import xarray as xr

from teametrics.common.general_functions import ref_cc_params

INPUT_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/')

PARAMS = ref_cc_params()


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


def create_cmap():
    """
    create colomap for panel 4b
    :return:
    """
    cmap = plt.cm.YlOrRd
    cmaplist = [cmap(i) for i in range(cmap.N)]
    col_idx = np.arange(8, 256, 32)
    cmaplist = [col for icol, col in enumerate(cmaplist) if icol in col_idx]
    col_list = []
    ii = 0
    while ii < 30:
        col_list.append((0.77, 0.77, 0.77, 1.0))
        ii += 1
    cmaplist_1990to2024 = []
    for idec in range(len(cmaplist)):
        ii = 0
        while ii < 5:
            cmaplist_1990to2024.append(cmaplist[idec])
            ii += 1
    col_list.extend(cmaplist_1990to2024[:-5])
    col_list = col_list[:-2]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap',
                                                        col_list, len(col_list))

    return cmap


def set_plot_props(ax, reg, acc, refv, tpoint):
    """
    set props of plot (figure 4b)
    :param ax: axis
    :param reg: region
    :param acc: Amplification in CC
    :param refv: threshold value
    :param tpoint: point to annotate for A_s^T(t)
    :return: props
    """

    xmin, xmax = 0, 3
    ymin, ymax = 0, 7
    ax.set_xticks(np.arange(xmin, xmax + 1, 1))
    ax.set_yticks(np.arange(ymin, ymax + 1, 1))

    apos = {'SAF': {'ref': (15, -4), 'cc': (5, 5), 't': (-37, -4)},
            'IBE': {'ref': (15, -4), 'cc': (10, -4), 't': (-37, -4)},
            'SCN': {'ref': (20, -4), 'cc': (15, -4), 't': (-50, -4)}}
    ax.annotate(r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$ = ' + f'{(acc[0] * acc[1]):.1f}',
                xy=(acc[0], acc[1]), textcoords='offset points', xytext=apos[reg]['cc'], fontsize=12)
    ax.annotate(r'$\mathcal{A}_\mathrm{Ref} = 1$',
                xy=(1, 1), textcoords='offset points', xytext=apos[reg]['ref'], fontsize=12)
    ax.annotate(r'$\mathcal{A}_\mathrm{s}^\mathrm{T}(t)$',
                xy=(tpoint[0], tpoint[1]), textcoords='offset points', xytext=apos[reg]['t'], fontsize=12)

    ax.text(0.03, 0.93,
            'ERA5-TMax-p99ANN-' + r'$\mathcal{A}_\mathrm{s|CC}^\mathrm{T}$' + '\n'
            + f'({reg} Avg.Ref-TMax = {refv:.1f} °C)',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=12, backgroundcolor='whitesmoke')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.xaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1, 2, 3]))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.2))

    ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1, 2, 3, 4, 5, 6, 7]))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.2))


def plot_scatter(fig, ax, ef, es, reg, ref):
    acc = [gmean(ef.sel(time=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy']))),
           gmean(es.sel(time=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy'])))]

    years = np.arange(1961, 2025)

    ax.fill_between([0, acc[0]], [acc[1], acc[1]], color='tab:red', alpha=0.1)
    ax.fill_between([0, 1], [1, 1], color='whitesmoke')

    ax.grid(color='lightgray', which='major', zorder=1)
    ax.minorticks_on()

    cmap = create_cmap()

    ax.plot(ef, es, color='tab:grey', zorder=2)
    dots = ax.scatter(ef, es, c=years, cmap=cmap, vmin=1961, vmax=2024, s=50, zorder=3)
    ax.scatter(
        np.round(gmean(ef.sel(time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))),
                 1),
        np.round(gmean(es.sel(time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy']))),
                 0), s=70,
        linewidth=1, facecolors='tab:grey', edgecolors='black', zorder=4)
    ax.scatter(acc[0], acc[1], s=70, linewidth=1,
               facecolors='tab:red', edgecolors='black', zorder=4)

    traj_point = [ef.sel(time='2003-01-01').values, es.sel(time='2003-01-01').values]

    set_plot_props(ax=ax, reg=reg, acc=acc, refv=ref, tpoint=traj_point)

    subtitles = {'SAF': 'C-Europe Region SAF', 'IBE': 'S-Europe Region IBE',
                 'SCN': 'N-Europe Region SCN'}

    ax.set_title(subtitles[reg], fontsize=14)
    ax.set_xlabel(r'Event frequency amplification ($\mathcal{A}^\mathrm{F}$)',
                  fontsize=14)

    if reg == 'SCN':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='10%', pad=0.15)
        cb = fig.colorbar(dots, cax=cax, orientation='vertical',
                          ticks=np.arange(1961, 2025, 5)[1:])
        cb.set_label(label='Time (core year of decadal-mean value)', fontsize=14)
        ylabels = [str(itick) for itick in np.arange(1965, 2025, 5)]
        cb.ax.set_yticklabels(ylabels)
        cb.ax.tick_params(labelsize=12)

    if reg == 'SAF':
        ax.set_ylabel(r'Event severity amplification ($\mathcal{A}^\mathrm{S}$)',
                      fontsize=14)


def run():
    fw, fh, dpi = scale_figsize(figwidth=14, figheight=5, figdpi=300)
    fig, axs = plt.subplots(1, 3, figsize=(fw, fh), dpi=dpi)

    data = xr.open_dataset(INPUT_PATH / 'results' / 'dec_indicator_variables' /
                           'amplification/AF_Tx99.0p_AGR_0p5_cell2-EUR_annual_ERA5_1961to2024.nc')
    thresh = xr.open_dataset(INPUT_PATH / 'static' / 'static_Tx99.0p_EUR_ERA5.nc')
    thresh = thresh.threshold

    regions = ['SAF', 'IBE', 'SCN']
    for ireg, reg in enumerate(regions):
        lat_lim, lon_lim, center = get_lims(reg=reg)
        rdata = data.sel(lat=slice(lat_lim[1], lat_lim[0]), lon=slice(lon_lim[0], lon_lim[1]))
        rthresh = thresh.sel(lat=slice(lat_lim[1], lat_lim[0]), lon=slice(lon_lim[0], lon_lim[1]))
        freq = rdata['EF_AF'].sel(lat=center[1], lon=center[0])
        sev = rdata['ES_avg_AF'].sel(lat=center[1], lon=center[0])
        rthresh = rthresh.mean(dim=('lat', 'lon'))

        plot_scatter(fig=fig, ax=axs[ireg], ef=freq, es=sev, ref=rthresh, reg=reg)

    title = ('Total Events Extremity (TEX) amplification trajectories | Heat '
             '(CYrs 1966-2020 vs Ref1961-1990)')
    fig.suptitle(title, fontsize=16)

    fig.tight_layout(rect=[0.05, 0.05, 0.93, 1])
    plt.subplots_adjust(hspace=0.15, wspace=0.2)

    plt.savefig('./Figure7b.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    run()
