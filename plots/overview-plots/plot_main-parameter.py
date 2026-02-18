"""
Plot main parameter (GR and CC map)
"""
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from scipy.stats import gmean
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


def get_data(opts):
    af = xr.open_dataset(f'{opts.outpath}dec_indicator_variables/amplification/'
                         f'AF_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}_{opts.start}to{opts.end}.nc')

    dec = xr.open_dataset(f'{opts.outpath}dec_indicator_variables/'
                          f'DEC_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}_{opts.start}to{opts.end}.nc')

    return af, dec


def gr_plot_params(opts, vname):
    params = {'EF_GR_AF': {'col': 'tab:blue',
                           'ylbl': r'EF amplification $(\mathcal{A}^\mathrm{F})$',
                           'title': 'Event Frequency (Annual)',
                           'unit': 'ev/yr',
                           'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$',
                           'nv_name': 's_EF_AF_NV'},
              'ED_avg_GR_AF': {'col': 'tab:purple',
                               'ylbl': r'ED amplification $(\mathcal{A}^\mathrm{D})$',
                               'title': 'Average Event Duration (events-mean)',
                               'unit': 'days',
                               'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{D}$',
                               'nv_name': 's_ED_avg_AF_NV'},
              'EM_avg_GR_AF': {'col': 'tab:orange',
                               'ylbl': r'EM amplification $(\mathcal{A}^\mathrm{M})$',
                               'title': 'Average Exceedance Magnitude (daily-mean)',
                               'unit': f'{opts.unit}',
                               'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$',
                               'nv_name': 's_EM_avg_AF_NV'},
              'EA_avg_GR_AF': {'col': 'tab:red',
                               'ylbl': r'EA amplification $(\mathcal{A}^\mathrm{A})$',
                               'title': 'Average Exceedance Area (daily-mean)',
                               'unit': 'areals',
                               'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$',
                               'nv_name': 's_EA_avg_AF_NV'}
              }

    return params[vname]


def map_plot_params(opts, vname):
    """
    set plot props for maps
    Args:
        opts: CLI parameter
        vname: variable name

    Returns:

    """
    params = {'EF_AF_CC': {'cmap': 'Blues',
                           'lbl': r'$\mathcal{A}^\mathrm{F}_\mathrm{CC}$',
                           'title': f'Event Frequency (EF) amplification (CC{opts.cc_period[0]}-{opts.cc_period[1]})'},
              'ED_avg_AF_CC': {'cmap': 'Purples',
                               'lbl': r'$\mathcal{A}^\mathrm{D}_\mathrm{CC}$',
                               'title': f'Event Duration (ED) amplification (CC{opts.cc_period[0]}-{opts.cc_period[1]})'},
              'EM_avg_AF_CC': {'cmap': 'Oranges',
                               'lbl': r'$\mathcal{A}^\mathrm{M}_\mathrm{CC}$',
                               'title': f'Exceedance Magnitude (EM) amplification (CC{opts.cc_period[0]}-{opts.cc_period[1]})'}
              }

    return params[vname]


def plot_gr_data(opts, ax, data, ddata, vname):
    """
    create plot of GR data of EF, ED, EM, and EA
    Args:
        opts: CLI parameter
        ax: axis
        data: AF data
        ddata: DEC data
        vname: name of variable

    Returns:

    """
    props = gr_plot_params(opts=opts, vname=vname)

    xvals = data.time
    xticks = np.arange(opts.start, opts.end + 1)

    ax.plot(xticks, data[vname], 'o-', color=props['col'], markersize=3, linewidth=2)

    # find indices of ref/cc period
    ref_sidx, ref_eidx = np.where(xticks == opts.ref_period[0])[0][0], np.where(xticks == opts.ref_period[1])[0][0]
    cc_sidx, cc_eidx = np.where(xticks == opts.cc_period[0])[0][0], np.where(xticks == opts.cc_period[1])[0][0]

    ax.plot(xticks[ref_sidx:ref_eidx + 1], np.ones(len(xvals[ref_sidx:ref_eidx + 1])), alpha=0.5, color=props['col'],
            linewidth=2)
    ax.plot(xticks[cc_sidx:cc_eidx + 1], np.ones(len(xvals[cc_sidx:cc_eidx + 1])) * data[f'{vname}_CC'].values,
            alpha=0.5, color=props['col'], linewidth=2)

    ax.set_ylabel(props['ylbl'], fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # round start/end year
    syr, eyr = np.floor(opts.start / 5) * 5, np.ceil(opts.end / 5) * 5
    ax.set_xlim(syr, eyr)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(syr, eyr)))

    ymin, ymax = 0.5, 2
    ax.set_yticks(np.arange(ymin, ymax + 0.5, 0.5))
    ax.set_ylim(ymin, ymax)

    ax.set_title(props['title'], fontsize=14)

    ypos_ref = 0.4
    ypos_cc = ((data[f'{vname}_CC'].values - ymin) / (ymax - ymin)) + 0.05
    ax.text(0.02, ypos_ref, r'$\mathcal{A}_\mathrm{Ref}$',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)
    ax.text(0.93, ypos_cc, props['acc'],
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    # TODO: take values directly from file
    ref_cy_syr_ts, ref_cy_eyr_ts = pd.Timestamp(f'{opts.ref_period[0] + 5}-01-01'), pd.Timestamp(
        f'{opts.ref_period[1] - 5}-01-01')
    cc_cy_syr_ts, cc_cy_eyr_ts = pd.Timestamp(f'{opts.cc_period[0] + 5}-01-01'), pd.Timestamp(
        f'{opts.cc_period[1] - 5}-01-01')
    ref_abs = gmean(ddata.sel(time=slice(ref_cy_syr_ts, ref_cy_eyr_ts)))
    cc_abs = gmean(ddata.sel(time=slice(cc_cy_syr_ts, cc_cy_eyr_ts)))

    if vname == 'EA_avg_GR_AF':
        ax.text(0.02, 0.9, f'{opts.param_str}-{opts.period}-{vname[:2]}' + r'$_\mathrm{Ref | CC}$ = '
                + f'{ref_abs:.1f}' + r'$\,$|$\,$'
                + f'{cc_abs:.1f} {props["unit"]} \n'
                + props['acc'] + ' = ' + f'{data[f'{vname}_CC'].values:.2f}',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)
        ax.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)
    else:
        ax.text(0.02, 0.9, f'{opts.param_str}-{opts.period}-{vname[:2]}' + r'$_\mathrm{Ref | CC}$ = '
                + f'{ref_abs:.2f}' + r'$\,$|$\,$'
                + f'{cc_abs:.2f} {props["unit"]} \n'
                + props['acc'] + ' = ' + f'{data[f"{vname}_CC"].values:.2f}',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)


def plot_tex_es(opts, ax, data, ddata):
    """
    plot GR values of ES and TEX
    Args:
        opts: CLI parameter
        ax: axis
        data: AF data
        ddata: decadal data
        af_cc: AF in CC period

    Returns:

    """
    xvals = data.time
    xticks = np.arange(opts.start, opts.end + 1)

    ax.plot(xticks, data['ES_avg_GR_AF'], 'o-', color='tab:grey', markersize=3, linewidth=2)
    ax.plot(xticks, data['TEX_GR_AF'], 'o-', color='tab:red', markersize=3, linewidth=2)

    # find indices of ref/cc period
    ref_sidx, ref_eidx = np.where(xticks == opts.ref_period[0])[0][0], np.where(xticks == opts.ref_period[1])[0][0]
    cc_sidx, cc_eidx = np.where(xticks == opts.cc_period[0])[0][0], np.where(xticks == opts.cc_period[1])[0][0]

    ax.plot(xticks[ref_sidx:ref_eidx + 1], np.ones(len(xvals[ref_sidx:ref_eidx + 1])), alpha=0.5, color='tab:grey',
            linewidth=2)
    ax.plot(xticks[cc_sidx:cc_eidx + 1], np.ones(len(xvals[cc_sidx:cc_eidx + 1])) * data['ES_avg_GR_AF_CC'].values,
            alpha=0.5, color='tab:grey', linewidth=2)
    ax.plot(xticks[cc_sidx:cc_eidx + 1], np.ones(len(xvals[cc_sidx:cc_eidx + 1])) * data['TEX_GR_AF_CC'].values,
            alpha=0.5, color='tab:red', linewidth=2)

    ax.set_ylabel(r'ES|TEX amplification $(\mathcal{A}^\mathrm{S}, \mathcal{A}^\mathrm{T})$',
                  fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # round start/end year
    syr, eyr = np.floor(opts.start / 5) * 5, np.ceil(opts.end / 5) * 5
    ax.set_xlim(syr, eyr)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(syr, eyr)))

    ymin, ymax = 0, 10
    ax.set_yticks(np.arange(ymin, ymax + 1, 1))
    ax.set_ylim(ymin, ymax)

    ax.set_title('Avg. Event Severity and Total Events Extremity', fontsize=14)
    ax.set_xlabel('Time (core year of decadal-mean value)', fontsize=10)

    ypos_ref = 0.12
    ypos_cc_tex = ((data['TEX_GR_AF_CC'].values - ymin) / (ymax - ymin)) + 0.05
    ypos_cc_es = ((data['ES_avg_GR_AF_CC'].values - ymin) / (ymax - ymin)) + 0.05
    ax.text(0.02, ypos_ref, r'$\mathcal{A}_\mathrm{Ref}$',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=11)
    ax.text(0.93, ypos_cc_es, r'$\mathcal{A}_\mathrm{CC}^\mathrm{S}$',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=11)
    ax.text(0.93, ypos_cc_tex, r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=11)

    # TODO: take values directly from file
    ref_cy_syr_ts, ref_cy_eyr_ts = pd.Timestamp(f'{opts.ref_period[0] + 5}-01-01'), pd.Timestamp(
        f'{opts.ref_period[1] - 5}-01-01')
    cc_cy_syr_ts, cc_cy_eyr_ts = pd.Timestamp(f'{opts.cc_period[0] + 5}-01-01'), pd.Timestamp(
        f'{opts.cc_period[1] - 5}-01-01')
    ref_abs_tex = gmean(ddata['TEX_GR'].sel(time=slice(ref_cy_syr_ts, ref_cy_eyr_ts)))
    cc_abs_tex = gmean(ddata['TEX_GR'].sel(time=slice(cc_cy_syr_ts, cc_cy_eyr_ts)))
    ref_abs_es = gmean(ddata['ES_avg_GR'].sel(time=slice(ref_cy_syr_ts, ref_cy_eyr_ts)))
    cc_abs_es = gmean(ddata['ES_avg_GR'].sel(time=slice(cc_cy_syr_ts, cc_cy_eyr_ts)))

    ax.text(0.02, 0.85,
            f'{opts.param_str}-{opts.period}-TEX' + r'$_\mathrm{Ref | CC}$'
            + f' = {ref_abs_tex:.0f}' + r'$\,$|$\,$'
            + f'{cc_abs_tex:.0f} ' + r'areal$\,$' + opts.unit + r'$\,$days$\,$/$\,$yr '
            + f'\n{opts.param_str}-{opts.period}-ES' + r'$_\mathrm{Ref | CC}$'
            + f' = {ref_abs_es:.0f}' + r'$\,$|$\,$'
            + f'{cc_abs_es:.0f} ' + r'areal$\,$' + opts.unit + r'$\,$days$\,$'
            + '\n'
            + r'$\mathcal{A}_\mathrm{CC}^\mathrm{S} | \mathcal{A}_\mathrm{CC}^\mathrm{T}$ = '
            + f'{data["ES_avg_GR_AF_CC"]:.2f}' + r'$\,$|$\,$'
            + f'{data["TEX_GR_AF_CC"]:.2f}',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)


def plot_map(opts, fig, ax, data):
    """
    create maps of EF, ED, and EM
    Args:
        opts: CLI parameter
        fig: figure
        ax: axis
        data: grided variable data

    Returns:

    """
    props = map_plot_params(opts=opts, vname=data.name)

    cntry = xr.open_dataset(f'{opts.maskpath}{opts.region}_masks_{opts.dataset}.nc')
    if 'x' in data.dims:
        cntry = cntry.sel(x=data.x, y=data.y)
    else:
        cntry = cntry.sel(x=data.lon, y=data.lat)
    ax.contourf(cntry.nw_mask, colors='mistyrose')

    lvls = np.arange(1, 4.25, 0.25)
    if data.max() > lvls[-1] and data.min() > lvls[0]:
        ext = 'max'
    elif data.max() < lvls[-1] and data.min() > lvls[0]:
        ext = 'neither'
    else:
        ext = 'min'

    gt0_data = data.where(data > 0)
    val_min, val_max = gt0_data.min().values, gt0_data.max().values
    range_vals = [val_min, val_max]

    map_vals = ax.contourf(data, cmap=props['cmap'], extend=ext, levels=lvls, vmin=1, vmax=4)

    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map_vals, cax=cax, orientation='vertical')
    cb.set_label(label=f'{opts.param_str}-{opts.period}-{props["lbl"]}', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(props["title"], fontsize=14)

    ax.text(0.02, 0.92, props['lbl'] + f'(i,j) [{range_vals[0]:.2f}, {range_vals[1]:.2f}]',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)


def plot_main_parameter(opts):
    """
    create plot of main TEA parameters (EF, ED, EM, EA, ES, TEX)
    Args:
        opts: CLI parameter

    Returns:

    """
    data, dec_data = get_data(opts)

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    gr_vars = ['EF_GR_AF', 'ED_avg_GR_AF', 'EM_avg_GR_AF', 'EA_avg_GR_AF']
    for irow, gr_var in enumerate(gr_vars):
        plot_gr_data(opts=opts, ax=axs[irow, 0], data=data[[gr_var, f'{gr_var}_CC']], vname=gr_var,
                     ddata=dec_data[gr_var.split('_AF')[0]])

    map_vars = ['EF_AF_CC', 'ED_avg_AF_CC', 'EM_avg_AF_CC']
    for irow, map_var in enumerate(map_vars):
        plot_map(opts=opts, fig=fig, ax=axs[irow, 1], data=data[map_var])

    plot_tex_es(opts=opts, ax=axs[3, 1], data=data[['TEX_GR_AF', 'ES_avg_GR_AF', f'TEX_GR_AF_CC', f'ES_avg_GR_AF_CC']],
                ddata=dec_data[['TEX_GR', 'ES_avg_GR']])


    # iterate over each subplot and add a text label
    labels = ['a)', 'e)', 'b)', 'f)', 'c)', 'g)', 'd)', 'h)']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.2, labels[i], transform=ax.transAxes, fontsize=14,
                va='top', ha='left')

    fig.subplots_adjust(wspace=0.2, hspace=0.33)

    plt.savefig(f'{opts.outpath}/plots/main-parameter_{opts.param_str}_{opts.region}_{opts.period}_{opts.dataset}'
                f'_{opts.start}to{opts.end}.png', dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    cmd_opts = _getopts()
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)
    # check and create output path
    plt_outpath = f'{opts.outpath}/plots'
    if not os.path.exists(plt_outpath):
        os.makedirs(plt_outpath)
    plot_main_parameter(opts)
