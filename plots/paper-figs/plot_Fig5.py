#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Figure 5
"""
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gmean
import xarray as xr

from plot_fig4 import find_range

from teametrics.common.general_functions import ref_cc_params

INPUT_DATA_PATH = Path('/home/wegnet/results/TEA_indicators_test/')
MASK_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/')

PARAMS = ref_cc_params()

END_YEAR = 2026


def get_data(varname='Tx30.0degC', ctp='june', input_data_path=INPUT_DATA_PATH):
    dec = xr.open_dataset(input_data_path / 'dec_indicator_variables' /
                          f'DEC_{varname}_AUT_{ctp}_SPARTACUS_1961to{END_YEAR}.nc')

    ctp_data = xr.open_mfdataset(
        sorted((input_data_path / 'ctp_indicator_variables').glob(f'CTP_{varname}_AUT_{ctp}_SPARTACUS_*.nc')),
        data_vars='minimal')
    
    af_path = (input_data_path / 'dec_indicator_variables' /
                         f'amplification/AF_{varname}_AUT_{ctp}_SPARTACUS_1961to{END_YEAR}.nc')
    print(f"Loading amplification factors from {af_path}")
    af = xr.open_dataset(af_path)

    return dec, ctp_data, af


def gr_plot_params(vname):
    params = {'EF_GR': {'col': 'tab:blue',
                        'ylbl': r'EF ($F_s$|$F_p$) (ev/yr)',
                        'title': 'Event Frequency (Annual)',
                        'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$',
                        'nv_name': 'EF',
                        'yx': 16, 'dy': 4,
                        'unit': 'ev/yr',
                        'ref': r'$F_\mathrm{Ref}$', 'cc': r'$F_\mathrm{CC}$'},
              'ED_avg_GR': {'col': 'tab:purple',
                            'ylbl': r'ED $(\overline{D}_s$|$\overline{D}_p)$ (days)',
                            'title': 'Average Event Duration (events-mean)',
                            'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{D}$',
                            'nv_name': 'ED',
                            'yx': 10, 'dy': 2,
                            'unit': 'days',
                            'ref': r'$\overline{D}_\mathrm{Ref}$', 'cc': r'$\overline{D}_\mathrm{CC}$'},
              'EM_avg_GR': {'col': 'tab:orange',
                            'ylbl': r'EM $(\overline{M}_s$|$\overline{M}_p)$ (°C)',
                            'title': 'Average Exceedance Magnitude (daily-mean)',
                            'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$',
                            'nv_name': 'EM',
                            'yx': 2, 'dy': 0.5, 'unit': '°C',
                            'ref': r'$\overline{M}_\mathrm{Ref}$',
                            'cc': r'$\overline{M}_\mathrm{CC}$'},
              'EA_avg_GR': {'col': 'tab:red',
                            'ylbl': r'EA $(\overline{A}_s$|$\overline{A}_p)$ (areals)',
                            'title': 'Average Exceedance Area (daily-mean)',
                            'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$', 'nv_name': 'EA',
                            'yx': 600, 'dy': 100, 'unit': 'areals',
                            'ref': r'$\overline{A}_\mathrm{Ref}$',
                            'cc': r'$\overline{A}_\mathrm{CC}$'},
              'TEX_GR': {'col': 'tab:red',
                         'ylbl': r'TEX $(\mathcal{T}_s|\mathcal{T}_p)$ (areal °C days/yr)',
                         'title': 'Total Events Extremity (Annual)',
                         'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$', 'nv_name': 'EA',
                         'yx': 45000, 'dy': 5000, 'unit': 'areal °C days/yr',
                         'ref': r'$\mathcal{T}_\mathrm{Ref}$', 'cc': r'$\mathcal{T}_\mathrm{CC}$'}
              }

    return params[vname]


def plot_gr_data(ax, adata, ddata, afdata, su, sl):
    props = gr_plot_params(vname=ddata.name)

    xticks = np.arange(1961, END_YEAR + 1)

    ref = gmean(ddata.sel(time=slice(PARAMS['REF']['start_cy'], PARAMS['REF']['end_cy'])))
    cc = gmean(ddata.sel(time=slice(PARAMS['CC']['start_cy'], PARAMS['CC']['end_cy'])))

    ax.fill_between(x=xticks, y1=ddata - sl, y2=ddata + su, color=props['col'], alpha=0.2)
    ax.plot(xticks, ddata, 'o-', color=props['col'], markersize=3, linewidth=2)
    ax.plot(xticks, adata, 'o-', color=props['col'], markersize=2, linewidth=1, alpha=0.5)
    ax.plot(xticks[:30], np.ones(len(xticks[:30])) * ref,
            alpha=0.8, color=props['col'], linewidth=2)
    ax.plot(xticks[49:], np.ones(len(xticks[49:])) * cc,
            alpha=0.6, color=props['col'], linewidth=2)

    ypos_ref = (ref / props['yx']) + 0.05
    ypos_cc = (cc / props['yx']) + 0.05
    ax.text(0.02, ypos_ref, props['ref'],
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    ax.text(0.93, ypos_cc, props['cc'],
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=11)

    ax.set_ylabel(props['ylbl'], fontsize=12)
    ax.minorticks_on()
    ax.grid(color='gray', which='major', linestyle=':')
    if ddata.name in ['EA_avg_GR', 'TEX_GR']:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(1960, END_YEAR + 1)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, END_YEAR + 1)))
    ax.set_title(props['title'], fontsize=14)
    ax.set_ylim(0, props['yx'])
    ax.yaxis.set_major_locator(FixedLocator(np.arange(0, props['yx'] + props['dy'], props['dy'])))

    if ddata.name == 'EA_avg_GR':
        ax.text(0.02, 0.89, f'TMax-p99ANN-{props["nv_name"]}' + r'$_\mathrm{Ref | CC}$ = '
                + f'{ref:.1f}' + r'$\,$|$\,$'
                + f'{cc:.1f} {props["unit"]} \n'
                + r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$ = '
                + f'{afdata:.2f}',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)
    elif ddata.name == 'TEX_GR':
        ax.text(0.02, 0.89, f'TMax-p99ANN-{props["nv_name"]}' + r'$_\mathrm{Ref | CC}$ = '
                + f'{ref:.0f}' + r'$\,$|$\,$'
                + f'{cc:.0f} {props["unit"]} \n'
                + r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$ = '
                + f'{afdata:.2f}',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)
    else:
        ax.text(0.02, 0.89, f'TMax-p99ANN-{props["nv_name"]}' + r'$_\mathrm{Ref | CC}$ = '
                + f'{ref:.2f}' + r'$\,$|$\,$'
                + f'{cc:.2f} {props["unit"]} \n'
                + r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$ = '
                + f'{afdata:.2f}',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)


def map_plot_params(vname):
    params = {'EF': {'cmap': 'Blues',
                     'lbl': r'EF$_\mathrm{CC}$(i,j) (ev/yr)',
                     'title': f'Event Frequency (Annual) (CC{END_YEAR-14}-{END_YEAR})',
                     'lvls': np.arange(1, 11)},
              'ED_avg': {'cmap': 'Purples',
                         'lbl': r'ED$_\mathrm{CC}$(i,j) (days)',
                         'title': f'Avarage Event Duration (CC{END_YEAR-14}-{END_YEAR})',
                         'lvls': np.arange(1, 3.75, 0.25)},
              'EM_avg': {'cmap': 'Oranges',
                         'lbl': r'EM$_\mathrm{CC}$(i,j) (°C)',
                         'title': f'Average Exceedance Magnitude (CC{END_YEAR-14}-{END_YEAR})',
                         'lvls': np.arange(1, 2.6, 0.2)}
              }

    return params[vname]


def plot_map(fig, ax, data):
    props = map_plot_params(vname=data.name)

    aut = xr.open_dataset(MASK_PATH / 'AUT_masks_SPARTACUS.nc')
    aut = aut.sel(x=data.x, y=data.y)
    ax.contourf(aut.nw_mask, colors='mistyrose')

    data = data.where(data > 0)

    if data.max() > props['lvls'][-1] and data.min() > props['lvls'][0]:
        ext = 'max'
    elif data.max() < props['lvls'][-1] and data.min() > props['lvls'][0]:
        ext = 'neither'
    else:
        ext = 'min'

    range_vals = find_range(data=data)

    map = ax.contourf(data, cmap=props['cmap'], levels=props['lvls'], extend=ext)
    ax.add_patch(pat.Rectangle(xy=(473, 53), height=20, width=25, edgecolor='black',
                               fill=False, linewidth=1))
    ax.add_patch(pat.Rectangle(xy=(410, 25), height=92, width=125, edgecolor='black',
                               fill=False, linewidth=1))
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map, cax=cax, orientation='vertical', extend=ext)
    cb.set_label(label=props['lbl'], fontsize=12)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(props['title'], fontsize=14)

    a_sym = props['lbl'].split(' ')[0]
    ax.text(0.02, 0.82, a_sym + '\n'
            + f'AUT: [{range_vals["AUT"][0]:.2f}, {range_vals["AUT"][1]:.2f}]\n'
              f'SEA: [{range_vals["SEA"][0]:.2f}, {range_vals["SEA"][1]:.2f}]\n'
              f'FBR: [{range_vals["FBR"][0]:.2f}, {range_vals["FBR"][1]:.2f}]',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)


def run(run_name):
    if run_name == 'RW1':
        input_data_path = INPUT_DATA_PATH
        ctp = 'june'
    elif run_name == 'RW2':
        input_data_path = INPUT_DATA_PATH
        ctp = 'JJA'
    elif run_name == 'CW1':
        input_data_path = Path('/home/wegnet/results/SPARTACUS_DETRENDED_June')
        ctp = 'june'
    elif run_name == 'CW2':
        input_data_path = Path('/home/wegnet/results/SPARTACUS_DETRENDED_JJA')
        ctp = 'JJA'
    elif run_name == 'CW3':
        input_data_path = Path('/home/wegnet/results/SPARTACUS_DETRENDED_JJA')
        ctp = 'june'
    dec, ann, af = get_data(varname='Tx30.0degC', ctp=ctp, input_data_path=input_data_path)

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    gr_vars = ['EF_GR', 'ED_avg_GR', 'EM_avg_GR', 'EA_avg_GR']
    for irow, gr_var in enumerate(gr_vars):
        plot_gr_data(ax=axs[irow, 0], adata=ann[gr_var], ddata=dec[gr_var],
                     afdata=af[f'{gr_var}_AF_CC'],
                     su=dec[f'{gr_var}_supp'], sl=dec[f'{gr_var}_slow'])
        su_mean = dec[f'{gr_var}_supp'].sel(time=slice(f'1961-01-01', f'1985-12-31')).mean().values
        sl_mean = dec[f'{gr_var}_slow'].sel(time=slice(f'1961-01-01', f'1985-12-31')).mean().values
        print(f"{gr_var}: mean supp = {su_mean:.3f}, mean slow = {sl_mean:.3f}")

    plot_gr_data(ax=axs[3, 1], adata=ann['TEX_GR'], ddata=dec['TEX_GR'], afdata=af['TEX_GR_AF_CC'],
                 su=dec['TEX_GR_supp'], sl=dec['TEX_GR_slow'])
    su_mean = dec['TEX_GR_supp'].sel(time=slice(f'1961-01-01', f'1985-12-31')).mean().values
    sl_mean = dec['TEX_GR_slow'].sel(time=slice(f'1961-01-01', f'1985-12-31')).mean().values
    print(f"TEX_GR: mean supp = {su_mean:.3f}, mean slow = {sl_mean:.3f}")

    map_vars = ['EF', 'ED_avg', 'EM_avg']
    for irow, map_var in enumerate(map_vars):
        mdata = gmean(dec[map_var].sel(time=slice(f'{END_YEAR-9}-01-01', f'{END_YEAR-4}-12-31')), axis=0)
        mdata = xr.DataArray(data=mdata, coords={'y': (['y'], dec.y.values),
                                                 'x': (['x'], dec.x.values)}, name=map_var)
        plot_map(fig=fig, ax=axs[irow, 1], data=mdata)

    axs[2, 1].text(0, 0, 'Alpine data at z > 1500m excluded.',
                   horizontalalignment='left', verticalalignment='center',
                   transform=axs[2, 1].transAxes, backgroundcolor='mistyrose',
                   fontsize=8)

    # iterate over each subplot and add a text label
    labels = ['a)', 'e)', 'b)', 'f)', 'c)', 'g)', 'd)', 'h)']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.2, labels[i], transform=ax.transAxes, fontsize=14,
                va='top', ha='left')

    fig.subplots_adjust(wspace=0.2, hspace=0.33)
    print(f"Saving Figure 5 to ./Figure5_{run_name}.png")
    plt.savefig(f'./Figure5_{run_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # run()
    run('RW1')
    run('RW2')
    run('CW1')
    run('CW2')
    run('CW3')
