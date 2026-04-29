"""
Plot Extended Data Figure 2: Natural variability amplification factors
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedLocator, MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gmean
import xarray as xr

INPUT_PATH = Path('/data/users/hst/TEA-clean/TEA/paper_data/')


def load_data(city, param):
    """
    load ACTEM station data
    :param city:
    :param param: T or P
    :return:
    """

    path = INPUT_PATH / 'ctp_indicator_variables'

    if param == 'Tx':
        pstr = 'Tx99.0p'
        mvar = 'EM_avg'
    else:
        pstr = 'P24h_7to7_95.0p'
        mvar = 'EM_avg'

    ds = xr.open_dataset(path / f'CTP_{pstr}_{city}_annual_HistAlp_1877to2024.nc')
    ds = ds.sel(time=slice('1875-01-01', '1990-12-31'))

    for ivar in ds.data_vars:
        ds[ivar] = ds[ivar].rolling(time=10, center=True).mean()

        if city == 'Kremsmuenster':
            ds[ivar][:44] = np.nan

    dm = 10 ** (np.log10(ds.ED_avg) + np.log10(ds[mvar]))
    ds['DM'] = dm
    ds['DM'].rename('DM')

    tex = 10 ** (np.log10(ds.EF) + np.log10(ds.ED_avg) + np.log10(ds[mvar]))
    ds['tEX'] = tex
    ds['tEX'].rename('tEX')

    return ds


def get_props():
    yn, yx = 0, 2.5
    props = {0: {'var': 'EF', 'vname': 'EF_AF', 'cmap': 'Blues', 'yn': yn, 'yx': yx,
                 'title': 'Event Frequency (Annual) NatVar',
                 'ylbl': r'EF amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{F})$',
                 'col': 'tab:blue', 'ref': 1,
                 'bname': r'TMax-p99ANN-$\mathcal{A}_\mathrm{NV}^\mathrm{F}$', 'ls': 'solid'},
             1: {'var': 'DM', 'vname': 'DM_avg_AF', 'cmap': 'Oranges', 'yn': yn, 'yx': yx,
                 'title': r'Avg. Duration $\times$ Magnitude (events-mean) NatVar',
                 'ylbl': r'D$\times$M  amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{DM})$',
                 'col': 'tab:orange', 'ref': 1, 'ls': 'solid',
                 'bname': r'TMax-p99ANN-$\mathcal{A}_\mathrm{NV}^\mathrm{DM}$'},
             2: {'var': 'ED_avg', 'vname': 'ED_avg_AF', 'cmap': 'Purples', 'yn': yn, 'yx': yx,
                 'title': 'Avg. Event Duration (events-mean)  NatVar',
                 'ylbl': r'ED amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{D})$',
                 'col': 'tab:purple', 'ref': 1, 'ls': 'solid',
                 'bname': r'TMax-p99ANN-$\mathcal{A}_\mathrm{NV}^\mathrm{D}$'},
             3: {'var': 'DM', 'vname': 'EA_avg_AF', 'cmap': 'Greys', 'yn': yn, 'yx': yx,
                 'title': 'Avg. Exceedance Area (daily-mean) NatVar',
                 'ylbl': r'EA  amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{A})$',
                 'col': 'tab:red', 'ref': 1, 'ls': ':',
                 'bname': r'TMax-p99ANN-$\mathcal{A}_\mathrm{NV}^\mathrm{A}$'},
             4: {'var': 'EM_avg', 'vname': 'EM_avg_AF', 'cmap': 'Oranges', 'yn': yn, 'yx': yx,
                 'title': 'Avg. Exceedance Magnitude (daily-mean) NatVar',
                 'ylbl': r'EM  amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{M})$',
                 'col': 'tab:orange', 'ref': 1, 'ls': 'solid',
                 'bname': r'TMax-p99ANN-$\mathcal{A}_\mathrm{NV}^\mathrm{M}$'},
             5: {'var': 'FDMA', 'vname': 'TEX_AF', 'cmap': 'Reds', 'yn': yn, 'yx': yx,
                 'title': 'Total Events Extremity (Annual) NatVar',
                 'ylbl': r'TEX  amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{T})$',
                 'col': 'tab:red', 'ref': 1401, 'ls': 'solid',
                 'bname': r'TMax-p99ANN-$\mathcal{A}_\mathrm{NV}^\mathrm{T}$'}}

    return props


def get_p_props():
    props = {0: {'var': 'TEF', 'vname': 'EF_AF', 'cmap': 'Blues', 'yn': 0, 'yx': 2,
                 'title': 'Event Frequency (Annual) NatVar',
                 'ylbl': r'EF amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{F})$',
                 'col': 'tab:blue', 'ref': 1,
                 'bname': r'TMax-p99ANN-$\mathcal{A}_\mathrm{NV}^\mathrm{F}$', 'ls': 'solid'},
             1: {'var': 'tEX', 'vname': 'tEX_AF', 'cmap': 'Oranges', 'yn': 0, 'yx': 2.5,
                 'title': r'Temporal Events Extremity (Annual) NatVar',
                 'ylbl': r'tEX  amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{tEX})$',
                 'col': 'tab:orange', 'ref': 1, 'ls': 'solid',
                 'bname': r'TMax-p99ANN-$\mathcal{A}_\mathrm{NV}^\mathrm{tEX}$'},
             2: {'var': 'AV_ED', 'vname': 'ED_avg_AF', 'cmap': 'Purples', 'yn': 0, 'yx': 2,
                 'title': 'Avg. Event Duration (events-mean)  NatVar',
                 'ylbl': r'ED amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{D})$',
                 'col': 'tab:purple', 'ref': 1, 'ls': 'solid',
                 'bname': r'P24H-p95WAS-$\mathcal{A}_\mathrm{NV}^\mathrm{D}$'},
             3: {'var': 'DM', 'vname': 'EA_avg_AF', 'cmap': 'Greys', 'yn': 0, 'yx': 2,
                 'title': 'Avg. Exceedance Area (daily-mean) NatVar',
                 'ylbl': r'EA  amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{A})$',
                 'col': 'tab:red', 'ref': 1, 'ls': ':',
                 'bname': r'P24H-p95WAS-$\mathcal{A}_\mathrm{NV}^\mathrm{A}$'},
             4: {'var': 'AV_EM', 'vname': 'EM_avg_AF', 'cmap': 'Oranges', 'yn': 0, 'yx': 2,
                 'title': 'Avg. Exceedance Magnitude (daily-mean) NatVar',
                 'ylbl': r'EM  amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{M})$',
                 'col': 'tab:orange', 'ref': 1, 'ls': 'solid',
                 'bname': r'P24H-p95WAS-$\mathcal{A}_\mathrm{NV}^\mathrm{M}$'},
             5: {'var': 'FDMA', 'vname': 'TEX_AF', 'cmap': 'Reds', 'yn': 0, 'yx': 2,
                 'title': 'Total Events Extremity (Annual) NatVar',
                 'ylbl': r'TEX  amplification $(\mathcal{A}_\mathrm{NV}^\mathrm{T})$',
                 'col': 'tab:red', 'ref': 1401, 'ls': 'solid',
                 'bname': r'P24H-p95WAS-$\mathcal{A}_\mathrm{NV}^\mathrm{T}$'}}

    return props


def add_natvar(ii, axs, props, nv):
    xticks = np.arange(1875, 1991)

    vname = props[ii]['vname']
    # if ii == 0:
    #     vname = props[ii]['var']

    lstd, ustd = nv[f's_{vname}_NVlow'].values, nv[f's_{vname}_NVupp'].values

    nv_90low = np.ones(len(xticks)) * (1 - lstd * 1.645)
    nv_90upp = np.ones(len(xticks)) * (1 + ustd * 1.645)

    nv_68low = np.ones(len(xticks)) * (1 - lstd)
    nv_68upp = np.ones(len(xticks)) * (1 + ustd)

    axs.fill_between(x=xticks, y1=nv_90low, y2=nv_90upp, color=props[ii]['col'], alpha=0.1,
                     zorder=2)
    axs.fill_between(x=xticks, y1=nv_68low, y2=nv_68upp, color=props[ii]['col'], alpha=0.1,
                     zorder=2)

    s = [nv_68low[0], nv_68upp[0]]
    ci90 = [nv_90low[0], nv_90upp[0]]

    return s, ci90


def plot_subplot(axs, data, ii, nv, rdata, param, region, sfac, no_facs):
    if param != 'P':
        props = get_props()
    else:
        props = get_p_props()

    colors = sns.color_palette(props[ii]['cmap'], 5)

    if rdata:
        refs = []
        for ireg, reg in enumerate(data.keys()):
            rdata = data[reg]
            ref = gmean(rdata[props[ii]['var']].sel(time=slice('1966-01-01', '1986-12-31')).values)
            refs.append(ref)
            rdata = rdata / ref
            syr = pd.Timestamp(rdata.time[0].values).year
            xticks = np.arange(syr, 1991)

            if ii == 3:
                rdata = rdata * sfac + (1 - sfac)
                fac = nv[f'GR_scaling_DM_avg_AF'].values
            else:
                fac = nv[f'GR_scaling_{props[ii]["vname"]}'].values
            if no_facs:
                fac = 1
            plot_data = rdata[props[ii]['var']] * fac + 1 - fac

            # some values are negative for Wien in ED, gki wants them to be zero
            if ii == 2 and reg == 'Wien':
                plot_data = plot_data.where(plot_data > 0, 0)
                plot_data[:5] = np.nan
                plot_data[-4:] = np.nan

            axs.plot(xticks, plot_data, color=colors[-(ireg + 1)], label=reg,
                     zorder=1, linestyle=props[ii]['ls'])

    # add natvar
    std, ci90 = add_natvar(ii, axs, props, nv)

    yxx = 0
    if no_facs and param == 'T':
        yxx = 0.5

    axs.set_title(props[ii]['title'], fontsize=14)
    axs.set_ylabel(props[ii]['ylbl'], fontsize=12)
    axs.set_ylim(props[ii]['yn'], props[ii]['yx'] + yxx)
    axs.minorticks_on()
    axs.xaxis.set_minor_locator(FixedLocator(np.arange(1877, 1990)))
    axs.yaxis.set_major_locator(FixedLocator(np.arange(0, props[ii]['yx'] + yxx + 0.5, 0.5)))
    axs.yaxis.set_minor_locator(FixedLocator(np.arange(0, props[ii]['yx'] + yxx + 0.1, 0.1)))

    axs.text(0.02, 0.85, f'{props[ii]["bname"]}\n'
             + r's$^\mathrm{upp}$ | CI90$^\mathrm{upp}$ = ' + f'{std[1] - 1:.2f} | {ci90[1] - 1:.2f}\n'
             + r's$^\mathrm{low}$ | CI90$^\mathrm{low}$ = ' + f'{1 - std[0]:.2f} | {1 - ci90[0]:.2f}',
             horizontalalignment='left',
             verticalalignment='center',
             transform=axs.transAxes, backgroundcolor='whitesmoke',
             fontsize=10)


def create_legend(fig, axs, reg):
    """
    add legend at the bottom
    :param fig: figure
    :param axs: axes
    :param reg: region
    :return:
    """

    ncols = 5
    cities = ('Graz', 'Wien', 'Kremsmünster', 'Salzburg', 'Innsbruck')
    if reg == 'SEA':
        ncols = 3
        cities = ('Graz', 'Deutschlandsberg', 'BadGleichenberg')

    cols = sns.color_palette('Greys', ncols)

    if reg == 'AUT':
        grz, = axs.plot([-9, -9], color=cols[4])
        wie, = axs.plot([-9, -9], color=cols[3])
        krm, = axs.plot([-9, -9], color=cols[2])
        slz, = axs.plot([-9, -9], color=cols[1])
        inn, = axs.plot([-9, -9], color=cols[0])
        leg_entries = (grz, wie, krm, slz, inn)
        xv = 0.27
    else:
        grz, = axs.plot([-9, -9], color=cols[2])
        dlb, = axs.plot([-9, -9], color=cols[1])
        bgb, = axs.plot([-9, -9], color=cols[0])
        leg_entries = (grz, dlb, bgb)
        xv = 0.35

    fig.legend(leg_entries, cities, loc=(xv, 0.02), ncols=ncols)


def set_subplot_props(axs, ii):
    """
    set props that apply to all subplots
    :param axs: axis
    :param ii: index of axis
    :return:
    """

    axs.set_xlim(1877, 1990)

    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.xaxis.set_major_locator(MultipleLocator(10))

    axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    axs.grid(color='lightgray', linestyle=':')

    if ii in [4, 5]:
        axs.set_xlabel('Time (core year of decadal-mean value)', fontsize=12)


def run():
    reg = 'AUT'
    param = 'Tx'

    no_facs = False

    cities = ['Graz', 'Wien', 'Kremsmuenster', 'Salzburg', 'Innsbruck']
    if reg == 'SEA':
        cities = ['Graz', 'Deutschlandsberg', 'BadGleichenberg']

    if param == 'Tx':
        pstr = 'Tx99.0p'
    else:
        pstr = 'P24h_7to7_95.0p'

    data = {}
    for ict in cities:
        data[ict] = load_data(city=ict, param=param)

    nat_var = xr.open_dataset(INPUT_PATH / 'natural_variability' / f'NV_AF_{pstr}_{reg}.nc')
    fac = nat_var['std_scaling_EA_DM'].values

    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    axes = axs.reshape(-1)

    for iax, ax in enumerate(axes):
        real_data = True
        if iax == 5:
            real_data = False
        plot_subplot(axs=ax, data=data, ii=iax, nv=nat_var, rdata=real_data, param=param,
                     region=reg, sfac=fac, no_facs=no_facs)
        set_subplot_props(axs=ax, ii=iax)

    create_legend(fig=fig, axs=axes[0], reg=reg)

    # iterate over each subplot and add a text label
    labels = ['a)', 'd)', 'b)', 'e)', 'c)', 'f)']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.2, labels[i], transform=ax.transAxes, fontsize=14,
                va='top', ha='left')

    fig.subplots_adjust(wspace=0.2, hspace=0.33)

    if reg == 'AUT' and param == 'Tx':
        outname = 'ExtDataFig2'
    else:
        outname = f'misc-EDF2-{reg}-{param}'
    plt.savefig(f'./{outname}.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    run()
