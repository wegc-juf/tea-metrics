from matplotlib.ticker import FixedLocator
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def load_data(gr_type):
    gr_str = ''
    if gr_type == 'AGR':
        gr_str = 'AGR_0.5-'
    ds = xr.open_dataset(f'/data/users/hst/TEA/TEA/testy_data/dec_indicator_variables/amplification/'
                         f'AF_Tx99.0p_{gr_str}AUT_annual_ERA5_1961to2025.nc')

    vvars = [f'EF_{gr_type}_AF', f'ED_avg_{gr_type}_AF', f'EM_avg_{gr_type}_AF', f'EA_avg_{gr_type}_AF',
             f'tEX_{gr_type}_AF', f'TEX_{gr_type}_AF',
             f'EF_{gr_type}_AF_CC', f'ED_avg_{gr_type}_AF_CC', f'EM_avg_{gr_type}_AF_CC', f'EA_avg_{gr_type}_AF_CC',
             f'tEX_{gr_type}_AF_CC', f'TEX_{gr_type}_AF_CC']
    dvars = [dvar for dvar in ds.data_vars if dvar not in vvars]

    ds = ds.drop_vars(dvars)

    return ds


def var_props(vname):
    props = {'EF': {'col': 'tab:blue',
                    'ylbl': r'EF amplification $(\mathcal{A}^\mathrm{F})$',
                    'title': 'Event Frequency (Annual)',
                    'unit': 'ev/yr',
                    'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{F}$',
                    'ylim': (0.5, 3)},
             'ED_avg': {'col': 'tab:purple',
                        'ylbl': r'ED amplification $(\mathcal{A}^\mathrm{D})$',
                        'title': 'Average Event Duration (events-mean)',
                        'unit': 'days',
                        'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{D}$',
                        'ylim': (0.5, 2)},
             'EM_avg': {'col': 'tab:orange',
                        'ylbl': r'EM amplification $(\mathcal{A}^\mathrm{M})$',
                        'title': 'Average Exceedance Magnitude (daily-mean)',
                        'unit': '°C',
                        'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{M}$',
                        'ylim': (0.5, 2)},
             'EA_avg': {'col': 'tab:red',
                        'ylbl': r'EA amplification $(\mathcal{A}^\mathrm{A})$',
                        'title': 'Average Exceedance Area (daily-mean)',
                        'unit': 'areals',
                        'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$',
                        'ylim': (0.5, 2)},
             'tEX': {'col': 'tab:orange',
                     'ylbl': r'tEX amplification $(\mathcal{A}^\mathrm{t})$',
                     'title': 'Temporal Events Extremity (daily-mean)',
                     'unit': '°C days/yr',
                     'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{t}$',
                     'ylim': (0, 10)},
             'TEX': {'col': 'tab:red',
                     'ylbl': r'TEX amplification $(\mathcal{A}^\mathrm{T})$',
                     'title': 'Total Events Extremity',
                     'unit': 'areal °C days/yr',
                     'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$',
                     'ylim': (0, 10)}
             }

    return props[vname]


def plot_subplot(fig, axs, gr, agr, vname):
    props = var_props(vname)

    years = np.arange(1961, 2026)

    axs.plot(years, gr[f'{vname}_GR_AF'], color=props['col'],
             label=f'GR-{props["acc"]} = {gr[f"{vname}_GR_AF_CC"].values:.2f}')
    axs.plot(years, agr[f'{vname}_AGR_AF'], linestyle='dashed', color=props['col'],
             label=f'AGR-{props["acc"]} = {agr[f"{vname}_AGR_AF_CC"].values:.2f}')

    axs.set_title(props['title'])
    axs.set_ylabel(props['ylbl'], fontsize=12)
    axs.legend(loc='upper left')

    axs.set_xlim(1960, 2025)
    axs.grid(color='lightgray', which='major', linestyle=':')
    axs.xaxis.set_minor_locator(FixedLocator(np.arange(1960, 2025)))

    axs.set_ylim(props['ylim'])


def run():
    gr = load_data(gr_type='GR')
    agr = load_data(gr_type='AGR')

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.reshape(-1)

    vvars = ['EF', 'ED_avg', 'EM_avg', 'EA_avg', 'tEX', 'TEX']

    for ivar, vvar in enumerate(vvars):
        gr_vars = [grvar for grvar in gr.data_vars if vvar in grvar]
        agr_vars = [agrvar for agrvar in agr.data_vars if vvar in agrvar]
        gr_ds = gr[gr_vars]
        agr_ds = agr[agr_vars]
        plot_subplot(fig=fig, axs=axs[ivar], gr=gr_ds, agr=agr_ds, vname=vvar)
        if ivar > 3:
            axs[ivar].set_xlabel('Time (core year of decadal-mean value)', fontsize=12)

    # iterate over each subplot and add a text label
    labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    for i, ax in enumerate(axs.flat):
        ax.text(-0.2, 1.2, labels[i], transform=ax.transAxes, fontsize=14,
                va='top', ha='left')

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.35)
    plt.savefig('/nas/home/hst/work/TEA/plots/misc/AUT_ERA5_Tx99p_GRvsAGR.png', bbox_inches='tight', dpi=300)
    # plt.show()


if __name__ == '__main__':
    run()
