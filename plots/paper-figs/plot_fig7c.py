"""
Plot Figure 7c
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import xarray as xr

from teametrics.common.general_functions import ref_cc_params

INPUT_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/')


def run():
    regs = ['EUR', 'S-EUR', 'C-EUR', 'N-EUR']
    cols = {'EUR': 'tab:grey',
            'S-EUR': sns.color_palette('Spectral', n_colors=10)[2],
            'C-EUR': sns.color_palette('RdYlGn', n_colors=10)[7],
            'N-EUR': sns.color_palette('Spectral', n_colors=10)[8]}

    fig, axs = plt.subplots(1, 1, figsize=(10, 9))
    xticks = np.arange(1960, 2025)

    axs.plot(np.arange(1960, 2025), np.ones(len(np.arange(1960, 2025))), linewidth=1,
             color='tab:grey', linestyle='--')

    cc_vals = {}
    xv = -4
    for ireg, reg in enumerate(regs):
        data = xr.open_dataset(INPUT_PATH / 'results/dec_indicator_variables' /
                               f'amplification/AF_Tx99.0p_AGR_0p5_cell2-{reg}_annual_ERA5_1961to2024.nc')
        if reg == 'EUR':
            axs.fill_between(x=xticks[1:],
                             y1=data['TEX_AGR_AF_p05'],
                             y2=data['TEX_AGR_AF_p95'],
                             color=cols[reg],
                             alpha=0.1, zorder=2)
            lw, ms = 3.5, 3.5
            vline_x = xticks[xv]
        else:
            lw, ms = 3, 3
            xv += 1
            vline_x = xticks[xv]

        cc_vals[reg] = {'mean': data['TEX_AGR_AF_CC'].values,
                        'low': data['TEX_AGR_AF_CC_p05'],
                        'upp': data['TEX_AGR_AF_CC_p95']}

        axs.vlines(x=vline_x, ymin=data['TEX_AGR_AF_CC_p05'], ymax=data['TEX_AGR_AF_CC_p95'],
                   colors=cols[reg],
                   linewidth=4, alpha=0.35)

        cc_low = (data['TEX_AGR_AF_CC']
                  - data['TEX_AGR_AF_CC_slow'] * (1 / np.sqrt(data['N_dof_AGR'])) * 1.645)
        cc_upp = (data['TEX_AGR_AF_CC']
                  + data['TEX_AGR_AF_CC_supp'] * (1 / np.sqrt(data['N_dof_AGR'])) * 1.645)
        axs.vlines(x=vline_x, ymin=cc_low,
                   ymax=cc_upp,
                   colors=cols[reg],
                   linewidth=4, alpha=0.9)

        axs.plot(xticks[1:], data['TEX_AGR_AF'], color=cols[reg], linewidth=lw, markersize=ms,
                 label=reg)

        axs.plot(xticks[-15:],
                 np.ones(len(xticks[-15:])) * data['TEX_AGR_AF_CC'].values,
                 color=cols[reg], linewidth=3, alpha=0.7)

    axs.plot(xticks[1:31], np.ones(len(xticks[1:31])), color='tab:gray', linewidth=2)

    axs.text(0.016, 0.9,
             r'ERA5-TMax-p99ANN-$\mathcal{A}^\mathrm{T}_\mathrm{CC}$' + '\n'
             + f'EUR: {np.round(cc_vals["EUR"]["mean"], 1):.1f} '
               f'[{np.round(cc_vals["EUR"]["low"], 1):.1f} to '
               f'{np.round(cc_vals["EUR"]["upp"], 1):.1f}]\n'
             + f'C-EUR: {np.round(cc_vals["C-EUR"]["mean"], 1):.1f} '
               f'[{np.round(cc_vals["C-EUR"]["low"], 1):.1f} to '
               f'{np.round(cc_vals["C-EUR"]["upp"], 1):.1f}]\n'
             + f'S-EUR: {np.round(cc_vals["S-EUR"]["mean"], 1):.1f} '
               f'[{np.round(cc_vals["S-EUR"]["low"], 1):.1f} to '
               f'{np.round(cc_vals["S-EUR"]["upp"], 1):.1f}]\n'
             + f'N-EUR: {np.round(cc_vals["N-EUR"]["mean"], 1):.1f} '
               f'[{np.round(cc_vals["N-EUR"]["low"], 1):.1f} to '
               f'{np.round(cc_vals["N-EUR"]["upp"], 1):.1f}]',
             horizontalalignment='left', verticalalignment='center', transform=axs.transAxes,
             fontsize=14, backgroundcolor='whitesmoke')

    fig.legend(loc=(0.25, 0.03), ncol=4, fontsize=12)

    axs.set_title('Total Events Extremity (TEX) amplification | EUR & Europe regions',
                  fontsize=16)
    axs.set_xlabel('Time (core year of decadal-mean value)', fontsize=14)
    axs.set_ylabel(r'TEX amplification ($\mathcal{A}_\mathrm{CC}^\mathrm{T}$)',
                   fontsize=14)
    axs.minorticks_on()
    axs.grid(color='gray', which='major', linestyle=':')
    axs.set_xlim(1960, 2025)
    axs.set_ylim(0, 30)
    axs.xaxis.set_minor_locator(mticker.FixedLocator(np.arange(1960, 2025)))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.9)
    plt.savefig('./Figure7c.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    run()
