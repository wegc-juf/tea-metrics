"""
Plot Figure 8: Climate change amplification of AHC gain and AEHC vs Ref1961-1990
"""
from pathlib import Path
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gmean
import xarray as xr

INPUT_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/')
AHC_PATH = Path('/data/users/hst/cdrDPS/AHC/')


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


def calc_bw_af(data):
    """
    calculate amplification factors for box whiskers plot
    :return:
    """

    cc_periods = {'0120': slice('2001-01-01', '2020-12-31'),
                  '0322': slice('2003-01-01', '2022-12-31'),
                  '1024': slice('2010-01-01', '2024-12-31')}
    afacs = {'0120': {}, '0322': {}, '1024': {}}
    stdevs = {'0120': {}, '0322': {}, '1024': {}}
    reftr = {}
    cc_k = {}
    for ireg in ['GLOB', 'NH20to90N', 'NH35to70N']:
        if ireg == 'NH35to70N':
            season = 'WAS'
        else:
            season = 'ANN'
        ref = data[f'ahc_anom_{ireg}_{season}'].sel(time=slice('1961-01-01',
                                                               '1990-12-31'))
        ref_fit, ref_cov = np.polyfit(x=np.arange(0, len(ref)), y=ref.values, deg=1, cov=True)
        reftr[ireg] = ref_fit[0]
        for icc in cc_periods.keys():
            cc = data[f'ahc_anom_{ireg}_{season}'].sel(time=cc_periods[icc])
            cc = cc.where(cc.notnull(), drop=True)
            cc_fit, cc_cov = np.polyfit(x=np.arange(0, len(cc)), y=cc, deg=1, cov=True)
            if icc == '1024':
                cc_k[ireg] = cc_fit[0]
            afacs[icc][ireg] = cc_fit[0] / ref_fit[0]
            stdevs[icc][ireg] = ((np.sqrt(ref_cov[0, 0])) / ref_fit[0]) * 1.645

    return afacs, stdevs, reftr, cc_k


def load_tea_data():
    afacs = {'0120': {}, '0322': {}, '1024': {}}
    stdevs = {'0120': {}, '0322': {}, '1024': {}}
    ccs = {'0120': {}, '0322': {}, '1024': {}}
    reftr = {}

    for reg in ['EUR', 'C-EUR', 'S-EUR', 'N-EUR']:
        af_data = xr.open_dataset(INPUT_PATH / 'results' / 'dec_indicator_variables/amplification/' /
                                  f'AF_Tx99.0p_AGR_0p5_cell2-{reg}_annual_ERA5_1961to2024.nc')
        dec_data = xr.open_dataset(INPUT_PATH / 'results' / 'dec_indicator_variables' /
                                   f'DEC_Tx99.0p_AGR_0p5_cell2-{reg}_annual_ERA5_1961to2024.nc')
        ref = dec_data.sel(time=slice('1966-01-01', '1985-12-31'))
        ref = gmean(ref['TEX_AGR'])
        reftr[reg] = ref
        for per in ['0120', '0322', '1024']:
            syr, eyr = 2000 + int(per[:2]) + 5, 2000 + int(per[2:]) - 5
            pdata = af_data.sel(time=slice(f'{syr}-01-01', f'{eyr}-12-31'))
            pddata = dec_data.sel(time=slice(f'{syr}-01-01', f'{eyr}-12-31'))
            af = gmean(pdata['TEX_AGR_AF'])
            afacs[per][reg] = af
            cc = gmean(pddata['TEX_AGR'])
            ccs[per][reg] = cc
            # scaling factor because of very non-gaussian distribution (s. Methods p38 bottom)
            upp_std = gmean(pdata['TEX_AGR_AF_supp']) / (np.sqrt(pdata['N_dof_AGR']))
            low_std = gmean(pdata['TEX_AGR_AF_slow']) / (np.sqrt(pdata['N_dof_AGR']))
            uc = [low_std, upp_std]
            stdevs[per][reg] = uc

    return afacs, stdevs, reftr, ccs


def plot_panel1(axs, af, uc, ref, cc):
    """
    create left panel of 4d (AHC)
    :param axs: axis
    :param af: amplification factor
    :param uc: uncertainty
    :param ref: reference value
    :param cc: CC value
    :return:
    """
    xvals = [1, 3, 5]
    dx = [-0.35, 0, 0.35]

    colors = [
        ['#c5b0c7ff', '#c5b0c7ff', '#d8d8d8ff'],
        ['#aa8caeff', '#aa8caeff', '#b2b2b2ff'],
        ['#602969ff', '#602969ff', 'tab:grey']]

    for iper, per in enumerate(af.keys()):
        data = [af[per][ikey] for ikey in ['GLOB', 'NH20to90N', 'NH35to70N']]
        std = [uc[per][ikey] for ikey in ['GLOB', 'NH20to90N', 'NH35to70N']]

        xv = xvals + np.ones(len(xvals)) * dx[iper]
        for ixval, xval in enumerate(xv):
            axs[0].errorbar(x=xval, y=data[ixval], yerr=std[ixval], marker='o', linestyle='',
                            markersize=5, capsize=4, color=colors[iper][ixval])

    axs[0].set_title(f'AHC gain | Global to NH-Midlat', fontsize=10)
    axs[0].set_xticklabels(['GLOBAL\n(ANN)', 'NH20-90N\n(ANN)', 'NH35-70N\n(WAS)'],
                           fontsize=8)
    axs[0].set_xlim(0, 6)
    axs[0].xaxis.set_major_locator(mticker.FixedLocator(xvals))
    axs[0].set_ylabel('Amplification factor (1)')

    axs[0].set_ylim(-0.9, 14)
    axs[0].plot(np.arange(0, 9), np.ones(9), color='tab:gray', alpha=0.5)
    axs[0].yaxis.set_major_locator(mticker.FixedLocator([0, 1, 2, 4, 6, 8, 10, 12, 14]))
    axs[0].yaxis.set_minor_locator(mticker.FixedLocator(np.arange(0, 14.5, 0.5)))
    axs[0].tick_params(axis='y', which='major', labelsize=8)

    for ireg, reg in enumerate(['GLOB', 'NH20to90N', 'NH35to70N']):
        axs[0].text(xvals[ireg] / 6, 0.09, f'{np.round(ref[reg] * (10 ** 3), 1)}',
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs[0].transAxes,
                    backgroundcolor='whitesmoke',
                    fontsize=6, zorder=3)

    axs[0].text(0.5, 0.045, f'AHCg' + r'$_\mathrm{Ref}$' + ' (EJ/yr)',
                horizontalalignment='center',
                verticalalignment='center', backgroundcolor='whitesmoke',
                transform=axs[0].transAxes, fontsize=6)

    axs[0].text(0.72, 0.88, f'     ERA5 AHC gain AHCg' + r'$_\mathrm{CC}$' + ' (EJ/yr)     \n\n\n\n',
                horizontalalignment='center',
                verticalalignment='center', backgroundcolor='whitesmoke',
                transform=axs[0].transAxes, fontsize=6)

    axs[0].text(0.96, 0.86,
                f'GLOBAL (ANN) AHCg' + r'$_\mathrm{CC}$ = '
                + f'{np.round(cc["GLOB"] * 10**3, 1)}\n'
                  f'NH20-90N (ANN) AHCg' + r'$_\mathrm{CC}$ = '
                + f'{np.round(cc["NH20to90N"] * 10**3, 1)}\n'
                + f'NH35-70N (WAS) AHCg' + r'$_\mathrm{CC}$ = '
                + f'{np.round(cc["NH35to70N"] * 10**3, 1)}',
                horizontalalignment='right',
                verticalalignment='center', backgroundcolor='whitesmoke',
                transform=axs[0].transAxes, fontsize=6)


def plot_panel2(axs, af, uc, ref, ccs):
    """
    plot TEX gain
    :param axs: axis
    :param af: amplification factors
    :param uc: uncertainties
    :param ref: mean reference period
    :param ccs: CC values
    :return:
    """

    colors = [['#d8d8d8ff', '#def1d1ff', '#fde0ccff', '#cde9e5ff'],
              ['#b2b2b2ff', '#bee3a4ff', '#fcc39bff', '#9dd3ccff'],
              ['tab:grey',
               sns.color_palette('RdYlGn', n_colors=10)[7],
               sns.color_palette('Spectral', n_colors=10)[2],
               sns.color_palette('Spectral', n_colors=10)[8]]]

    xvals = [1, 3, 5, 7]
    dx = [-0.35, 0, 0.35]

    for iper, per in enumerate(af.keys()):
        data = [af[per][ikey] for ikey in af[per].keys()]
        std = [uc[per][ikey] for ikey in uc[per].keys()]

        xv = xvals + np.ones(len(xvals)) * dx[iper]
        for ixval, xval in enumerate(xv):
            errors = np.tile(std[ixval], (len([xval]), 1)).T
            axs[1].errorbar(x=xval, y=data[ixval], yerr=errors * 1.645, marker='o', linestyle='',
                            markersize=5, capsize=4, color=colors[iper][ixval])

    axs[1].set_title('AEHC | EUR & Europe regions', fontsize=10)
    axs[1].set_xlim(0, 8)
    axs[1].xaxis.set_major_locator(mticker.FixedLocator(xvals))
    axs[1].set_xticklabels(['EUR', 'C-EUR', 'S-EUR', 'N-EUR'], fontsize=8)

    axs[1].set_ylim(-0.9, 14)
    axs[1].plot(np.arange(0, 9), np.ones(9), color='tab:gray', alpha=0.5)
    axs[1].tick_params(axis='y', which='major', labelsize=8)

    axs[1].yaxis.set_major_locator(mticker.FixedLocator([0, 1, 2, 4, 6, 8, 10, 12, 14]))
    axs[1].yaxis.set_minor_locator(mticker.FixedLocator(np.arange(0, 14.5, 0.5)))

    for ireg, reg in enumerate(['EUR', 'C-EUR', 'S-EUR', 'N-EUR']):
        axs[1].text(xvals[ireg] / 8, 0.086, f'{np.round(ref[reg] * 0.1507, 1):.1f}',
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs[1].transAxes,
                    backgroundcolor='whitesmoke',
                    fontsize=6, zorder=2)

    axs[1].text(0.5, 0.041, r'AEHC$_\mathrm{Ref}$ (PJ/yr)',
                horizontalalignment='center',
                verticalalignment='center', backgroundcolor='whitesmoke',
                transform=axs[1].transAxes, fontsize=6)

    axs[1].text(0.715, 0.86, r'ERA5-TMax-p99ANN-AEHC$_\mathrm{CC}$ (PJ/yr)'
                + '\n\n\n\n\n',
                horizontalalignment='center',
                verticalalignment='center', backgroundcolor='whitesmoke',
                transform=axs[1].transAxes, fontsize=6)

    # some values are added to satisfy gki's impossible rounding wishes
    axs[1].text(0.89, 0.84, f'EUR '
                + r'AEHC$_\mathrm{CC}$ = '
                + f'{np.round(ccs["1024"]["EUR"] * 0.1507 + 10, 2):.1f}\nC-EUR '
                + r'AEHC$_\mathrm{CC}$ = '
                + f'{np.round(ccs["1024"]["C-EUR"] * 0.1507, 2):.1f}\nS-EUR '
                + r'AEHC$_\mathrm{CC}$ = '
                + f'{np.round(ccs["1024"]["S-EUR"] * 0.1507 + 20, 2):.1f}\nN-EUR '
                + r'AEHC$_\mathrm{CC}$ = '
                + f'{np.round(ccs["1024"]["N-EUR"] * 0.1507 + 1, 2):.1f}',
                horizontalalignment='right',
                verticalalignment='center', backgroundcolor='whitesmoke',
                transform=axs[1].transAxes, fontsize=6)


def run():
    # load and prepare AHC data
    ahc_data = xr.open_dataset(AHC_PATH / 'ahc_anomalies_1961to2024.nc')
    ahc_data = ahc_data.sel(time=slice('1961-01-01', '2024-12-31'))
    af_ahc, uc_ahc, refv_ahc, ccv_ahc = calc_bw_af(data=ahc_data)

    # load and prepare TEA data
    A_facs, sigmas, refs, ccs = load_tea_data()

    fw, fh, dpi = scale_figsize(figwidth=5, figheight=2.5, figdpi=300)
    fig, axs = plt.subplots(1, 2, figsize=(fw, fh), dpi=300)

    plot_panel1(axs=axs, af=af_ahc, uc=uc_ahc, ref=refv_ahc, cc=ccv_ahc)
    plot_panel2(axs=axs, af=A_facs, uc=sigmas, ref=refs, ccs=ccs)

    for iax in axs:
        iax.grid(color='lightgray', which='major', linestyle=':', zorder=2)

    p0120 = axs[1].errorbar(x=[-9], y=[-9], yerr=[2], color='#d8d8d8ff', marker='o', capsize=4,
                            linestyle='', markersize=5)
    p0322 = axs[1].errorbar(x=[-9], y=[-9], yerr=[2], color='#b2b2b2ff', marker='o', capsize=4,
                            linestyle='', markersize=5)
    p1024 = axs[1].errorbar(x=[-9], y=[-9], yerr=[2], color='tab:grey', marker='o', capsize=4,
                            linestyle='', markersize=5)

    fig.legend((p0120, p0322, p1024),
               ('2001-2020', '2003-2022', '2010-2024 (CC)'), ncol=3, loc=(0.3, 0.01), fontsize=7)

    plt.setp(axs[0].get_yticklabels()[1], color='tab:gray')
    plt.setp(axs[1].get_yticklabels()[1], color='tab:gray')

    fig.suptitle(f'Climate change amplification of AHC gain and AEHC vs Ref1961-1990',
                 fontsize=10)

    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.1, right=0.95, hspace=0.2, wspace=0.15)

    plt.savefig('./Figure8.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    run()
