#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Figure 5
"""
import argparse
import csv
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gmean
import xarray as xr

from teametrics.common.general_functions import ref_cc_params

INPUT_DATA_PATH = Path('/home/wegnet/results/TEA_indicators_test/')
MASK_PATH = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/masks/')

PARAMS = ref_cc_params()

END_YEAR = 2026
LOGGER = logging.getLogger(__name__)
TEXT_BOX_CLEARANCE = 0.80


def nice_tick_step(max_value):
    """Return a conventional tick interval with at least four intervals."""
    target = max_value / 4
    exponent = int(np.floor(np.log10(target)))
    candidates = []
    for current_exponent in range(exponent - 1, exponent + 2):
        scale = 10 ** current_exponent
        candidates.extend(factor * scale for factor in (1, 2, 2.5, 5, 10))
    valid = [candidate for candidate in candidates if candidate <= target]
    return max(valid) if valid else min(candidates)


def get_data(varname='Tx30.0degC', ctp='june', region='AUT',
             input_data_path=INPUT_DATA_PATH):
    """Load CTP, decadal and amplification data for one TEA region."""
    dec_path = (input_data_path / 'dec_indicator_variables' /
                f'DEC_{varname}_{region}_{ctp}_SPARTACUS_1961to{END_YEAR}.nc')
    ctp_paths = sorted((input_data_path / 'ctp_indicator_variables').glob(
        f'CTP_{varname}_{region}_{ctp}_SPARTACUS_*.nc'))
    af_path = (input_data_path / 'dec_indicator_variables' /
               f'amplification/AF_{varname}_{region}_{ctp}_SPARTACUS_1961to{END_YEAR}.nc')

    for label, path in [('decadal indicators', dec_path),
                        ('amplification factors', af_path)]:
        if not path.exists():
            raise FileNotFoundError(f'{label.capitalize()} file not found: {path}')

    LOGGER.info('Loading decadal indicators from %s', dec_path)
    dec = xr.open_dataset(dec_path)
    LOGGER.info('Loading %d CTP files from %s', len(ctp_paths),
                input_data_path / 'ctp_indicator_variables')
    if not ctp_paths:
        raise FileNotFoundError(f'No CTP files found for region {region}: '
                                f'{input_data_path / "ctp_indicator_variables"}')
    LOGGER.debug('CTP files: %s', ', '.join(str(path) for path in ctp_paths))
    ctp_data = xr.open_mfdataset(ctp_paths, data_vars='minimal', compat='no_conflicts')
    LOGGER.info('Loading amplification factors from %s', af_path)
    af = xr.open_dataset(af_path)
    LOGGER.info('Loaded datasets: dec=%s, ctp=%s, af=%s',
                dict(dec.sizes), dict(ctp_data.sizes), dict(af.sizes))

    return dec, ctp_data, af


def find_mask(region, input_data_path):
    """Find a mask matching the region and data generation run."""
    candidates = [
        input_data_path / 'masks' / f'{region}_mask_SPARTACUS_1500.nc',
        input_data_path / 'masks' / f'{region}_masks_SPARTACUS.nc',
        MASK_PATH / f'{region}_mask_SPARTACUS_1500.nc',
        MASK_PATH / f'{region}_masks_SPARTACUS.nc',
    ]
    for path in candidates:
        if path.exists():
            LOGGER.info('Using region mask %s', path)
            return path
    LOGGER.warning('No SPARTACUS mask found for region %s. Checked: %s',
                   region, ', '.join(str(path) for path in candidates))
    return None


def write_timeseries_csv(dec, ann, region, run_name, output_path):
    """Write all regional time series used by Figure 5 to a wide CSV."""
    variables = ['EF_GR', 'ED_avg_GR', 'EM_avg_GR', 'EA_avg_GR',
                 'TEX_GR', 'TEX_HW_max_GR']
    series = {}

    def add_series(name, data):
        values = np.asarray(data.values).reshape(-1)
        years = [int(str(value)[:4]) for value in data.time.values]
        series[name] = dict(zip(years, values))

    for variable in variables:
        add_series(f'{variable}_annual', ann[variable])
        add_series(f'{variable}_decadal', dec[variable])
        add_series(f'{variable}_supp', dec[f'{variable}_supp'])
        add_series(f'{variable}_slow', dec[f'{variable}_slow'])

    years = sorted({year for values in series.values() for year in values})
    fieldnames = ['year', 'run_name', 'region', *series]
    LOGGER.info('Writing %d time-series rows and %d columns to %s',
                len(years), len(fieldnames), output_path)
    with output_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for year in years:
            row = {
                'year': year,
                'run_name': run_name,
                'region': region,
            }
            for name, values in series.items():
                value = values.get(year)
                row[name] = '' if value is None or not np.isfinite(value) else value
            writer.writerow(row)
    LOGGER.info('Completed CSV export: %s', output_path)


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
                         'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{T}$', 'nv_name': 'TEX',
                         'yx': 45000, 'dy': 5000, 'unit': 'areal °C days/yr',
                         'ref': r'$\mathcal{T}_\mathrm{Ref}$', 'cc': r'$\mathcal{T}_\mathrm{CC}$'},
              'TEX_max_GR': {'col': 'tab:red',
                             'ylbl': r'Max. Event Extremity $(\mathcal{T}_\mathrm{max})$ (areal °C days/yr)',
                             'title': 'Maximum Event Extremity (Annual)',
                             'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{Tmax}$', 'nv_name': 'TEX_max',
                             'yx': 25000, 'dy': 5000, 'unit': 'areal °C days/yr',
                             'ref': r'$\mathcal{T}_\mathrm{max,Ref}$', 'cc': r'$\mathcal{T}_\mathrm{max,CC}$'},
              'TEX_HW_max_GR': {'col': 'tab:red',
                                'ylbl': r'Max. Heatwave Extremity $(\mathcal{T}_\mathrm{HW,max})$ (areal °C days/yr)',
                                'title': 'Maximum Heatwave Extremity (Annual)',
                                'acc': r'$\mathcal{A}_\mathrm{CC}^\mathrm{THW,max}$',
                                'nv_name': 'TEX_HW_max',
                                'yx': 25000, 'dy': 5000, 'unit': 'areal °C days/yr',
                                'ref': r'$\mathcal{T}_\mathrm{HW,max,Ref}$',
                                'cc': r'$\mathcal{T}_\mathrm{HW,max,CC}$'}
              }

    return params[vname]


def plot_gr_data(ax, adata, ddata, afdata, su, sl):
    props = gr_plot_params(vname=ddata.name)
    plotted_values = np.concatenate((
        np.atleast_1d(np.asarray(adata)),
        np.atleast_1d(np.asarray(ddata + su)),
    ))
    finite_values = plotted_values[np.isfinite(plotted_values)]
    if finite_values.size:
        plotted_max = finite_values.max()
        dy = props['dy']
        if plotted_max > 0 and ddata.name in ['EA_avg_GR', 'TEX_GR', 'TEX_max_GR', 'TEX_HW_max_GR']:
            dy = nice_tick_step(plotted_max)
        else:
            while plotted_max > 0 and plotted_max / dy < 4:
                dy /= 2
        # Reserve the upper part of the axes for the multi-line annotation box.
        ymax = dy * np.ceil((plotted_max / TEXT_BOX_CLEARANCE) / dy)
        ymax = max(dy, ymax)
        LOGGER.info('%s y-axis: upper limit %.1f, major tick interval %.g, plotted maximum %.1f',
                    ddata.name, ymax, dy, plotted_max)
    else:
        dy = props['dy']
        ymax = props['yx']
        LOGGER.warning('%s contains no finite plotted values; using fallback y-axis upper limit %.1f',
                       ddata.name, ymax)

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

    ypos_ref = (ref / ymax) + 0.05
    ypos_cc = (cc / ymax) + 0.05
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
    if ddata.name in ['EA_avg_GR', 'TEX_GR', 'TEX_max_GR', 'TEX_HW_max_GR']:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(1960, END_YEAR + 1)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1960, END_YEAR + 1)))
    ax.set_title(props['title'], fontsize=14)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(FixedLocator(np.arange(0, ymax + dy, dy)))

    if ddata.name == 'EA_avg_GR':
        ax.text(0.02, 0.89, f'TMax-p99ANN-{props["nv_name"]}' + r'$_\mathrm{Ref | CC}$ = '
                + f'{ref:.1f}' + r'$\,$|$\,$'
                + f'{cc:.1f} {props["unit"]} \n'
                + r'$\mathcal{A}_\mathrm{CC}^\mathrm{A}$ = '
                + f'{afdata:.2f}',
                horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
                fontsize=9)
    elif ddata.name in ['TEX_GR', 'TEX_max_GR', 'TEX_HW_max_GR']:
        ax.text(0.02, 0.89, f'TMax-p99ANN-{props["nv_name"]}' + r'$_\mathrm{Ref | CC}$ = '
                + f'{ref:.0f}' + r'$\,$|$\,$'
                + f'{cc:.0f} {props["unit"]} \n'
                 + props['acc'] + ' = '
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


def plot_map(fig, ax, data, region, mask_path=None):
    props = map_plot_params(vname=data.name)

    if mask_path is not None:
        LOGGER.info('Loading map mask from %s', mask_path)
        mask = xr.open_dataset(mask_path)
        mask = mask.sel(x=data.x, y=data.y)
        mask_var = 'nw_mask' if 'nw_mask' in mask else 'mask'
        LOGGER.info('Plotting map background from mask variable %s', mask_var)
        ax.contourf(mask.x, mask.y, mask[mask_var], colors='mistyrose')

    data = data.where(data > 0)

    if data.max() > props['lvls'][-1] and data.min() > props['lvls'][0]:
        ext = 'max'
    elif data.max() < props['lvls'][-1] and data.min() > props['lvls'][0]:
        ext = 'neither'
    else:
        ext = 'min'

    range_vals = [data.min().values, data.max().values]

    map = ax.contourf(data.x, data.y, data, cmap=props['cmap'],
                      levels=props['lvls'], extend=ext)
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(map, cax=cax, orientation='vertical', extend=ext)
    cb.set_label(label=props['lbl'], fontsize=12)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(props['title'], fontsize=14)

    a_sym = props['lbl'].split(' ')[0]
    ax.text(0.02, 0.82, a_sym + '\n'
            + f'{region}: [{range_vals[0]:.2f}, {range_vals[1]:.2f}]',
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes, backgroundcolor='whitesmoke',
            fontsize=9)


def run(run_name, region='AUT', tex_variable='TEX_HW_max_GR',
        output_dir=Path('.'), show=True, csv=False):
    LOGGER.info('Starting Figure 5: run=%s, region=%s, extremity=%s',
                run_name, region, tex_variable)
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
    elif run_name == 'current_detrended':
        input_data_path = Path(
            '/data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwave_paper/SPARTACUS_detrended/JJA/')
        ctp = 'annual'
    elif run_name == 'current':
        input_data_path = Path('/data/arsclisys/normal/clim-hydro/TEA-Indicators/results/heatwave_paper/')
        ctp = 'annual'
    else:
        raise ValueError(f'Unknown run name: {run_name}')

    LOGGER.info('Input data directory: %s; CTP: %s', input_data_path, ctp)
    dec, ann, af = get_data(varname='Tx30.0degC', ctp=ctp, region=region,
                            input_data_path=input_data_path)

    LOGGER.info('Creating Figure 5 canvas')
    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    gr_vars = ['EF_GR', 'ED_avg_GR', 'EM_avg_GR', 'EA_avg_GR']
    for irow, gr_var in enumerate(gr_vars):
        LOGGER.info('Plotting regional time series: %s', gr_var)
        plot_gr_data(ax=axs[irow, 0], adata=ann[gr_var], ddata=dec[gr_var],
                     afdata=af[f'{gr_var}_AF_CC'],
                     su=dec[f'{gr_var}_supp'], sl=dec[f'{gr_var}_slow'])
        su_mean = dec[f'{gr_var}_supp'].sel(time=slice(f'1961-01-01', f'1985-12-31')).mean().values
        sl_mean = dec[f'{gr_var}_slow'].sel(time=slice(f'1961-01-01', f'1985-12-31')).mean().values
        LOGGER.info('%s: mean supp = %.3f, mean slow = %.3f',
                    gr_var, su_mean, sl_mean)

    LOGGER.info('Plotting extremity time series: %s', tex_variable)
    plot_gr_data(ax=axs[3, 1], adata=ann[tex_variable], ddata=dec[tex_variable],
                 afdata=af[f'{tex_variable}_AF_CC'],
                 su=dec[f'{tex_variable}_supp'], sl=dec[f'{tex_variable}_slow'])
    su_mean = dec[f'{tex_variable}_supp'].sel(
        time=slice(f'1961-01-01', f'1985-12-31')).mean().values
    sl_mean = dec[f'{tex_variable}_slow'].sel(
        time=slice(f'1961-01-01', f'1985-12-31')).mean().values
    LOGGER.info('%s: mean supp = %.3f, mean slow = %.3f',
                tex_variable, su_mean, sl_mean)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / f'Figure5_{run_name}_{region}_{tex_variable}'
    if csv:
        write_timeseries_csv(dec=dec, ann=ann, region=region, run_name=run_name,
                             output_path=output_dir / f'Figure5_data_{run_name}_{region}.csv')

    mask_path = find_mask(region, input_data_path)
    map_vars = ['EF', 'ED_avg', 'EM_avg']
    LOGGER.info('Preparing %d map panels', len(map_vars))
    for irow, map_var in enumerate(map_vars):
        mdata = gmean(dec[map_var].sel(
            time=slice(f'{END_YEAR-9}-01-01', f'{END_YEAR-4}-12-31')), axis=0)
        mdata = xr.DataArray(data=mdata, coords={'y': (['y'], dec.y.values),
                                                 'x': (['x'], dec.x.values)}, name=map_var)
        plot_map(fig=fig, ax=axs[irow, 1], data=mdata, region=region,
                 mask_path=mask_path)

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
    output_path = output_stem.with_suffix('.png')
    LOGGER.info('Saving Figure 5 to %s', output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if show:
        LOGGER.info('Showing Figure 5')
        plt.show()
    else:
        plt.close(fig)
    LOGGER.info('Completed Figure 5: %s', output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='current',
                        choices=['RW1', 'RW2', 'CW1', 'CW2', 'CW3',
                                 'current_detrended', 'current'],
                        help='Configured data run to plot (default: current).')
    parser.add_argument('--region', default='AUT',
                        help='calc_TEA region name, e.g. AUT, SEA, FBR, EUR or an Austrian state (default: AUT).')
    parser.add_argument('--tex-variable', default='TEX_GR',
                        choices=['TEX_GR', 'TEX_max_GR', 'TEX_HW_max_GR'],
                        help='Extremity series for the final panel (default: TEX_GR).')
    parser.add_argument('--output-dir', type=Path, default=Path('.'),
                        help='Directory for generated figures (default: current directory).')
    parser.add_argument('--no-show', action='store_true',
                        help='Save the figure without opening an interactive window.')
    parser.add_argument('--csv', action='store_true',
                        help='Write the plotted time-series values beside the PNG.')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity (default: INFO).')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    run(args.run_name, region=args.region, tex_variable=args.tex_variable,
        output_dir=args.output_dir, csv=args.csv,
        show=not args.no_show)
