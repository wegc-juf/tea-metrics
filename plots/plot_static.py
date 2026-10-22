"""
plot static input of TEA
@author: hst
"""

import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import xarray as xr

from common.config import load_opts
from teametrics.common.grid_naming import grg_grid_token


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

    myopts = parser.parse_args()

    return myopts


def load_data(opts):
    """
    load static input data
    Args:
        opts: CLI parameter

    Returns:
        stat: static input data
        masks: mask data
    """
    stat = {}

    # load mask(s) data
    stat['mask'] = xr.open_dataarray(f'{opts.maskpath}/masks/{opts.region}_mask_{opts.dataset}_'
                                     f'{opts.altitude_threshold}.nc')
    if opts.dataset == 'ERA5':
        stat['mask_grg'] = xr.open_dataarray(
            f'{opts.maskpath}/masks/{opts.region}_mask_{grg_grid_token(opts)}_ERA5_'
            f'{opts.altitude_threshold}.nc')

    # load static data
    stat['thresh'] = xr.open_dataarray(f'{opts.maskpath}/'
                                       f'threshold_{opts.param_str}_{opts.period}_{opts.region}'
                                       f'_{opts.dataset}.nc')
    if opts.dataset == 'ERA5':
        stat['area'] = xr.open_dataarray(f'{opts.maskpath}/'
                                         f'area_grid_{grg_grid_token(opts)}_{opts.region}_{opts.dataset}.nc')

    return stat


def get_lims(data, rval):
    """
    get min and max values and round them for plotting
    Args:
        data: input data to get lims for
        rval: rounding value

    Returns:
        vmin: rounded minimum value
        vmax: rounded maximum value
    """

    vn = data.min().values
    vx = data.max().values

    # round to next round val (rval)
    vmin = np.floor(vn / rval) * rval
    vmax = np.ceil(vx / rval) * rval

    return vmin, vmax


def plot_props(opts, var_name, data):

    # color map
    if var_name != 'thresh':
        cmap = 'cividis'
    elif 'T' in opts.parameter:
        cmap = 'YlOrRd'
    else:
        cmap = 'YlGnBu'

    # data range
    if 'mask' in var_name:
        vn, vx = 0, 1
        delta = 0.1
        unit = '1'
    else:
        if var_name == 'area_grid':
            rval = 0.01
            delta = 0.05
            unit = 'areals'
        else:
            rval = 5
            delta = 2.5
            unit = opts.unit
        vn, vx = get_lims(data=data, rval=rval)

    # x and y ranges
    valid_cells = data.where(data.notnull(), drop=True)
    xn, xx = get_lims(data=valid_cells[opts.xname], rval=5)
    yn, yx = get_lims(data=valid_cells[opts.yname], rval=5)

    props = {'cmap': cmap, 'vn': vn, 'vmax': vx, 'xn': xn, 'xx': xx, 'yn': yn, 'yx': yx,
             'step': delta, 'cb_lbl': f'{var_name} [{unit}]'}

    return props


def plot_static_data(opts, data):
    """
    plot static data
    Args:
        opts: CLI parameter
        data: dictionary with static data

    Returns:

    """

    # set up figure
    n_subs = len(data.keys())
    nrow, ncol = 1, n_subs
    if n_subs == 4:
        nrow, ncol = 2, 2
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol * 4, nrow * 3.5))
    axs = axs.flatten()

    # get names of x and y dims
    dims = opts.xy_name.split(',')
    opts.xname, opts.yname = dims[0], dims[1]

    for i, (key, da) in enumerate(data.items()):
        props = plot_props(opts=opts, var_name=key, data=da)
        lvls = np.arange(props['vn'], props['vmax'] + props['step'], props['step'])
        map_vals = axs[i].contourf(da[opts.xname], da[opts.yname], da.values,
                                   cmap=props['cmap'], levels=lvls)

        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(map_vals, cax=cax, orientation='vertical', label=props['cb_lbl'])

        axs[i].set_xlabel(opts.xname)
        axs[i].set_ylabel(opts.yname)
        axs[i].set_xlim(props['xn'], props['xx'])
        axs[i].set_ylim(props['yn'], props['yx'])

    fig.suptitle(f'Static data for {opts.region} ({opts.dataset})', fontsize=16)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.25, wspace=0.4)

    # check and create output path
    outpath = f'{opts.outpath}/plots'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    plt.savefig(f'{outpath}/static_data_{opts.region}_{opts.param_str}_{opts.dataset}.png',
                dpi=300, bbox_inches='tight')


def run():
    # get command line parameters
    cmd_opts = _getopts()

    # load CFG parameters
    opts = load_opts(fname=__file__, config_file=cmd_opts.config_file)

    # load static data and mask
    static_dict = load_data(opts)

    plot_static_data(opts=opts, data=static_dict)


if __name__ == '__main__':
    run()
