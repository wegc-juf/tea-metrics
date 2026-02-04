import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

def load_data(gr_type):

    gr_str = ''
    if gr_type == 'AGR':
        gr_str = 'AGR-'
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
    props = {'EF': {'color': 'tab:blue', 'title': 'EF'},
             'ED_avg': {'color': 'tab:purple', 'title': 'ED_avg'},
             'EM_avg': {'color': 'tab:orange', 'title': 'EM_avg'},
             'EA_avg': {'color': 'tab:red', 'title': 'EA_avg'},
             'tEX': {'color': 'tab:orange', 'title': 'tEX'},
             'TEX': {'color': 'tab:red', 'title': 'TEX'}, }

    return props[vname]

def plot_subplot(fig, axs, gr, agr, vname):

    props = var_props(vname)
    axs.plot(gr.time, gr[f'{vname}_GR_AF'], color=props['color'], label=f'GR_AF_CC = {gr[f"{vname}_GR_AF_CC"].values:.2f}')
    axs.plot(agr.time, agr[f'{vname}_AGR_AF'], linestyle='dashed', color=props['color'],
             label=f'AGR_AF_CC = {agr[f"{vname}_AGR_AF_CC"].values:.2f}')
    axs.set_title(props['title'])
    axs.legend(loc='upper left')

def run():

    gr = load_data(gr_type='GR')
    agr = load_data(gr_type='AGR')

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    axs = axs.reshape(-1)

    vvars = ['EF', 'ED_avg', 'EM_avg', 'EA_avg', 'tEX', 'TEX']

    for ivar, vvar in enumerate(vvars):
        gr_vars = [grvar for grvar in gr.data_vars if vvar in grvar]
        agr_vars = [agrvar for agrvar in agr.data_vars if vvar in agrvar]
        gr_ds = gr[gr_vars]
        agr_ds = agr[agr_vars]
        plot_subplot(fig=fig, axs=axs[ivar], gr=gr_ds, agr=agr_ds, vname=vvar)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.3, wspace=0.3)
    plt.savefig('/nas/home/hst/work/TEA/plots/misc/AUT_ERA5_Tx99p_GRvsAGR.png', bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    run()
