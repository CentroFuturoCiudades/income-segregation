import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
import datashader as ds
import holoviews as hv
from holoviews.operation.datashader import datashade
import pandas as pd
import geopandas as gpd
import bootstrap
from pathlib import Path
import pickle
from collections import defaultdict


WIDTH = 6.4
HEIGHT = 4.8
DPI = 600
LW = 0.2

plt.rcdefaults()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.sans-serif": ["Times"],
    "font.size": 14,
    "ps.usedistiller": "xpdf",
    'ps.fonttype': 42,
    'pdf.fonttype': 42
    })
mpl.use('agg')


def plot_H_KL(df_cdf, norm_H_series, mean_kl_series, fig_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, HEIGHT), dpi=DPI)
    ax.set_ylabel(r'$H$')
    ax.set_xlabel('Percentile rank p')
    ax.plot(df_cdf.w_MZ.values[:-1], norm_H_series.values, color='k')

    ax_2 = ax.twinx()
    ax_2.spines['right'].set_color('grey')
    ax_2.tick_params(axis='y', colors='grey')
    ax_2.yaxis.label.set_color('grey')
    ax_2.set_ylabel(
        r'$E\left[D_{KL}\left(Y_p |\  J\parallel Y_p\right)\right]$')
    ax_2.plot(df_cdf.w_MZ.values[:-1], mean_kl_series.values,
              color='grey')

    ax_ins = inset_axes(ax, width='35%', height='35%', loc=2)
    ax_ins.yaxis.tick_right()
    ax_ins.yaxis.set_label_position("right")
    ax_ins.set_ylabel('$F(y)$')
    df_cdf.set_index(df_cdf.index, drop=True).plot(
        legend=False, ax=ax_ins, color='grey', alpha=0.5)
    ax_ins.semilogx(df_cdf.index, df_cdf.w_MZ, color='k', lw=2)
    ax_ins.set_xlabel('$y$')

    plt.tight_layout()
    if fig_path is not None:
        # fig.savefig(fig_path / 'cont_variables.eps', dpi=DPI)
        fig.savefig(fig_path / 'cont_variables.pdf', dpi=DPI)


def plot_local_c_profiles(C_ds, extension='bokeh', q=5):
    C = C_ds['centrality'].sel(income_quantile=q).values
    df = ds.utils.dataframe_from_multiple_sequences(
        np.arange(C.shape[1]), C)

    hv.extension(extension)
    profile = datashade(hv.Curve(df), cnorm='eq_hist')
    profile = profile.options(width=1000, height=400)

    return profile


def get_missing_agebs(met_zone_codes, data_path, pop_income):

    scodes = np.unique([c // 1000 for c in met_zone_codes])
    agebs_gdf = pd.concat([
        gpd.read_file(
            f'{data_path}/agebs.zip', layer=f'{scode:02d}a')
        for scode in scodes
    ])

    agebs_gdf = agebs_gdf.to_crs(agebs_gdf.estimate_utm_crs())
    agebs_gdf['CVE_MZ'] = agebs_gdf.CVE_ENT.astype(int)*1000 \
        + agebs_gdf.CVE_MUN.astype(int)
    agebs_gdf = agebs_gdf[agebs_gdf['CVE_MZ'].isin(met_zone_codes)]
    agebs_gdf.columns = [i.lower() for i in agebs_gdf.columns]
    agebs_df = agebs_gdf[['cvegeo', 'geometry']].copy()
    missing_agebs = pd.merge(right=pop_income, left=agebs_df,
                             on='cvegeo', how='left')
    missing_agebs = missing_agebs[missing_agebs.isna().any(axis=1)]
    missing_agebs = missing_agebs[['cvegeo', 'geometry_x']].copy()
    missing_agebs = gpd.GeoDataFrame(missing_agebs)
    missing_agebs = missing_agebs.set_geometry('geometry_x')

    return missing_agebs


def plot_income_pc(pop_income, met_zone_codes,
                   data_path, fig_path=None):

    minx, miny, maxx, maxy = pop_income.total_bounds
    ratio = (maxy - miny)/(maxx-minx)

    fig, ax = plt.subplots(1, 2,
                           figsize=(WIDTH, WIDTH*ratio), dpi=DPI,
                           gridspec_kw=dict(hspace=0, wspace=0,
                                            width_ratios=[1, 0.05]))

    pop_income = pop_income.copy()
    # To thousands of USD
    pop_income['income_pc'] = pop_income['income_pc']/20/1000
    ax[0].set_axis_off()
    pop_income.plot(column='income_pc', legend=True, ax=ax[0], cax=ax[1],
                    cmap='viridis', edgecolor='grey', linewidth=LW)

    missing_agebs = get_missing_agebs(
        met_zone_codes, data_path, pop_income)
    missing_agebs.plot(ax=ax[0], facecolor='lightgrey',
                       edgecolor='grey', linewidth=LW)

    if fig_path is not None:
        # fig.savefig(fig_path / 'income_pc.eps', dpi=DPI)
        fig.savefig(fig_path / 'income_pc.pdf', dpi=DPI)


def get_not_significant_mask(res_bs, q, k):
    c = f'cent_idx.q_{q}.k_{k}'
    ci = bootstrap.ci_single(res_bs[c], conf_level=0.99)
    mask = (np.sign(ci[:, 0]) != np.sign(ci[:, 1]))
    return mask


def plot_income_q(pop_income, C_ds, res_bs, ax, q, k, vmax):
    pop_income = pop_income.copy()

    ax.set_axis_off()
    ax.tick_params(axis='both', which='both',
                   bottom=False, labelbottom=False,
                   left=False, labelleft=False)
    col_name = f'local_centralization_q_{q}_k_{k}'
    pop_income[col_name] = C_ds['centrality'].sel(
        income_quantile=q, k_neighbors=k)
    pop_income.plot(column=col_name, legend=False, ax=ax,
                    cmap='RdBu', edgecolor='grey', linewidth=LW,
                    vmin=-vmax, vmax=vmax)
    ax.text(0.05, 0.05, f'Q: {q}, k: {k}', transform=ax.transAxes)

    # Mark non significant agebs
    mask = get_not_significant_mask(res_bs, q, k)
    mask_gdf = pop_income[mask]
    mask_gdf.plot(legend=False, ax=ax,
                  edgecolor='k', facecolor='none',
                  linewidth=LW)


def plot_cent_idxs(pop_income, C_ds, res_bs, fig_path=None):
    minx, miny, maxx, maxy = pop_income.total_bounds
    ratio = (maxy - miny)/(maxx-minx)

    fig, ax = plt.subplots(
        2, 3,
        figsize=(WIDTH, WIDTH*ratio), dpi=DPI,
        gridspec_kw=dict(hspace=0.02, wspace=0,
                         width_ratios=[1, 1, 0.05]))
    # Find maximum abs index
    max_c = abs(
        C_ds['centrality'].sel(
            income_quantile=[1, 5], k_neighbors=[5, 100])).max().item()

    plot_income_q(pop_income, C_ds, res_bs,
                  ax=ax[0, 0], q=1, k=5, vmax=max_c)
    plot_income_q(pop_income, C_ds, res_bs,
                  ax=ax[0, 1], q=1, k=100, vmax=max_c)
    plot_income_q(pop_income, C_ds, res_bs,
                  ax=ax[1, 0], q=5, k=5, vmax=max_c)
    plot_income_q(pop_income, C_ds, res_bs,
                  ax=ax[1, 1], q=5, k=100, vmax=max_c)

    gs = ax[0, 0].get_gridspec()
    cax = fig.add_subplot(gs[:, 2])
    # remove the underlying axes
    for axx in ax[:, 2]:
        axx.remove()

    sm = cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=-max_c, vmax=max_c),
        cmap='RdBu')
    plt.colorbar(sm, cax=cax)

    if fig_path is not None:
        # fig.savefig(fig_path / 'centrality.eps', dpi=DPI)
        fig.savefig(fig_path / 'centrality.pdf', dpi=DPI)


def plot_ci(points_estimates, c_intervals, ax, q, k):
    for i, ci in enumerate(c_intervals):
        if ci[0] == ci[1]:
            continue
        ax.plot([i, i], ci, 'grey')
    for i, ci in enumerate(c_intervals):
        ax.plot(i, points_estimates[i], '.k')
    ax.axhline(ls='--', c='k')
    ax.text(0.55, 0.05, f'Q: {q}, k: {k}', transform=ax.transAxes)


def plot_cis(results, res_bs,  fig_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, WIDTH),
                             dpi=DPI, sharex=True, sharey=True)
    axes = axes.ravel()

    c_list = ['cent_idx.q_1.k_5', 'cent_idx.q_1.k_100',
              'cent_idx.q_5.k_5', 'cent_idx.q_5.k_100']
    for c, ax in zip(c_list, axes):
        q = c.split('q_')[1].split('.')[0]
        k = c.split('k_')[1].split('.')[0]
        r = results[c].iloc[0]
        idx = np.argsort(r)
        ci = bootstrap.ci_single(res_bs[c], conf_level=0.99)
        plot_ci(r[idx], ci[idx], ax=ax, q=q, k=k)
    plt.tight_layout()

    if fig_path is not None:
        # fig.savefig(fig_path / 'conf_intervals.eps', dpi=DPI)
        fig.savefig(fig_path / 'conf_intervals.pdf', dpi=DPI)


def make_all(met_zone_codes, opath, inpath):
    fig_path = Path(opath / 'figures')

    print('Making figures for metropolitan zone ...')
    print('Loading data ...')
    with open(opath / 'results.pkl', 'rb') as f:
        results = pickle.load(f)
    pop_income = gpd.read_file(opath / 'income_quantiles.gpkg')
    df_cdf = pd.read_csv(opath / 'ecdf_income_per_ageb.csv',
                         index_col='Ingreso_orig')
    norm_H_series = pd.read_csv(opath / 'H_index_per_percentile.csv',
                                index_col='Ingreso_orig')
    mean_kl_series = pd.read_csv(opath / 'mean_KL_per_percentile.csv',
                                 index_col='Ingreso_orig')
    C_ds = xr.open_dataset(opath / 'centrality_index.nc')
    res_bs = pd.read_parquet(
        opath / 'bs_results.parquet/').reset_index(drop=True)
    print('Done.')

    print('Making plot of continuous variables ...')
    plot_H_KL(df_cdf, norm_H_series, mean_kl_series, fig_path)
    print('Done.')
    print('Making plot of income per capita ...')
    plot_income_pc(pop_income, met_zone_codes,
                   inpath, fig_path)
    print('Done.')
    print('Making plot of centrality index ...')
    plot_cent_idxs(pop_income, C_ds, res_bs, fig_path)
    print('Done.')
    print('Making plot of conficende intervals ...')
    plot_cis(results, res_bs,  fig_path)
    print('Done.')
