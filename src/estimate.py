import preprocessing
import ipf
import seg
import pickle
import warnings

import pandas as pd
import numpy as np
import xarray as xr

from pathlib import Path


def flatten_res(res, prefix=''):
    new = {}
    for k, v in res.items():
        if not isinstance(v, dict):
            new[prefix + k] = v
        else:
            new = new | flatten_res(v, prefix=f'{prefix}{k}.')
    return new


def res2pd(r):
    return pd.DataFrame.from_dict(
        {k: [v] for k, v in flatten_res(r).items()}
    )


def reshape_results(results_dict):
    results_reshaped = dict(
        median_MZ = [results_dict["median_MZ"]],
        H = [results_dict["H"]]
    )

    for quantile in results_dict["cent_idx"]:
        for k in results_dict["cent_idx"][quantile]:
            for i, elem in enumerate(results_dict["cent_idx"][quantile][k]):
                key = f"cent_idx.{quantile}.{k}.{i}"
                results_reshaped[key] = [elem]

    results_reshaped = pd.DataFrame.from_dict(results_reshaped)
    return results_reshaped


def get_seg_full(
        met_zone_codes,
        linking_cols = ['Sexo', 'Edad', 'Nivel', 'SeguroIMSS', 'SeguroPriv', 'ConexionInt'],
        q = 5,
        data_path = './data/', 
        out_path = None,
        bs_idxs = None, 
        write_to_disk = False,
        k_list = [5, 100]
):

    # Configure out_path variables
    if out_path is not None:
        out_path = Path(out_path)
    # Create results dict
    results_dict = {}

    # Load survey with processed linking and target columns
    # This discretizes income into q quantiles into
    # columns Ingreso and keeps continuos income in Ingreso_orig.
    df_survey = preprocessing.load_survey(
        data_path, met_zone_codes, linking_cols, q)
    # Bootstrap resampling
    if bs_idxs is not None:
        df_survey = df_survey.iloc[bs_idxs].reset_index(drop=True)
    if write_to_disk:
        df_survey.to_csv(out_path / 'survey.csv')

    # Load cesus data
    df_censo = preprocessing.load_census(data_path, met_zone_codes)
    if write_to_disk:
        df_censo.to_csv(out_path / 'census.csv')

    # Create the global contingency table/distribution
    seed_xr = pd.crosstab(
        [df_survey[c] for c in linking_cols],
        df_survey['Ingreso'],
        dropna=False
    ).stack().to_xarray().astype(float)

    # Apply ipf to each zone
    # Return a dictionary with ctable for each ageb (cvegeo)
    ds, status = ipf.apply_ipf(df_censo, seed_xr)
    assert 'not converged' not in status
    ds = xr.Dataset(ds)
    agebs = list(ds.keys())
    ds['seed'] = seed_xr
    if write_to_disk:
        ds.to_netcdf(path=out_path / 'contingency_tables.nc',
                     engine='netcdf4')

    # Create a df of individuals from survey dataframe
    # with multi index on linking categories
    df_ind = df_survey.set_index(
        list(df_survey.columns.drop('Ingreso_orig')))
    df_ind.sort_index(inplace=True)
    # Add individual weights for each ageb
    df_ind = ipf.weight_ind_fast(df_ind, ds, agebs)
    # Sum weights for all met zone
    df_ind['w_MZ'] = df_ind[agebs].sum(axis=1)
    if write_to_disk:
        df_ind.to_csv(out_path / 'weight_tables.csv')

    # Find weighted median
    median_MZ = ipf.weighted_mean(df_ind.Ingreso_orig.values,
                                  df_ind.w_MZ.values)
    results_dict['median_MZ'] = median_MZ

    # Get global segregations and segregation profile
    # H is the global index
    # df_cdf contains the empirical cdf for all agebs (columns)
    # entropy_index_df contains the binary H index for all
    # empirical partitions p/1-p.
    # kl_df the same as above but unormalized, so E[KL] over all agebs
    # local_dev_df: contains local H deviations for all agebs(cols) for
    # all percentiles (rows)
    (
        H, 
        df_cdf, 
        norm_H_series,
        mean_kl_series, 
        local_kl
    ) = seg.global_H_index(df_ind, agebs)

    results_dict['H'] = H
    if write_to_disk:
        df_cdf.to_csv(out_path / 'ecdf_income_per_ageb.csv')
        norm_H_series.to_csv(out_path / 'H_index_per_percentile.csv')
        mean_kl_series.to_csv(out_path / 'mean_KL_per_percentile.csv')
        local_kl.to_csv(out_path / 'KL_per_ageb_per_pecentile.csv')

    # Create a dataframe with population per income bracket per ageb
    # Marginalizing over all other variables in all local
    # contingency tables.
    
    # Also calculates total and per capita income
    pop_income = ipf.get_income_df(ds, df_censo, df_ind, data_path, agebs)
    
    # Keep only agebs witg geometry (error in marco geo?)
    agebs = pop_income[~pop_income.geometry.isna()].cvegeo.to_list()
    pop_income = pop_income.dropna()
    if write_to_disk:
        pop_income.to_file(out_path / 'income_quantiles.gpkg')

    # Find local centralization index for top and low percentiles
    cent_idx_dict = {}
    C_list = []
    for qq in range(1, q+1):
        xname = f'q_{qq}'
        cent_idx_dict[xname] = {}
        C, nlist, dlist = seg.local_cent(pop_income, x_name=xname)

        max_k = C.shape[1] - 1
        for k in k_list:
            if k > max_k:
                k = max_k
                warnings.warn('k greater than number of entries in DataFrame. Value has been automatically adjusted.')
            cent_idx_dict[xname][f'k_{k}'] = C[:, k].copy()

        if write_to_disk:
            C_list.append(C)

    results_dict['cent_idx'] = cent_idx_dict
    if write_to_disk:
        C_xr = xr.DataArray(data=np.stack(C_list, axis=0),
                            coords={
                                'income_quantile': list(range(1, q+1)),
                                'ageb': agebs,
                                'k_neighbors': list(range(len(agebs)))
                            })
        n_info = xr.DataArray(data=np.stack([nlist, dlist]),
                              coords={
                                  'info': ['n_idx', 'n_distance'],
                                  'ageb': agebs,
                                  'k_neighbors': list(range(len(agebs)))
                              })
        C_ds = xr.Dataset({'centrality': C_xr, 'n_info': n_info})
        C_ds.to_netcdf(path=out_path / 'centrality_index.nc',
                       engine='netcdf4')

    # Return a dataframe
    results = reshape_results(results_dict)
    if write_to_disk:
        # Store results as a pickle object
        with open(out_path / 'results.pkl', 'wb') as f:
            pickle.dump(results, f)

    return results
