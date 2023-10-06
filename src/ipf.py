import numpy as np
import pandas as pd
import geopandas as gpd


def weight_ind(ctable, df, column_name='weight'):
    # Appends weight column in place to df
    # Super slow due to inneficient lookups
    # on duplicated multi index

    # Make a df from ctable
    # Should have same multi index as df
    ctable_df = ctable.to_dataframe('counts')
    ctable_df = ctable_df[ctable_df.counts > 0]

    # Create weight column
    df[column_name] = 0

    # Apply weights uniformly distributed on repeated
    # individuals of the same class
    for idx, row in ctable_df.iterrows():
        single_class_df = df.loc[idx]
        n_ind = len(single_class_df)
        df.loc[idx, column_name] = row.counts/n_ind


def weighted_mean(x, w):
    # Sort both arrays acording to x values
    # and normalize weights
    idxs = np.argsort(x)
    x = x[idxs]
    w = w[idxs]/w.sum()

    # Find mid value
    ws = np.cumsum(w)
    idx = ws.searchsorted(0.5)

    return x[idx]


def get_marginals(seed, ageb):
    # create tuples (dim, i, value) where
    # x[..., dim=i, ...].sum() = value
    marginals_np = []
    for dim, (k, v) in enumerate(seed.coords.items()):
        for i, vv in enumerate([f'{k}_{vv}' for vv in v.values]):
            if vv in ageb.index:
                marginals_np.append((dim, i, ageb[vv]))
    return marginals_np


def print_marginals(ctable, marginals):
    for constraint in marginals:
        dim, i, m_count = constraint
        idxs = [slice(None)]*ctable.ndim
        idxs[dim] = i
        idxs = tuple(idxs)
        c_count = ctable[idxs].sum()
        print(dim, i, m_count, c_count.item())


def ipf(seed, marginals, maxiters=100, rel_tol=1e-5):
    ctable = seed.values.copy()
    status = 'not converged'
    for niter in range(maxiters):
        for constraint in marginals:
            dim, i, m_count = constraint
            idxs = [slice(None)]*ctable.ndim
            idxs[dim] = i
            idxs = tuple(idxs)

            c_count = ctable[idxs].sum()
            if c_count == 0:
                continue
            ctable[idxs] *= m_count / c_count

        # Evaluate convergence
        delta = 0.0
        for constraint in marginals:
            dim, i, m_count = constraint
            idxs = [slice(None)]*ctable.ndim
            idxs[dim] = i
            idxs = tuple(idxs)

            if m_count == 0:
                pass
            else:
                c_count = ctable[idxs].sum()
                delta = max(delta, abs(1 - c_count/m_count))
        if delta <= rel_tol:
            status = 'converged'
            break
    return seed.copy(data=ctable), niter, status


def apply_ipf(df_censo, seed):
    niter_list = []
    status_list = []
    ageb_dict = {}
    for cvegeo, ageb in df_censo.iterrows():
        marginals = get_marginals(seed, ageb)
        ctable, niter, status = ipf(seed, marginals)
        ageb_dict[cvegeo] = ctable
        niter_list.append(niter)
        status_list.append(status)

    df_status = pd.DataFrame({'niter': niter_list, 'status': status_list})

    return ageb_dict, df_status


def weight_ind_fast(df, ds, agebs):
    # Crete multiindex dataframe from contingency tables with
    # counts for each variable combination
    ctables_df = pd.concat(
        [ctable.to_dataframe(cvegeo)
         for cvegeo, ctable in ds.items()
         if cvegeo in agebs],
        axis=1)
    # Drop zero counts which are missing from survey
    ctables_df = ctables_df[ctables_df.sum(axis=1) > 0]

    # Ageb list
    columns = ctables_df.columns

    # Add zeroed columns for all agebs and
    # Create the weight table
    df_w = pd.concat(
        [df,
         pd.DataFrame(
             data=np.zeros((len(df), len(columns))),
             index=df.index,
             columns=columns)],
        axis=1)

    # Assign appropriate weight to individuals
    # for each unique combination of variables
    for idx, row in ctables_df.iterrows():
        # This is prbably much faster using integer indexing

        # A small dataframe with individuals with the same
        # combination of variables
        # single_class_df = df_w.loc[idx]
        n_ind = len(df_w.loc[idx])

        # Distribute weight uniformly among all individuals
        df_w.loc[idx, columns] = np.broadcast_to(
            row.values/n_ind, (n_ind, len(row))
        )
        # df_w.loc[idx] = single_class_df

    return df_w


def get_income_df(ds, df_censo, df_ind, data_path, agebs):
    to_concat = []
    dim = [d for d in ds.dims if d != 'Ingreso']
    for cvegeo, ctable in ds.items():
        if cvegeo in agebs:
            df = ctable.sum(dim=dim).to_dataframe(name=cvegeo).T
            to_concat.append(df)
            
    pop_income = pd.concat(to_concat)
    pop_income['total_ipf'] = pop_income.sum(axis=1)

    pop_income.index.name = 'cvegeo'
    pop_income = pop_income.join(df_censo['P_15YMAS'])
    pop_income.rename(columns={'P_15YMAS': 'total_census'})

    income_by_ageb = df_ind[agebs].multiply(
        df_ind['Ingreso_orig'], axis='index').sum()
    income_by_ageb.name = 'income'
    pop_income = pop_income.join(income_by_ageb)

    pop_income['income_pc'] = pop_income.income/pop_income.total_ipf

    # Import geo data
    scodes = np.unique([a[:2] for a in agebs])
    agebs_gdf = pd.concat([
        gpd.read_file(
            f'{data_path}/agebs.zip', layer=f'{scode}a')
        for scode in scodes
    ])
    agebs_gdf = agebs_gdf.to_crs(agebs_gdf.estimate_utm_crs())
    agebs_gdf.columns = [i.lower() for i in agebs_gdf.columns]
    agebs_gdf.set_index('cvegeo', inplace=True)
    agebs_gdf = agebs_gdf[['geometry']]

    income_gdf = agebs_gdf.join(pop_income, how='right')
    income_gdf.reset_index(inplace=True)
    income_gdf.rename(columns={i: f'q_{i}' for i in range(10)},
                      inplace=True)

    return income_gdf
