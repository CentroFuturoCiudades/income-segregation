import numpy as np
import xarray as xr
import pandas as pd
from scipy import integrate


def binary_entropy(p):
    p = np.asanyarray(p)
    e = np.zeros_like(p)
    mask = np.logical_and(0 < p, p < 1)
    pm = p[mask]
    e[mask] = -pm*np.log2(pm) - (1-pm)*np.log2(1-pm)
    return e


def local_bin_normalized_dev(p_local, p_global):
    p_local = np.asarray(p_local)
    p_global = np.asarray(p_global)

    x = binary_entropy(p_global) - binary_entropy(p_local)
    x /= binary_entropy(p_global)
    return x


def binary_entropy_index(p_local, px, p_global):
    p_local = np.asarray(p_local)
    px = np.asarray(px)
    p_global = np.asarray(p_global)

    return np.sum(px*local_bin_normalized_dev(p_local, p_global))


def global_H_index(df_ind, agebs):
    # The probability distribution of income for each ageb, p(y|n)
    # with n indexing agebs
    df_prob = df_ind.reset_index(drop=True).groupby('Ingreso_orig').sum()
    df_prob = df_prob/df_prob.sum()

    # The cdf F(y|n) = \sum_{y'=0}^y p(y|n)
    df_cdf = df_prob.cumsum()

    # The local deviations for each ageb, for all percentiles of global
    # income distribution, (E(p) - E(p_n))/E(p)
    local_deviations = df_cdf[agebs].apply(
        lambda x: local_bin_normalized_dev(x, df_cdf.w_MZ))

    # Since the last row corresponds to F(y) = 1,
    # and reflects a single group, we drop it
    local_deviations.drop(local_deviations.tail(1).index, inplace=True)

    # The fraction population of each ageb are the
    # probabilities p(n)
    pn = df_ind[agebs].sum() / df_ind.w_MZ.sum()

    # The entropy indices for each percentile are a weighted mean
    # of local deviations
    entropy_index_df = local_deviations.multiply(pn).sum(axis=1)

    # But the above is not the function to integrate,
    # we must multiply by E(p) to recovr the
    # expected KL divergene.
    # Reme,ber remove last row
    kl_df = entropy_index_df * binary_entropy(df_cdf.w_MZ.values[:-1])

    # Since the flutuations have been attenuated by the values of the
    # global entropy , it seems safe to integrate numerically the KL
    # function directly, despite the high level of noise at the tails
    # of H (see plots)
    H = integrate.simpson(y=kl_df.values, x=df_cdf.w_MZ.values[:-1])

    # Return the cdf, the local_h, the expected kl, and H
    return H, df_cdf, entropy_index_df, kl_df
