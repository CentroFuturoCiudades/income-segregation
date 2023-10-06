import numpy as np

from scipy import integrate
from scipy.special import xlogy
from sklearn.neighbors import KDTree


def binary_entropy(p):
    p = np.asanyarray(p)
    e = np.zeros_like(p)
    mask = np.logical_and(0 < p, p < 1)
    pm = p[mask]
    e[mask] = -pm*np.log2(pm) - (1-pm)*np.log2(1-pm)
    return e


def local_binary_KL(p_local, p_global):
    p_local = np.asarray(p_local)
    p_global = np.asarray(p_global)

    KL = xlogy(p_local, p_local/p_global)
    # KL = p_local*np.log2(p_local/p_global)
    KL += xlogy(1 - p_local, (1 - p_local)/(1 - p_global))
    # KL += (1 - p_local)*np.log2((1 - p_local)/(1 - p_global))

    return KL/np.log(2)


def global_H_index(df_ind, agebs):
    # The probability distribution of income for each ageb, p(y|n)
    # with n indexing agebs
    df_prob = df_ind.reset_index(drop=True).groupby('Ingreso_orig').sum()
    df_prob = df_prob/df_prob.sum()

    # The cdf F(y|n) = \sum_{y'=0}^y p(y|n)
    df_cdf = df_prob.cumsum()

    # The local deviations for each ageb, for all percentiles of global
    # income distribution, (E(p) - E(p_n))/E(p)
    # local_deviations = df_cdf[agebs].apply(
    #     lambda x: local_bin_normalized_dev(x, df_cdf.w_MZ))
    local_kl = df_cdf[agebs].apply(
        lambda x: local_binary_KL(x, df_cdf.w_MZ))

    # Since the last row corresponds to F(y) = 1,
    # and reflects a single group, we drop it
    # local_deviations.drop(local_deviations.tail(1).index, inplace=True)
    local_kl.drop(local_kl.tail(1).index, inplace=True)

    # The fraction population of each ageb are the
    # probabilities p(n)
    pn = df_ind[agebs].sum() / df_ind.w_MZ.sum()

    # The entropy indices for each percentile are a weighted mean
    # of local deviations
    # entropy_index_df = local_deviations.multiply(pn).sum(axis=1)
    mean_kl_series = local_kl.multiply(pn).sum(axis=1)
    norm_H_series = mean_kl_series / binary_entropy(df_cdf.w_MZ.values[:-1])

    # But the above is not the function to integrate,
    # we must multiply by E(p) to recovr the
    # expected KL divergene.
    # Reme,ber remove last row
    # kl_df = entropy_index_df * binary_entropy(df_cdf.w_MZ.values[:-1])

    # Since the flutuations have been attenuated by the values of the
    # global entropy , it seems safe to integrate numerically the KL
    # function directly, despite the high level of noise at the tails
    # of H (see plots)
    H = integrate.simpson(y=mean_kl_series.values, x=df_cdf.w_MZ.values[:-1])

    # Return the cdf, the local_h, the expected kl, and H
    return (H, df_cdf, norm_H_series, mean_kl_series,
            local_kl.set_index(df_cdf.w_MZ.iloc[:-1]))


def local_cent(gdf, x_name='q_5', total_name='total_ipf'):

    # Get centroids as an array of x,y points
    # build and get sorted neighbors lists
    xp = gdf.geometry.centroid.x.values[:, None]
    yp = gdf.geometry.centroid.y.values[:, None]
    points = np.hstack([xp, yp])
    tree = KDTree(points)
    dlist, nlist = tree.query(points, k=len(points), sort_results=True, return_distance=True)

    # Get an array of population counts for the required quantile
    totals_list = gdf[total_name].values
    x_list = gdf[x_name].values
    y_list = totals_list - x_list

    # Create array to hold cent indices
    n = len(x_list)
    C = np.zeros((n, n))

    for i in range(n):
        # For location i, we need to sort the vectors
        i_idxs = nlist[i]
        x = x_list[i_idxs].cumsum()
        y = y_list[i_idxs].cumsum()

        # Get the cumulative populations
        XY = x*y

        # The shifted products
        x_j_1_y_j = x[:-1]*y[1:]
        x_j_y_j_1 = x[1:]*y[:-1]

        # The shifted cumsums
        X_j_1_Y_j = x_j_1_y_j.cumsum()
        X_j_Y_j_1 = x_j_y_j_1.cumsum()

        # The index array for all scales
        for k in range(1, len(x)):
            C[i, k] = (X_j_1_Y_j[k-1] - X_j_Y_j_1[k-1])/XY[k]

    return C, nlist, dlist
