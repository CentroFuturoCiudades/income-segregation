import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dask import delayed
from dask.distributed import Client
from estimate import get_seg_full
from numpy.random import default_rng
from pathlib import Path
from scipy.optimize import brentq


def ci_single(x, conf_level=0.95):
    """ Computes the single confidence interval of a variable
    estimated from a sample x using the quantiles. """

    x = np.asarray(x)
    if x.dtype == 'O':
        # wee need to stack arrays
        x = np.stack(x, axis=0)

    lower_t = (1 - conf_level)/2
    upper_t = 1 - lower_t

    return np.quantile(x, [lower_t, upper_t], axis=0).T


def ci_simultaneous(alpha_col, x, return_mask=False):
    """ Computes the simulatenous alpha for confidence intervales
    of a set of related variables from a sample x, where x has
    sample vectors as rows for a per column alpha value.

    The per column alpha is not the simultaneous alpha value.

    Returns the simulataneous alpha value, and confidence intervals.
    """
    a_lower = alpha_col/2
    a_upper = 1 - a_lower

    n_samples, _ = x.shape

    # Quantiles
    ci = np.quantile(x, [a_lower, a_upper], axis=0)

    # upper and lower bounds for all variables
    l = ci[0][None, :]
    u = ci[1][None, :]

    # Find rejected samples, not counting twice the same row
    rejected = np.logical_or(x < l, x > u)
    mask = np.any(rejected, axis=1)

    # Actual alpha level (rho) is the fractions of rejected samples
    rho = mask.sum()/n_samples

    if return_mask:
        return rho, mask
    else:
        return rho


def ci_opt(alpha_col, alpha_target, x):
    """ Wrapper function to optimize alpha_col."""
    return ci_simultaneous(alpha_col, x) - alpha_target


def search_ci_simultaneous(x, conf_level=0.95):
    x = np.asarray(x)
    if x.dtype == 'O':
        # wee need to stack arrays
        x = np.stack(x, axis=0)

    alpha = 1 - conf_level

    # Search for optimal alpha_col value
    y_low = ci_opt(1e-10, alpha, x)
    y_up = ci_opt(1.0, alpha, x)
    assert np.sign(y_low) != np.sign(y_up), 'You need more samples'

    alpha_col = brentq(ci_opt, 1e-10, 1.0, args=(alpha, x))
    rho, mask = ci_simultaneous(alpha_col, x, return_mask=True)

    ci = np.zeros((x.shape[1], 2))
    ci[:, 0] = np.min(x[~mask], axis=0)
    ci[:, 1] = np.max(x[~mask], axis=0)

    return alpha_col, rho, ci


def plot_ci(points_estimates, c_intervals, o_file=None):
    fig = plt.figure(figsize=(25, 6))

    for i, ci in enumerate(c_intervals):
        if ci[0] == ci[1]:
            continue
        plt.plot([i, i], ci, 'k')
        plt.plot(i, points_estimates[i], '.b')
    if o_file is not None:
        fig.savefig(o_file)


def get_bs_samples(n_samples, met_zone_codes,
                   opath, data_path='./data/',
                   q=5, k_list=[5, 100], seed=123456):
    opath = Path(opath)
    data_path = Path(data_path)

    # Run workflow with original sample, save all to disk
    print('Running with original sample ...')
    base_results = get_seg_full(
        met_zone_codes,
        q=q, 
        k_list=k_list,
        data_path=data_path, 
        out_path=opath,
        write_to_disk=True
    )
    print('Done.')

    if n_samples == 0:
        print('No bootstrap requested. Done.')
        return

    client = Client()

    meta = dd.utils.make_meta(base_results)

    survey = pd.read_csv(opath / 'survey.csv')
    n_ind = len(survey)

    # Seed for reproducibility
    rng = default_rng(seed=seed)

    # Get indices for all bs samples as an array
    bs_idxs = rng.integers(0, n_ind, size=(n_samples, n_ind))

    # Create dicts to store the H index and centralization indices
    # for each sample, as
    print(f'Runnning for {n_samples} bootstrap samples ...')
    bs_results = []
    for idxs in bs_idxs:
        # Run the full workflow
        results = delayed(get_seg_full)(
            met_zone_codes,
            q=q, 
            k_list=k_list,
            data_path=data_path,
            write_to_disk=False,
            bs_idxs=idxs
        )

    bs_results.append(results)
    results_df = dd.from_delayed(bs_results, meta=meta)
    results_df.to_parquet(opath / 'bs_results.parquet')
    print('Done.')

    client.close()
