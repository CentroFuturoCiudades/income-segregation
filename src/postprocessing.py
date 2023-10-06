import pandas as pd
from pathlib import Path
import bootstrap
import pickle

opath = Path('output/')
met_paths = sorted(list(opath.glob('M*.*')))

met_res = []
for mpath in met_paths:
    with open(mpath / 'results.pkl', 'rb') as f:
        results = pickle.load(f)
    point_H = results.H.iloc[0]
    point_med_income = results.median_MZ.iloc[0]

    res_bs = pd.read_parquet(
        mpath / 'bs_results.parquet/'
    ).reset_index(drop=True)
    median_ci = bootstrap.ci_single(res_bs['median_MZ'])
    H_ci = bootstrap.ci_single(res_bs['H'])

    met_res.append(
        {
            'met_zone': mpath.name,
            'H': point_H,
            'H_low': H_ci[0],
            'H_high': H_ci[1],
            'median_income': point_med_income,
            'median_low': median_ci[0],
            'median_high': median_ci[1],
        }
    )

res_df = pd.DataFrame(met_res)
res_df.to_csv(opath / 'met_zones_results.csv', index=False)
