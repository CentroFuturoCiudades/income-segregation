import yaml
import numpy as np
import pandas as pd

sun = pd.read_csv('data/Base_SUN_2018.csv', encoding='Latin-1')
groups = sun.groupby('CVE_SUN')
met_zones = {}

for CVE_SUN, df in groups:
    if not CVE_SUN.startswith('M'):
        continue
    CVE_ENT = int(df.CVE_ENT.iloc[0])
    CVE_MUN = [int(x) for x in np.unique(df.CVE_MUN % 1000)]
    if CVE_ENT in met_zones.keys():
        met_zones[CVE_ENT][CVE_SUN] = CVE_MUN
    else:
        met_zones[CVE_ENT] = {CVE_SUN: CVE_MUN}

with open('./output/met_zones.yaml', 'w') as f:
    yaml.dump(met_zones, f)
