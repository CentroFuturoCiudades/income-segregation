import os
import yaml
import numpy as np
import pandas as pd

sun = pd.read_csv('data/Base_SUN_2018.csv', encoding='Latin-1')
groups = sun.groupby('CVE_SUN')
met_zones = {}

for CVE_SUN, df in groups:
    if not CVE_SUN.startswith('M'):
        continue
    CVE_MUN = sorted(df.CVE_MUN.to_list())
    met_zones[CVE_SUN] = CVE_MUN

if not os.path.isdir('./output'):
    os.makedirs('./output')

with open('./output/met_zones.yaml', 'w') as f:
    yaml.dump(met_zones, f)
