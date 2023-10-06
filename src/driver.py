import argparse
import os
import time
import yaml

from bootstrap import get_bs_samples
from plots import make_all
from pathlib import Path


def check_positive(value):
    try:
        value = int(value)
        if value < 0:
            raise argparse.ArgumentTypeError(
                "{} is not a postivive integer.".format(value))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "{} is not a postivive integer.".format(value))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Estimate segregation indices using IPF '
        'with bootstraping confidence intervals.')
    parser.add_argument(
        'CVE_SUN',
        help="Metropolitan zone identifier from the national urban system (SUN). See met_zones.yaml for a list."
    )
    parser.add_argument(
        '-n', 
        '--n_samples', 
        type=check_positive, 
        default=0,
        help="Number of bootstrap samples to use, defaults to 0 for no bootstraping."
    )
    parser.add_argument(
        '--plot', 
        action='store_true',
        help="Make plots for the respective state. Assumes output files have been created."
    )
    parser.add_argument(
        "--time", 
        action="store_true", 
        help="Print total execution time.",
    )
    parser.add_argument(
        "--seed",
        default=123456,
        type=int,
        help="Seed for random number generation."
    )

    args = parser.parse_args()
    assert args.n_samples <= 10000
    print(f"Initiating estimation for metropolitan zone {args.CVE_SUN}"
          f" with {args.n_samples} samples.")

    # Load met_zones

    if not os.path.exists('./output/met_zones.yaml'):
        raise Exception("met_zones.yaml not found. Run get_met_zones.py first.")
    
    with open('./output/met_zones.yaml', 'r') as f:
        met_zones = yaml.safe_load(f)
    met_zone_codes = met_zones[args.CVE_SUN]

    opath = Path(f'./output/{args.CVE_SUN}/')
    if not opath.exists():
        opath.mkdir(parents=True, exist_ok=True)
    ipath = Path('./data/')

    if args.plot:
        make_all(met_zone_codes, opath, ipath)
    else:
        start_time = time.time()
        get_bs_samples(
            args.n_samples,
            met_zone_codes,
            opath=opath,
            data_path=ipath,
            q=5, 
            k_list=[5, 100],
            seed=args.seed
        )
        stop_time = time.time()

        if args.time:
            print(f"Total time: {stop_time - start_time}")