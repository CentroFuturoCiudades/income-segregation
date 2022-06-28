import argparse
from bootstrap import get_bs_samples
from plots import make_all
from pathlib import Path
import yaml


met_zones = {
    2: [3, 4, 5],  # Tijuana
    14: [2, 39, 44, 51, 70, 97, 98, 101, 120, 124],  # Guadalajara
    19:  [1,  6,  9, 10, 12, 18, 19, 21, 25,
          26, 31, 39, 41, 45, 46, 47, 48, 49],  # Monterrey
    22: [6, 8, 11, 14],  # Querétaro (missing one, 11004)
    31: [2, 13, 38, 41, 50, 63, 90, 93, 95, 100, 101]  # Mérida
}


def check_state(value):
    try:
        value = int(value)
        if value > 32 or value < 1:
            raise argparse.ArgumentTypeError(
                "{} is not a valid state code.".format(value))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "{} is not a valid state code.".format(value))
    return value


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
    parser.add_argument('CVE_SUN',
                        help="Metropolitan zone identifier from "
                        "the national urban system (SUN)."
                        " See met_zones.yaml for a list.")
    parser.add_argument('-n', '--n_samples', type=check_positive, default=0,
                        help="Number of bootstrap samples to use,"
                        "defaults to 0 for no bootstraping.")
    parser.add_argument('--plot', action='store_true',
                        help="Make plots for the respective sate. "
                        "Assumes output files have been created.")

    args = parser.parse_args()
    assert args.n_samples <= 10000
    print(f"Initiating estimation for metropolitan zone {args.CVE_SUN}"
          f" with {args.n_samples} samples.")

    # Load met_zones
    with open('./output/met_zones.yaml', 'r') as f:
        met_zones = yaml.safe_load(f)
    met_zone_codes = met_zones[args.CVE_SUN]

    opath = Path(f'./output/{args.CVE_SUN}/')
    ipath = Path('./data/')

    if args.plot:
        make_all(met_zone_codes,
                 opath, ipath)
    else:
        get_bs_samples(args.n_samples,
                       met_zone_codes,
                       opath=opath,
                       data_path=ipath,
                       q=5, k_list=[5, 100])
