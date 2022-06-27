import argparse
from bootstrap import get_bs_samples
from plots import make_all


states = {
    '01': 'aguascalientes',
    '02': 'bajacalifornia',
    '03': 'bajacaliforniasur',
    '04': 'campeche',
    '05': 'coahuiladezaragoza',
    '06': 'colima',
    '07': 'chiapas',
    '08': 'chihuahua',
    '09': 'ciudaddemexico',
    '10': 'durango',
    '11': 'guanajuato',
    '12': 'guerrero',
    '13': 'hidalgo',
    '14': 'jalisco',
    '15': 'mexico',
    '16': 'michoacandeocampo',
    '17': 'morelos',
    '18': 'nayarit',
    '19': 'nuevoleon',
    '20': 'oaxaca',
    '21': 'puebla',
    '22': 'queretaro',
    '23': 'quintanaroo',
    '24': 'sanluispotosi',
    '25': 'sinaloa',
    '26': 'sonora',
    '27': 'tabasco',
    '28': 'tamaulipas',
    '29': 'tlaxcala',
    '30': 'veracruzignaciodelallave',
    '31': 'yucatan',
    '32': 'zacatecas'
}

met_zones = {
    2: [3, 4, 5],
    11: [2, 37],
    14: [2, 39, 44, 51, 7, 97, 98, 101, 12, 124],
    19:  [1,  6,  9, 10, 12, 18, 19, 21, 25,
          26, 31, 39, 41, 45, 46, 47, 48, 49],
    22: [6, 8, 11, 14],
    31: [2, 13, 38, 41, 5, 63, 9, 93, 95, 1, 101]
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
    parser.add_argument('state_code', type=check_state,
                        help="Mexican state for calculate indices for."
                        " Must be an integer between 1 and 32.")
    parser.add_argument('-n', '--n_samples', type=check_positive, default=0,
                        help="Number of bootstrap samples to use,"
                        "defaults to 0 for no bootstraping.")
    parser.add_argument('--plot', action='store_true',
                        help="Make plots for the respective sate. "
                        "Assumes output files have been created.")

    args = parser.parse_args()
    assert args.n_samples <= 10000
    print(f"Initiating estimation for state {args.state_code} with"
          f" {args.n_samples} samples.")

    opath = f'./output/{args.state_code}/'
    ipath = './data/'
    met_zone_codes = met_zones[args.state_code]
    assert False
    if args.plot:
        make_all(args.state_code, met_zone_codes,
                 opath, ipath)
    else:
        get_bs_samples(args.n_samples,
                       args.state_code,
                       met_zone_codes,
                       opath=opath,
                       data_path=ipath,
                       q=5, k_list=[5, 100])
