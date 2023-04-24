import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict


def load_survey(data_path, met_zone_codes, linking_cols, q):

    data_path = Path(data_path)

    # Folio de vivienda
    path_folio = data_path / 'survey/enigh2018_ns_viviendas_csv.zip'
    # Ingreso
    path_ing = data_path / 'survey/enigh2018_ns_ingresos_csv.zip'
    # conjunto de hogares para obtener la info de conexión a internet
    path_hog = data_path / 'survey/enigh2018_ns_hogares_csv.zip'
    # Información demogŕafica
    path_pob = data_path / 'survey/enigh2018_ns_poblacion_csv.zip'

    df_folio = pd.read_csv(
        path_folio,
        usecols=['folioviv', 'ubica_geo'])
    
    df_ing = pd.read_csv(
        path_ing,
        usecols = ['folioviv', 'foliohog', 'numren', 'ing_tri'],
        dtype = {
            'folioviv': int,
            'foliohog': int,
            'numren': int,
            'ing_tri': pd.Float64Dtype()
        }
    )
    
    df_hog = pd.read_csv(
        path_hog,
        usecols = ['folioviv', 'foliohog', 'conex_inte'],
        dtype = {
            'folioviv': int,
            'foliohog': int,
            'conex_inte': pd.Int64Dtype()
        }
    )
    
    df_poblacion = pd.read_csv(
        path_pob,
        usecols=['folioviv', 'foliohog', 'numren', 'sexo',
                 'edad', 'edo_conyug', 'nivelaprob',
                 'inst_1', 'inst_6'],
        na_values=[' '],
        dtype = {
            'folioviv': int,
            'foliohog': int,
            'numren': int,
            'sexo': pd.Int64Dtype(),
            'edad': pd.Int64Dtype(),
            'edo_conyug': pd.Int64Dtype(),
            'nivelaprob': pd.Int64Dtype(),
            'inst_1': pd.Int64Dtype(),
            'inst_6': pd.Int64Dtype()
        }
    ).fillna(0)

    # Agregate income for duplicate individuals
    ing_agg = df_ing.groupby(
        ['folioviv', 'foliohog', 'numren']).agg(sum).reset_index()

    # Filter for state and metropolitan zone
    df_location = df_folio[df_folio.ubica_geo.isin(met_zone_codes)]

    # Filter for working population and add location data
    df_poblacion = df_poblacion[df_poblacion['edad'] >= 15]

    # Merge all dataframes
    df_ind_orig = pd.merge(df_location, df_poblacion, how='left')
    df_ind_orig = pd.merge(df_ind_orig, ing_agg)
    df_ind_orig = pd.merge(df_ind_orig, df_hog)
    df_ind_orig = df_ind_orig.reset_index(drop=True)

    # Rename columns to match link and targer variable names
    rename_dict = {
        'des_mun': 'Municipio',
        'sexo': 'Sexo',
        'edad': 'Edad',
        'nivelaprob': 'Nivel',
        'edo_conyug': 'EstadoConyu',
        'inst_1': 'SeguroIMSS',
        'inst_6': 'SeguroPriv',
        'conex_inte': 'ConexionInt',
        'ing_tri': 'Ingreso'
    }
    df_ind_orig.rename(columns=rename_dict, inplace=True)

    # Keep a df with linking variables, make variables explicitly categorical.
    df_ind = df_ind_orig[linking_cols + ['Ingreso']].copy()

    df_ind['Sexo'] = df_ind['Sexo'].astype('category')
    df_ind['Sexo'] = df_ind['Sexo'].cat.rename_categories({1: 'm', 2: 'f'})

    df_ind['Edad'] = pd.cut(
        df_ind['Edad'],
        bins = [15, 64, 200],
        labels = ['p15_64', 'p65mas'], include_lowest=True
    )

    df_ind['Nivel'] = pd.cut(
        df_ind['Nivel'],
        bins = [-1, 0, 2, 3, 100],
        labels = ['ninguno', 'primaria', 'secundaria', 'posbasica']
    )

    df_ind['SeguroIMSS'] = df_ind['SeguroIMSS'].astype('category')
    df_ind['SeguroIMSS'] = df_ind['SeguroIMSS'].cat.rename_categories({0: 'no_imss', 1: 'imss'})


    df_ind['SeguroPriv'] = df_ind['SeguroPriv'].astype('category')
    df_ind['SeguroPriv'] = df_ind['SeguroPriv'].cat.rename_categories({0: 'no_privado', 6:'privado'})

    df_ind['ConexionInt'] = df_ind['ConexionInt'].astype('category')
    df_ind['ConexionInt'] = df_ind['ConexionInt'].cat.rename_categories({1: 'internet', 2: 'no_internet'})

    if 'EstadoConyu' in linking_cols:
        df_ind['EstadoConyu'] = pd.cut(
            df_ind['EstadoConyu'],
            bins=[-1, 2, 5, 100],
            labels=['casada', 'separada', 'soltera']
        )

    # Find the bin ranges
    df_ind['Ingreso'] = pd.qcut(df_ind['Ingreso'], q, labels=list(range(1, q+1)))

    df_ind['Ingreso_orig'] = df_ind_orig.Ingreso

    return df_ind


def load_census(data_path, met_zone_codes):
    data_path = Path(data_path)

    cols = ['ENTIDAD', 'MUN', 'LOC', 'NOM_MUN', 'NOM_LOC', 'AGEB',
            'POBTOT', 'P_15YMAS', 'P_15YMAS_F', 'P_15YMAS_M',
            'POB15_64', 'POB65_MAS', 'P15YM_SE',
            'P15PRI_IN', 'P15PRI_CO', 'P15SEC_IN', 'P15SEC_CO',
            'P18YM_PB', 'PDER_IMSS', 'PAFIL_IPRIV', 'P12YM_SOLT',
            'P12YM_CASA', 'P12YM_SEPA', 'VPH_INTER', 'PROM_OCUP']

    # Split state codes and mun codes as required for census data
    s_codes = defaultdict(list)
    for c in met_zone_codes:
        s_code = c // 1000
        mun_code = c % 1000
        s_codes[s_code].append(mun_code)

    census_paths = [
        data_path / f'census/RESAGEBURB_{scode:02d}_2020_csv.zip'
        for scode in s_codes.keys()
    ]

    df_list = [
        pd.read_csv(cpath, usecols=cols, na_values=['*', 'N/D'], low_memory=False)
        for cpath in census_paths
    ]

    # Keep only aggregates by AGEB
    df_list = [
        df[df['NOM_LOC'] == 'Total AGEB urbana'].reset_index(drop=True)
        for df in df_list
    ]

    # Filter by state and met zone
    df_list = [
        df[
            (df['ENTIDAD'] == scode)
            & (df['MUN'].isin(s_codes[scode]))
        ].copy()
        for df, scode in zip(df_list, s_codes.keys())
    ]

    df_censo = pd.concat(df_list, ignore_index=True, copy=True)

    # Create CVEGEO column, and drop columns no longer useful
    df_censo['cvegeo'] = df_censo.apply(
        lambda x: f'{x.ENTIDAD:02}{x.MUN:03}{x.LOC:04}{x.AGEB.zfill(4)}',
        axis=1
    )
    df_censo.drop(
        columns=['ENTIDAD', 'MUN', 'NOM_LOC', 'LOC', 'AGEB', 'NOM_MUN'],
        inplace=True
    )

    # Remove null values and make integer, except fractional counts
    df_censo = df_censo.dropna()
    int_cols = df_censo.columns.drop(['PROM_OCUP', 'cvegeo']).copy()
    df_censo = df_censo.astype({col: int for col in int_cols})

    # Remove AGEBS with less than 20 in working population
    df_censo = df_censo[df_censo['P_15YMAS'] > 20].copy()

    # Build linking variables ###

    # Indicator variable for internet for the working population
    # Useful vars:
    #   - PROM_OCU: Promedio de ocupantes en viviendas particulares habitadas
    #   - VPH_INTER: Viviendas particulares habitadas que disponen de internet
    has_internet = df_censo['PROM_OCUP']*df_censo['VPH_INTER']
    has_internet_working = (has_internet
                            * df_censo['P_15YMAS']/df_censo['POBTOT'])
    has_internet_working = has_internet_working.astype(int)
    df_censo['ConexionInt_internet'] = has_internet_working
    df_censo['ConexionInt_no_internet'] = (
        df_censo['P_15YMAS'] - df_censo['ConexionInt_internet'])
    # Drop VPH_INTER, no longer used
    df_censo.drop(columns=['VPH_INTER', 'PROM_OCUP'], inplace=True)

    # Discretize education related variables
    #  - P15YM_SE: Población de 15 años y más sin escolaridad
    #  - P15PRI_IN: Población de 15 años y más con primaria incompleta
    #  - P15PRI_CO: Población de 15 años y más con primaria completa
    #  - P15SEC_IN: Población de 15 años y más con secundaria incompleta
    df_censo['Nivel_ninguno'] = df_censo['P15YM_SE'] + df_censo['P15PRI_IN']
    df_censo['Nivel_primaria'] = df_censo['P15PRI_CO'] + df_censo['P15SEC_IN']
    df_censo.drop(
        columns=['P15YM_SE', 'P15PRI_IN', 'P15PRI_CO', 'P15SEC_IN'],
        inplace=True)
    # Rename variables
    df_censo.rename({'P15SEC_CO': 'Nivel_secundaria',
                     'P18YM_PB': 'Nivel_posbasica'},
                    axis=1, inplace=True)
    # Make sure all education counts equal the total working population (>15)
    # In order to this, add missing counts to posbasic educaction assuming
    # they correspond to people < 18 with posbasic edu
    missing_posbasic = df_censo['P_15YMAS'] \
        - (df_censo['Nivel_primaria'] + df_censo['Nivel_secundaria']
           + df_censo['Nivel_ninguno'] + df_censo['Nivel_posbasica'])
    df_censo['Nivel_posbasica'] = (df_censo['Nivel_posbasica']
                                   + missing_posbasic)

    # Rename variables for working population,
    # which is the population we are interested in
    df_censo.rename({'P_15YMAS_F': 'Sexo_f',
                     'P_15YMAS_M': 'Sexo_m',
                     'POB15_64': 'Edad_p15_64',
                     'POB65_MAS': 'Edad_p65mas'},
                    axis=1, inplace=True)

    # Marital Status
    df_censo.rename(
        columns={'P12YM_SOLT': 'EstadoConyu_soltera',
                 'P12YM_CASA': 'EstadoConyu_casada',
                 'P12YM_SEPA': 'EstadoConyu_separada'},
        inplace=True)
    # Adjust counts assimung almost all 12-14 are single
    diff = (
        df_censo['EstadoConyu_soltera']
        + df_censo['EstadoConyu_casada']
        + df_censo['EstadoConyu_separada'] - df_censo['P_15YMAS'])
    df_censo['EstadoConyu_soltera'] = df_censo['EstadoConyu_soltera'] - diff

    # Health insurance
    df_censo['SeguroIMSS_imss'] = (
        (df_censo['PDER_IMSS']/df_censo['POBTOT'])
        * df_censo['P_15YMAS']).astype(int)
    df_censo['SeguroPriv_privado'] = (
        (df_censo['PAFIL_IPRIV']/df_censo['POBTOT'])
        * df_censo['P_15YMAS']).astype(int)
    df_censo['SeguroIMSS_no_imss'] = (
        df_censo['P_15YMAS'] - df_censo['SeguroIMSS_imss'])
    df_censo['SeguroPriv_no_privado'] = (
        df_censo['P_15YMAS'] - df_censo['SeguroPriv_privado'])
    df_censo.drop(columns=['PDER_IMSS', 'PAFIL_IPRIV'], inplace=True)

    df_censo.drop(columns='POBTOT', inplace=True)

    # Reorder cols
    cols = sorted(df_censo.columns.drop(['P_15YMAS', 'cvegeo']))
    df_censo = df_censo[['cvegeo', 'P_15YMAS'] + cols]

    # Assert total counts equal total working pop
    prefixes = [c.split('_')[0] for c in cols if '_' in c]
    prefixes = np.unique(prefixes)
    for prefix in prefixes:
        pcols = [c for c in cols if prefix in c]
        assert (df_censo[pcols].sum(axis=1) == df_censo.P_15YMAS).all()

    # Set index to cvegeo
    df_censo.set_index('cvegeo', inplace=True)

    return df_censo
