"""
enrich.py

This script aims to take the cleaned BWV data, and enrich it with up-to-date BAG data.
In this process, all BWV entries that cannot be coupled with new BAG data are removed.
After running this script, the resulting data should be ready for feature extraction.

Input: cleaned BWV data (~48k entries @ 2018-11-21).
Output: enriched BWV data, i.e. coupled with up-to-date BAG data (~38k entries @ 2018-11-21)
        unenriched BWV data (~10k entries, no match found with BAG code)

Written by Swaan Dekkers & Thomas Jongstra
"""


# Import statements
from pathlib import Path
import pandas.io.sql as sqlio
import pandas as pd
import psycopg2
import numpy as np

# Import own module
import clean


def prepare_bag(bag):
    # To lower
    bag['_openbare_ruimte_naam@bag'] = bag['_openbare_ruimte_naam@bag'].str.lower()
    bag['_huisletter@bag'] = bag['_huisletter@bag'].str.lower()
    bag['_huisnummer_toevoeging@bag'] = bag['_huisnummer_toevoeging@bag'].str.lower()

    # To int
    bag['_huisnummer@bag'] = bag['_huisnummer@bag'].fillna(0).astype(int)
    bag['_huisnummer@bag'] = bag['_huisnummer@bag'].replace(0, -1)

    # Fillna and replace ''
    bag['_huisletter@bag'] = bag['_huisletter@bag'].fillna('None')
    bag['_huisletter@bag'] = bag['_huisletter@bag'].replace('', 'None')

    bag['_openbare_ruimte_naam@bag'] = bag['_openbare_ruimte_naam@bag'].fillna('None')
    bag['_openbare_ruimte_naam@bag'] = bag['_openbare_ruimte_naam@bag'].replace('', 'None')

    bag['_huisnummer_toevoeging@bag'] = bag['_huisnummer_toevoeging@bag'].fillna('None')
    bag['_huisnummer_toevoeging@bag'] = bag['_huisnummer_toevoeging@bag'].replace('', 'None')

    return bag

def prepare_adres(adres):
    # To lower
    adres['sttnaam'] = adres['sttnaam'].str.lower()
    adres['hsltr'] = adres['hsltr'].str.lower()
    adres['toev'] = adres['toev'].str.lower()

    # To int
    adres['hsnr'] = adres['hsnr'].fillna(0).astype(int)
    adres['hsnr'] = adres['hsnr'].replace(0, -1)

    # Fillna
    adres['sttnaam'] = adres['sttnaam'].fillna('None')
    adres['hsltr'] = adres['hsltr'].fillna('None')
    adres['toev'] = adres['toev'].fillna('None')
    return adres

def replace_string_nan_bag(bag):
    bag['_huisnummer@bag'] = bag['_huisnummer@bag'].replace(-1, np.nan)
    bag['_huisletter@bag'] = bag['_huisletter@bag'].replace('None', np.nan)
    bag['_openbare_ruimte_naam@bag'] = bag['_openbare_ruimte_naam@bag'].replace('None', np.nan)
    bag['_huisnummer_toevoeging@bag'] = bag['_huisnummer_toevoeging@bag'].replace('None', np.nan)
    return bag

def replace_string_nan_adres(adres):
    adres['hsnr'] = adres['hsnr'].replace(-1, np.nan)
    adres['sttnaam'] = adres['sttnaam'].replace('None', np.nan)
    adres['hsltr'] = adres['hsltr'].replace('None', np.nan)
    adres['toev'] = adres['toev'].replace('None', np.nan)
    return adres

def match_bwv_bag(adres, bag):
    # Merge dataframes on adres dataframe
    new_df = pd.merge(adres, bag,  how='left', left_on=['sttnaam','hsnr'], right_on = ['_openbare_ruimte_naam@bag', '_huisnummer@bag'])

    # Find id's that have a direct match and that have multiple matches
    g = new_df.groupby('adres_id')
    df_direct = g.filter(lambda x: len(x) == 1)
    df_multiple = g.filter(lambda x: len(x) > 1)

    # Make multiplematch more specific to construct perfect match
    df_multiple = df_multiple[(df_multiple['hsltr'] == df_multiple['_huisletter@bag']) & (df_multiple['toev'] == df_multiple['_huisnummer_toevoeging@bag'])]

    # Concat df_direct and df_multiple
    df_result = pd.concat([df_direct, df_multiple])

    # Because of the seperation of an object, there are two matching objects. Keep the oldest object with definif point
    df_result = df_result.sort_values(['adres_id', 'status_coordinaat_code@bag'])
    df_result = df_result.drop_duplicates(subset='adres_id', keep='first')

    # Add adresses without match
    final_df = pd.merge(adres, df_result,  how='left', on='adres_id', suffixes=('', '_y'))
    final_df.drop(list(final_df.filter(regex='_y$')), axis=1, inplace=True)

    # Drop unwanted bag columns
    bag_remove = ['einde_geldigheid@bag',               # Only 2 entries in column.
                  'verhuurbare_eenheden@bag',           # Only ~2k entries in column.
                  'geometrie@bag',                      # Needs a lot of processing before being useful.
                  'bron_id@bag',                        # Only 2 entries in column.
                  'locatie_ingang_id@bag',              # Only 2 entries in column.
                  'reden_afvoer_id@bag',                # Only a few entries in column.
                  '_gebiedsgerichtwerken_id@bag',       # Superfluous (gebied).
                  '_grootstedelijkgebied_id@bag',       # Superfluous (grootstedelijkgebied).
                  'buurt_id@bag',                       # Superfluous (buurt).
                  '_openbare_ruimte_naam@bag',          # Superfluous (straatnaam).
                  '_huisnummer@bag',                    # Superfluous (huisnummer).
                  '_huisletter@bag',                    # Superfluous (huisletter).
                  '_huisnummer_toevoeging@bag',         # Superfluous (huisnummer toevoeging).
                  'vervallen@bag',                      # Superfluous (all values in col are equal).
                  'mutatie_gebruiker@bag',               # Superfluous (all values in col are equal).
                  'document_mutatie@bag',               # Not available at time of signal.
                  'date_modified@bag',                  # Not available at time of signal.
                  'document_nummer@bag',                # Not needed? (Swaan?)
                  'status_coordinaat_omschrijving@bag', # Not needed? (Swaan?)
                  'type_woonobject_code@bag',           # Not needed? (Swaan?)
                  'id@bag',                             # Not needed.
                  'landelijk_id@bag'                    # Not needed.
                  ]
    final_df.drop(columns=bag_remove, inplace=True)

    return final_df


def impute_values_for_bagless_addresses(adres):
    """Impute values for adresses where no BAG-match could be found."""
    clean.impute_missing_values(adres)
    clean.impute_missing_values_mode(adres, ['status_coordinaat_code@bag', 'indicatie_geconstateerd@bag', 'indicatie_in_onderzoek@bag', 'woningvoorraad@bag'])
    adres.fillna(value={'type_woonobject_omschrijving': 'None',
                      'eigendomsverhouding_id@bag': 'None',
                      'financieringswijze_id@bag': -1,
                      'gebruik_id@bag': -1,
                      'reden_opvoer_id@bag': -1,
                      'status_id@bag': -1,
                      'toegang_id@bag': 'None'}, inplace=True)


def adres_bag_enrich(adres, bag):
    """Enrich the adres data with information from the BAG data."""
    bag = prepare_bag(bag)
    adres = prepare_adres(adres)
    adres = match_bwv_bag(adres, bag)
    bag = replace_string_nan_bag(bag)
    adres = replace_string_nan_adres(adres)
    impute_values_for_bagless_addresses(adres)
    adres.name = 'adres'
    return adres
