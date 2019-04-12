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
	final_df = final_df.drop(['document_mutatie@bag', 'document_nummer@bag', 'status_coordinaat_omschrijving@bag',
        'type_woonobject_code@bag', 'geometrie@bag', '_gebiedsgerichtwerken_id@bag', '_grootstedelijkgebied_id@bag',
       'bron_id@bag', 'buurt_id@bag'], axis=1)

	return final_df


def adres_bag_enrich(adres, bag):
	"""Enrich the adres data with information from the BAG data."""
	bag = prepare_bag(bag)
	adres = prepare_adres(adres)
	adres = match_bwv_bag(adres, bag)
	bag = replace_string_nan_bag(bag)
	adres = replace_string_nan_adres(adres)
	adres.name = 'adres'
	return adres
