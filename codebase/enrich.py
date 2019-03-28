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

# Import own modules
import core

def load_bag_data():
	# Open right server connection
	conn = psycopg2.connect(database='bag', host='85.222.227.107', user='vao_reader', password='XgmrIt44vanFQGy')
	sql=f"select * from public.bag_verblijfsobject"
	bag = sqlio.read_sql_query(sql, conn)
	conn.close()
	return bag

def prepare_bag(bag):
	# To lower
	bag['_openbare_ruimte_naam'] = bag['_openbare_ruimte_naam'].str.lower()
	bag['_huisletter'] = bag['_huisletter'].str.lower()
	bag['_huisnummer_toevoeging'] = bag['_huisnummer_toevoeging'].str.lower()

	# To int
	bag['_huisnummer'] = bag['_huisnummer'].fillna(0).astype(int)
	bag['_huisnummer'] = bag['_huisnummer'].replace(0, -1)

	# Fillna and replace ''
	bag['_huisletter'] = bag['_huisletter'].fillna('None')
	bag['_huisletter'] = bag['_huisletter'].replace('', 'None')

	bag['_openbare_ruimte_naam'] = bag['_openbare_ruimte_naam'].fillna('None')
	bag['_openbare_ruimte_naam'] = bag['_openbare_ruimte_naam'].replace('', 'None')

	bag['_huisnummer_toevoeging'] = bag['_huisnummer_toevoeging'].fillna('None')
	bag['_huisnummer_toevoeging'] = bag['_huisnummer_toevoeging'].replace('', 'None') 

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

def enrich_row(x, df):
    x['a_dam_bag_real'] = df['id']
    x['landelijk_bag_real'] = df['landelijk_id']
    x['begin_geldigheid'] = df['begin_geldigheid']
    x['einde_geldigheid'] = df['einde_geldigheid']
    x['oppervlakte'] = df['oppervlakte']
    x['bouwlaag_toegang'] = df['bouwlaag_toegang']
    x['verhuurbare_eenheden'] = df['verhuurbare_eenheden']
    x['bouwlagen'] = df['bouwlagen']
    x['type_woonobject_code'] = df['type_woonobject_code']
    x['type_woonobject_omschrijving'] = df['type_woonobject_omschrijving']
    x['woningvoorraad'] = df['woningvoorraad']
    x['aantal_kamers'] = df['aantal_kamers']
    x['vervallen'] = df['vervallen']
    x['indicatie_geconstateerd'] = df['indicatie_geconstateerd']
    x['indicatie_in_onderzoek'] = df['indicatie_in_onderzoek']
    x['eigendomsverhouding_id'] = df['eigendomsverhouding_id']
    x['financieringswijze_id'] = df['financieringswijze_id']
    x['gebruik_id'] = df['gebruik_id']
    x['ligging_id'] = df['ligging_id']
    x['reden_afvoer_id'] = df['reden_afvoer_id']
    x['reden_opvoer_id'] = df['reden_opvoer_id']
    x['status_id'] = df['status_id']
    x['toegang_id'] = df['toegang_id']
    return x

def match_bag_bwv(x, bag):
    df_x = []
    df_x = bag[(bag['_openbare_ruimte_naam'] == x['sttnaam']) & (bag['_huisnummer'] == x['hsnr']) & (bag['_huisletter'] == x['hsltr']) & (bag['_huisnummer_toevoeging'] == x['toev'])]
    x['len_df_new'] = len(df_x)
    if len(df_x) == 1:
        x = enrich_row(x, df_x.iloc[0])
        
    else:
        df_y = bag[(bag['_openbare_ruimte_naam'] == x['sttnaam']) & (bag['_huisnummer'] == x['hsnr'])]

        if len(df_y) == 1:
            x['len_df_new'] = len(df_y)
            x = enrich_row(x, df_y.iloc[0])
            
    return x

def replace_string_nan_bag(bag):
	bag['_huisnummer'] = bag['_huisnummer'].replace(-1, np.nan)
	bag['_huisletter'] = bag['_huisletter'].replace('None', np.nan)
	bag['_openbare_ruimte_naam'] = bag['_openbare_ruimte_naam'].replace('None', np.nan)
	bag['_huisnummer_toevoeging'] = bag['_huisnummer_toevoeging'].replace('None', np.nan)
	return bag

def replace_string_nan_adres(adres):
	adres['hsnr'] = adres['hsnr'].replace(-1, np.nan)
	adres['sttnaam'] = adres['sttnaam'].replace('None', np.nan)
	adres['hsltr'] = adres['hsltr'].replace('None', np.nan)
	adres['toev'] = adres['toev'].replace('None', np.nan)
	return adres

def adres_bag_enrich(adres):
	print('laod')
	bag = load_bag_data()
	print('prepare')
	bag = prepare_bag(bag)
	adres = prepare_adres(adres)
	print('match')
	adres = adres.apply(lambda x: match_bag_bwv(x, bag), axis=1)
	print('replace bag')
	bag = replace_string_nan_bag(bag)
	print('replace adres')
	adres = replace_string_nan_adres(adres)
	print('done')
	adres.name = adres

	return adres

def main():
    """Add BAG data to cleaned BWV data."""

    # Load pre-cleaned adres/zaken/stadia tables.
    dfs = core.load_dfs('3')
    adres = dfs['adres']
    adres = adres.sample(100)
    # Load BAG data
    adres = adres_bag_enrich(adres)
    print('save')
    #core.save_dfs([adres], '12')
    adres.to_pickle('adres_12.pkl')
    # Save data to new pickle files.
    

if __name__ == "__main__":
    main()