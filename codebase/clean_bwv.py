"""
clean_bwv.py

This script implements functions to download BWV data from the wonen servers, to clean this data,
and to categorize it.

Download:
    - Download any tables in the bwv set.

Cleaning:
    - Removing entries with incorrect dates
    - Transforming column data to the correct type

Output: Cleaned and categorized/labeled BWV data
        - ~48k adresses
        - ~67k zaken
        - ~150k stadia

Written by Swaan Dekkers & Thomas Jongstra
"""

# Import statements
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import matplotlib.pyplot as plt
import numpy as np
import warnings
import requests
import random
import pickle  # vervangen door PytTables? (http://www.pytables.org)
import time
import re
import q
from pathlib import Path

# Load local passwords, if config file exists.
# if Path("config.py").is_file():
import config

# Turn off pandas chained assignment warnings.
pd.options.mode.chained_assignment = None  # default='warn'


def download_data(table, limit=9223372036854775807):
    """
    Download data from wonen server, from specific table.

    Table options: "adres", "zaken", "stadia", "adres_periodes", "hotline_melding",
                   "hotline_bevinding", "personen", "personen_huwelijk", e.a..

    """

    # Open right server connection.
    if table in ['adres', 'zaken', 'stadia']:
        conn = psycopg2.connect(config.server_1)
    else:
        conn = psycopg2.connect(config.server_2)

    # Create query to download specific table data from server.
    sql = f"select * from public.bwv_%s limit %s;" % (table, limit)

    # Get data & convert to dataframe.
    df = sqlio.read_sql_query(sql, conn)

    # Close connection to server.
    conn.close()

    # Name dataframe according to table name. Won't be saved after pickling.
    df.name = table

    # Return dataframe.
    return df


def save_dfs(adres, zaken, stadia, version, path="E:\\woonfraude\\data\\"):
    """Save a version of the given dataframes to dir."""
    adres.to_pickle("%sadres_%s.p" % (path, version))
    zaken.to_pickle("%szaken_%s.p" % (path, version))
    stadia.to_pickle("%sstadia_%s.p" % (path, version))
    print("Dataframes saved as version \"%s\"." % version)


def load_dfs(version, path="E:\\woonfraude\\data\\"):
    """Load a version of the dataframes from dir. Rename them (pickling removes name)."""
    adres = pd.read_pickle("%sadres_%s.p" % (path, version))
    zaken = pd.read_pickle("%szaken_%s.p" % (path, version))
    stadia = pd.read_pickle("%sstadia_%s.p" % (path, version))
    name_dfs(adres, zaken, stadia)
    return adres, zaken, stadia


def name_dfs(adres, zaken, stadia):
    """Name dataframes."""
    adres.name = 'adres'
    zaken.name = 'zaken'
    stadia.name = 'stadia'


def lower_strings(df, cols=None):
    """Convert all strings in given columns to lowercase. Type remains Object (Pandas standard)."""
    if cols == None:  # By default, select all eligible columns perform string-lowering on.
        cols = df.columns
        cols = [col for col in cols if df[col].dtype == object]
    for col in cols:
        df[col] = df[col].str.lower()
    print("Lowered strings of cols %s in df %s!" % (cols, df.name))


def fix_dates(df, cols):
    """Convert columns containing dates to datetime objects)."""
    df[cols] = df[cols].apply(pd.to_datetime, errors='coerce')
    print(f"Dataframe \"%s\": Fixed dates!" % df.name)


def clean_dates(df):
    """Filter out incorrect dates."""
    before = df.shape[0]
    today = pd.to_datetime('today')
    year = 2010
    l1 = df[df['begindatum'].dt.year <= year].index.tolist()
    l2 = df[df['begindatum'] > today].index.tolist()
    l3 = df[df['einddatum'].dt.year <= year].index.tolist()
    l4 = df[df['einddatum'] > today].index.tolist()
    l5 = df[df['begindatum'] > df['einddatum']].index.tolist() # begin > eind, 363 in stadia en 13 in zaken
    ltot = list(set().union(l1,l2,l3,l4,l5))
    df.drop(ltot, inplace=True)

    # Print info about number of removed rows.
    after = df.shape[0]
    removed = before - after
    print(f"Dataframe \"%s\": Cleaned out %s dates!" % (df.name, removed))


def drop_duplicates(df, cols):
    """Drop duplicates in dataframe based on given column values. Print results in terminal."""
    before = df.shape[0]
    df.drop_duplicates(subset = cols)
    after = df.shape[0]
    duplicates = before - after
    print(f"Dataframe \"%s\": Dropped %s duplicates!" % (df.name, duplicates))


def add_column(df, new_col, match_col, csv_path, key='lcolumn', val='ncolumn'):
    """Add a new column to dataframe based on the match_column, and the mapping in the csv.

    df: dataframe to be augmented.
    new_col: name of new dataframe column.
    match_col: colum to match with the csv variable 'key'.
    csv_path: path to the csv file which is used for augmentation.
    key: name of column in csv file containing keys.
    val: name of column in csv file containing values.
    """

    # Load csv file.
    df_label = pd.read_csv(csv_path)

    # Transform csv string data to lowercase.
    df_label[key] = df_label[key].str.lower()
    df_label[val] = df_label[val].str.lower()

    # Create a dict mapping: key -> val, based on the csv data.
    label_dict = dict(zip(df_label[key], df_label[val]))

    # Create a new dataframe column. If match_col matches with 'key', set the value to 'val'.
    df[new_col] = df[match_col].apply(lambda x: label_dict.get(x))

    # Print information about performed operation to terminal.
    print(f"Dataframe \"%s\": added column \"%s\"!" % (df.name, new_col))


def add_binary_label_zaken(zaken, stadia):
    """Create a binary label defining whether there was woonfraude."""

    # Only set "woonfraude" label to True when the zaken_mask and/or stadia_mask is True.
    zaken['woonfraude'] = False  # Set default value to false
    zaken_mask = zaken.loc[zaken['afs_oms'].str.contains('zl woning is beschikbaar gekomen',
                                                         regex=True, flags=re.IGNORECASE) == True]
    stadia_mask = stadia.loc[stadia['sta_oms'].str.contains('rapport naar han', regex=True,
                                                            flags=re.IGNORECASE) == True]
    zaken_ids_1 = zaken_mask['zaak_id'].tolist()
    zaken_ids_2 = stadia_mask['zaak_id'].tolist()
    zaken_ids = list(set(zaken_ids_1 + zaken_ids_2))  # Get uniques
    zaken.loc[zaken['zaak_id'].isin(zaken_ids), 'woonfraude'] = True

    # Print results
    print(f"Dataframe \"zaken\": added column \"woonfraude\" (binary label)")


def fix_dfs(adres, zaken, stadia):
    """Fix adres, zaken en stadia dataframes."""

    # Adres
    lower_strings(adres)
    fix_dates(adres, ['hvv_dag_tek', 'max_vestig_dtm', 'wzs_update_datumtijd'])
    drop_duplicates(adres, "adres_id")

    # Zaken
    lower_strings(zaken)
    drop_duplicates(zaken, "zaak_id")
    fix_dates(zaken, ['begindatum','einddatum', 'wzs_update_datumtijd'])
    clean_dates(zaken)
    add_column(df=zaken, new_col='categorie', match_col='beh_oms',
               csv_path='E:/woonfraude/data/aanvulling_beh_oms.csv')

    # Stadia
    lower_strings(stadia)
    fix_dates(stadia, ['begindatum', 'peildatum', 'einddatum', 'date_created',
                      'date_modified', 'wzs_update_datumtijd'])
    clean_dates(stadia)
    stadia['zaak_id'] = stadia['adres_id'].astype(int).astype(str) + '_' + stadia['wvs_nr'].astype(int).astype(str)
    stadia['stadium_id'] = stadia['zaak_id'] + '_' + stadia['sta_nr'].astype(int).astype(str)
    drop_duplicates(stadia, "stadium_id")
    add_column(df=stadia, new_col='label', match_col='sta_oms',
               csv_path='E:/woonfraude/data/aanvulling_sta_oms.csv')


def main(DOWNLOAD=False, FIX=False, ADD_LABEL=False):

    # Downloads & saves tables to dataframes.
    if DOWNLOAD == True:
        start = time.time()
        print("\n######## Starting download...\n")
        # adres = download_data('adres')
        # zaken = download_data('zaken')
        # stadia = download_data('stadia')
        adres_periodes = download_data("adres_periodes", limit=100)
        hotline_melding = download_data("hotline_melding", limit=100)
        hotline_bevinding = download_data("hotline_bevinding", limit=100)
        personen = download_data("personen", limit=100)
        personen_huwelijk = download_data("personen_huwelijk", limit=100)
        q.d()
        # Name and save the dataframes.
        # name_dfs(adres, zaken, stadia)
        # save_dfs(adres, zaken, stadia, '1')
        print("\n#### ...download done! Spent %.2f seconds.\n" % (time.time()-start))


    # Load and fix the dataframes.
    if FIX == True:
        start = time.time()
        print("\n######## Starting fix...\n")
        adres, zaken, stadia = load_dfs('1')
        fix_dfs(adres, zaken, stadia)
        save_dfs(adres, zaken, stadia, '2')
        print("\n#### ...fix done! Spent %.2f seconds.\n" % (time.time()-start))


    if ADD_LABEL == True:
        start = time.time()
        print("\n######## Starting to add label...\n")
        adres, zaken, stadia = load_dfs('2')
        add_binary_label_zaken(zaken, stadia)
        save_dfs(adres, zaken, stadia, '3')
        print("\n#### ...adding label done! Spent %.2f seconds.\n" % (time.time()-start))

    adres, zaken, stadia = load_dfs('3')


if __name__ == "__main__":
    main(DOWNLOAD=True)