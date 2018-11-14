import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import requests
import timeit
import pickle
import time
import q

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

from IPython.core.interactiveshell import InteractiveShell
from tqdm import tqdm
tqdm.pandas()


def download_data(table, limit=18446744073709551615):
    """
    Download data from wonen server, from specific table.

    Table options: "adres", "zaken", "stadia", "adres_periodes", "hotline_melding",
                   "hotline_bevinding", "personen", "personen_huwelijk", e.a..

    """

    # Open right server connection.
    if table in ['adres', 'zaken', 'stadia']:
        conn = psycopg2.connect("EXAMPLE_LOGIN")
    else:
        conn = psycopg2.connect("EXAMPLE_LOGIN")

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


def drop_duplicates(df, cols):
    """Drop duplicates in dataframe based on given column values. Print results in terminal."""
    before = df.shape[0]
    df.drop_duplicates(subset = cols)
    after = df.shape[0]
    duplicates = before - after
    print(f"Dataframe \"%s\": Dropped %s duplicates!" % (df.name, duplicates))


def fix_integers(df, cols):
    """Correctly set datatype to integer for several columns."""
    df[cols] = df[cols].apply(pd.to_numeric, downcast='integer',
                                                  errors='coerce')


def fix_dates(df, cols):
    """Convert columns containing dates to datetime objects)."""
    df[cols] = df[cols].apply(pd.to_datetime, errors='coerce')


# TODO: Change function so it tries to do this for all columns (if containing text)?
def lower_strings(df, cols):
    """Convert all strings in given columns to lowercase."""
    df[cols] = df[cols].str.lower()


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


# TODO: fuzzy string matching (match_col <-> key) implementeren?
def add_column(df, col, match_col, csv, key='lcolumn', val='ncolumn'):
    """Add a new column to dataframe based on the match_column, and the mapping in the csv.

    df: dataframe to be augmented.
    col: name of new dataframe column.
    match_col: colum to match with the csv variable 'key'.
    csv: path to the csv file which is used for augmentation.
    key: name of column in csv file containing keys.
    val: name of column in csv file containing values.
    """

    # Load csv file.
    df_label = pd.read_csv(csv)

    # Transform csv string data to lowercase.
    df_label[key] = df_label[key].str.lower()
    df_label[val] = df_label[val].str.lower()

    # Create a dict mapping: key -> val, based on the csv data.
    label_dict = dict(zip(df_label[key], df_label[val]))

    # Create a new dataframe column. If match_col matches with 'key', set the value to 'val'.
    df[col] = df[match_col].apply(lambda x: label_dict.get(x))

    # Print information about performed operation to terminal.
    print(f"Dataframe \"%s\": added column \"%s\"!" % (df.name, col))


# TODO: fuzzy match implementeren?
# TODO: code sneller maken?
def add_binary_label_zaken(zaken, stadia):
    """Create a binary label which defines whether there was woonfraude."""

    # Create a translation function from sta_oms and afs_oms to binary woonfraude label.
    # def woonfraude_binary_2(x):
    #     r = stadia[stadia['zaak_id'] == x['zaak_id']]
    #     r = r.sort_values('sta_nr')
    #     r_list = r['sta_oms'].tolist()
    #     if 'rapport naar han' in r_list:
    #         return True
    #     elif x['afs_oms'] == 'ZL Woning is beschikbaar gekomen':
    #         return True
    #     else:
    #         return False

    # Create a translation function from sta_oms and afs_oms to binary woonfraude label.
    # TODO: weggooien
    def woonfraude_binary(x):
        r = stadia[stadia['zaak_id'] == x['zaak_id']]
        r = r.sort_values('sta_nr')
        if 'rapport naar han' in r['sta_oms'].tolist():
            return True
        elif x['afs_oms'] == 'ZL Woning is beschikbaar gekomen':
            return True
        else:
            return False

    # Apply translation to zaken dataframe. Add output to new column "woonfraude".
    def a():
        zaken['woonfraude'] = zaken.apply(lambda x: woonfraude_binary(x), axis=1)

    # def b():
    #     zaken['woonfraude'] = zaken.apply(lambda x: woonfraude_binary_2(x), axis=1)

    def c():
        zaken['woonfraude'] = False  # Set default value to false
        zaken_mask = zaken.loc[zaken['afs_oms'] == 'ZL Woning is beschikbaar gekomen']
        stadia_mask = stadia.loc[stadia['sta_oms'] == 'rapport naar han']
        zaken_ids_1 = zaken_mask['zaak_id'].tolist()
        zaken_ids_2 = stadia_mask['zaak_id'].tolist()
        zaken_ids = list(set(zaken_ids_1 + zaken_ids_2))  # Get uniques
        zaken['woonfraude'] = zaken.apply(lambda x: True if x['zaak_id'] in zaken_ids else False, axis=1)
        # q.d()

    print(timeit.timeit(a, number=30))
    # print(timeit.timeit(b, number=10))
    print(timeit.timeit(c, number=30))

    # Print results
    print(f"Dataframe \"zaken\": added column \"woonfraude\" (binary label)")


# Module maken en in notebook importeren
# Optimaliseren binary label toekenning
# Toevoegen basisregistratie info

def main():

    # Downloads & saves
    # adres = download_data('adres')
    # zaken = download_data('zaken')
    # stadia = download_data('stadia')
    # save_dfs(adres, zaken, stadia, '1')

    '''
    # Load previously downloaded snapshot of data.
    adres, zaken, stadia = load_dfs('1')

    # Name the dataframes
    name_dfs(adres, zaken, stadia)

    # Adres
    fix_integers(adres, ['adres_id', 'straatcode', 'sbw_code', 'xref', 'yref',
                         'sbv_code', 'inwnrs', 'kmrs'])
    fix_dates(adres, ['hvv_dag_tek', 'max_vestig_dtm', 'wzs_update_datumtijd'])
    drop_duplicates(adres, "adres_id")

    # Zaken
    fix_integers(zaken, ['adres_id', 'wvs_nr', 'kamer_aantal', 'nuttig_woonoppervlak',
                         'vloeroppervlak_totaal', 'bedrag_huur'])
    fix_dates(zaken, ['begindatum','einddatum', 'wzs_update_datumtijd'])
    clean_dates(zaken)
    lower_strings(zaken, "beh_oms")
    drop_duplicates(zaken, "zaak_id")
    add_column(zaken, 'categorie', 'beh_oms', 'E:/woonfraude/data/aanvulling_beh_oms.csv')

    # Stadia
    fix_integers(stadia, ['adres_id', 'wvs_nr', 'sta_nr'])
    fix_dates(stadia, ['begindatum', 'peildatum', 'einddatum',
                       'date_created', 'date_modified', 'wzs_update_datumtijd'])
    clean_dates(stadia)
    lower_strings(stadia, "sta_oms")

    # Add zaak_id and stadium_id
    stadia['zaak_id'] = stadia['adres_id'].map(str) + '_' + stadia['wvs_nr'].map(str)
    stadia['stadium_id'] = stadia['zaak_id'] + '_' + stadia['sta_nr'].map(str)
    drop_duplicates(stadia, "stadium_id")
    add_column(stadia, 'label', 'sta_oms', 'E:/woonfraude/data/aanvulling_sta_oms.csv')

    # Make snapshot of data
    save_dfs(adres, zaken, stadia, '2')
    '''

    # Optionele downloads
    adres_periodes = download_data("adres_periodes", 100)
    hotline_melding = download_data("hotline_melding", 100)
    hotline_bevinding = download_data("hotline_bevinding", 100)
    personen = download_data("personen", 100)
    personen_huwelijk = download_data("personen_huwelijk", 100)

    q.d()

    adres, zaken, stadia = load_dfs('2')

    # add_binary_label_zaken(zaken, stadia)

    # Create small set for optimization of binary label code
    zaken_small = zaken.loc[:99, :]
    stadia_small = stadia.loc[:99, :]
    add_binary_label_zaken(zaken_small, stadia_small)

    q.d()


if __name__ == "__main__":
    main()