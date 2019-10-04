####################################################################################################
"""
datasets.py

This module implements several classes to perform dataset-specific downloading, saving and
data-transformation operations.

Written by Swaan Dekkers & Thomas Jongstra
"""
####################################################################################################

##################
## Manage Paths ##
##################

# Load environment variables.
MAIN_PATH = os.getenv("WOONFRAUDE_PATH")
DATA_PATH = os.getenv("WOONFRAUDE_DATA_PATH")
CODEBASE_PATH = os.abspath(os.join(MAIN_PATH, 'codebase'))

# Add system paths.
sys.path.insert(1, CODEBASE_PATH)


#############
## Imports ##
#############

from pathlib import Path
import pandas.io.sql as sqlio
import pandas as pd
import numpy as np
import requests
import psycopg2
import time
import os
import re

# Import own modules.
import config, clean


###################
## Dataset class ##
###################

class MyDataset():
    """
    Generic dataset template implementing generic functionalities.
    All other dataset classes inherit from this one.
    """

    # Define class attributed (name, id_column), which have to get a value in all subclasses.
    name = None
    table_name = None
    id_column = None


    def __init__(self):
        self._data = None
        self._version = None


    def save(self):
        """Save a previously processed version of the dataset."""
        print(f"Saving version '{self.version}' of dataframe '{self.name}'.")
        save_dataset(self.data, self.name, self.version)


    def load(self, version):
        """Load a previously processed version of the dataset."""
        try:
            self.data = load_dataset(self.name, version)
            self.version = version
            print(f"Version '{self.version}' of dataset '{self.name}' loaded!")
        except FileNotFoundError as e:
            print(f"Sorry, version '{version}' of dataset '{self.name}' is not available on local storage.")
            if version == 'download':
                print("The software will now download the dataset instead.")
                self._force_download()
            else:
                print("Please try loading another version, or creating the version you need.")


    def download(self, force=False, limit: int = 9223372036854775807):
        """Download a copy of the dataset, or restore a previous version if available."""
        if force == True:
            self._force_download()
        else:
            try:
                self.data = load(version='download')
                print("Loaded cached dataset from local storage. To force a download, \
                       use the 'download' method with the flag 'force' set to 'True'")
            except Exception:
                self._force_download()


    def _force_download(self, limit=9223372036854775807):
        """Force a dataset download."""
        self.data = download_dataset(self.name, self.table_name, limit)
        self.version = 'download'
        save_dataset(self.data, self.name, self.version)  # cache dataset locally



######################
## Helper functions ##
######################

def download_dataset(dataset_name, table_name, limit=9223372036854775807):
        """Download a new copy of the dataset from its source."""

        start = time.time()
        print(f"#### Starting download of dataset '{dataset_name}'...")

        if dataset_name == 'bbga':
            # Download BBGA file, interpret as dataframe, and return.
            url = "https://api.data.amsterdam.nl/dcatd/datasets/G5JpqNbhweXZSw/purls/LXGOPUQQfAXBbg"
            res = requests.get(url)
            df = pd.read_csv(res.content)
            return df

        # Create a server connection.
        # By default, we assume the table is in ['import_adres', 'import_wvs', 'import_stadia', 'bwv_personen', 'bag_verblijfsobject']
        conn = psycopg2.connect(host = config.HOST,
                                dbname = config.DB,
                                user = config.USER,
                                password = config.PASSWORD)
        if table_name in ['bag_nummeraanduiding', 'bag_verblijfsobject']:
            conn = psycopg2.connect(host = config.BAG_HOST,
                            dbname = config.BAG_DB,
                            user = config.BAG_USER,
                            password = config.BAG_PASSWORD)

        # Create query to download the specific table data from the server.
        # By default, we assume the table is in ['import_adres', 'import_wvs', 'import_stadia', 'bwv_personen', 'bag_verblijfsobject']
        sql = f"select * from public.{table_name} limit {limit};"
        if table_name in ['bag_nummeraanduiding']:
            sql = """
            SELECT *
            FROM public.bag_nummeraanduiding
            FULL JOIN bag_ligplaats ON bag_nummeraanduiding.ligplaats_id = bag_ligplaats.id
            FULL JOIN bag_standplaats ON bag_nummeraanduiding.standplaats_id = bag_standplaats.id
            FULL JOIN bag_verblijfsobject ON bag_nummeraanduiding.verblijfsobject_id = bag_verblijfsobject.id;
            """

        # Get data & convert to dataframe.
        df = sqlio.read_sql_query(sql, conn)

        # Close connection to server.
        conn.close()

        # Name dataframe according to table name. Beware: name will be removed by pickling.
        df.name = dataset_name

        if dataset_name == 'bag':
            df = apply_bag_colname_fix(df)

        print("\n#### ...download done! Spent %.2f seconds.\n" % (time.time()-start))
        return df


def apply_bag_colname_fix(df):
    """Fix BAG columns directly after download."""

    # Rename duplicate columns using a suffix _idx.
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols

    # Rename huisnummer, huisletter and huisnummer_toevoeging columns to specify their table origin.
    df = df.rename(index=str, columns={'huisnummer': 'huisnummer_nummeraanduiding',
                                       'huisletter': 'huisletter_nummeraanduiding',
                                       'huisnummer_toevoeging': 'huisnummer_toevoeging_nummeraanduiding'})

    # Rename columns (two copies).
    d_rename1 = {}
    l_rename1 = ['_huisnummer', '_huisletter', '_huisnummer_toevoeging']
    for r in l_rename1:
        d_rename1[r] = r + '_ligplaats'
        d_rename1[r + '_1'] = r + '_standplaats'
        d_rename1[r + '_2'] = r + '_verblijfsobject'
    df = df.rename(index=str, columns=d_rename1)

    # Rename columns (three copies).
    d_rename2 = {}
    l_rename2 = ['indicatie_geconstateerd', 'indicatie_in_onderzoek', 'geometrie',
                '_gebiedsgerichtwerken_id', '_grootstedelijkgebied_id', 'buurt_id']
    for r in l_rename2:
        d_rename2[r] = r + '_ligplaats'
        d_rename2[r + '_1'] = r + '_standplaats'
        d_rename2[r + '_2'] = r + '_verblijfsobject'
    df = df.rename(index=str, columns=d_rename2)

    # Rename columns (four copies).
    d_rename3 = {}
    l_rename3 = ['document_mutatie', 'document_nummer', 'begin_geldigheid', 'einde_geldigheid',
                'mutatie_gebruiker', 'id', 'landelijk_id', 'vervallen', 'date_modified',
                '_openbare_ruimte_naam', 'bron_id', 'status_id']
    for r in l_rename3:
        d_rename3[r] = r + '_nummeraanduiding'
        d_rename3[r + '_1'] = r + '_ligplaats'
        d_rename3[r + '_2'] = r + '_standplaats'
        d_rename3[r + '_3'] = r + '_verblijfsobject'
    df = df.rename(index=str, columns=d_rename3)

    return df


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


def save_dataset(data, dataset_name, version):
    """Save a version of the given dataframe."""
    dataset_path = os.path.join(DATA_PATH, f'{dataset_name}_{version}.h5')
    data.to_hdf(path_or_buf=dataset_path, key=dataset_name, mode='w')


def load_dataset(dataset_name, version):
    """Load a version of the dataframe from file. Rename it (pickling removes name)."""
    dataset_path = os.path.join(DATA_PATH, f'{dataset_name}_{version}.h5')
    data = pd.read_hdf(path_or_buf=dataset_path, key=dataset_name, mode='r')
    data.name = dataset_name # Set the dataframe name again after loading (it is lost when saving).
    return data
