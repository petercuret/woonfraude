"""
datasets_oo.py

This code implements dataset-specific downloading, saving and data-transformation operations.

Written by Swaan Dekkers & Thomas Jongstra
"""

# Import statements
from pathlib import Path
import pandas.io.sql as sqlio
import pandas as pd
import requests
import psycopg2
import time
# from torch.utils.data import Dataset

# Import own modules
import config


class MyDataset():
    """Dataset containing address data."""

    # Define class attributed (name, id_column), which have to get a value in all subclasses.
    name = None
    table_name = None
    id_column = None


    def __init__(self):
        self._data = None
        self._version = None


    def save(self):
        """Save a previously processed version of the dataset."""
        print(self.name, self.version)
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
    return df


class AdresDataset(MyDataset):
    """Create a dataset for the adres data."""

    # Set the class attributes.
    name = 'adres'
    table_name = 'import_adres'
    id_column = 'adres_id'


    def extract_leegstand(self):
        """Create a column indicating leegstand (no inhabitants on the address)."""
        self.data['leegstand'] = ~self.data.inwnrs.notnull()


    def enrich_woning_id(self):
        """Add woning ids to the adres dataframe."""
        adres_periodes = download_dataset('bwv_adres_periodes', 'bwv_adres_periodes')
        self.data = self.data.merge(adres_periodes, how='left', left_on='adres_id', right_on='ads_id')


    def add_hotline_features(self, hotline_df):
        """Add the hotline features to the adres dataframe."""
        # Create a temporary merged df using the adres and hotline dataframes.
        merge = self.data.merge(hotline_df, on='wng_id', how='left')
        # Create a group for each adres_id
        adres_groups = merge.groupby(by='adres_id')
        # Count the number of hotline meldingen per group/adres_id
        hotline_counts = adres_groups['id'].agg(['count'])
        # Rename column
        hotline_counts.columns = ['aantal_hotline_meldingen']
        # Enrich the 'adres' dataframe with the computed hotline counts.
        self.data = self.data.merge(hotline_counts, on='adres_id', how='left')


class ZakenDataset(MyDataset):
    """Create a dataset for the zaken data."""

    ## Set the class attributes.
    name = 'zaken'
    table_name = 'import_wvs'
    id_column = 'zaak_id'


class StadiaDataset(MyDataset):
    """Create a dataset for the stadia data."""

    # Set the class attributes.
    name = 'stadia'
    table_name = 'import_stadia'
    id_column = 'stadium_id'


class PersonenDataset(MyDataset):
    """Create a dataset for the stadia data."""

    # Set the class attributes.
    name = 'personen'
    table_name = 'bwv_personen'
    id_column = 'id'


class BagDataset(MyDataset):
    """Create a dataset for the bag data."""

    # Set the class attributes.
    name = 'bag'
    table_name = 'bag_nummeraanduiding'
    id_column = 'id'

    def bag_fix(self):
        """Apply specific fixes for the BAG dataset."""

        df = self.data

        # Merge columns.
        l_merge = ['_gebiedsgerichtwerken_id', 'indicatie_geconstateerd', 'indicatie_in_onderzoek', '_grootstedelijkgebied_id', 'buurt_id']
        for m in l_merge:
            df[m] = df[m].combine_first(df[m + '_2'])
            df[m] = df[m].combine_first(df[m + '_1'])
            df.drop(columns=[m + '_2', m + '_1'], inplace=True)

        # Rename columns.
        d_rename = {}
        l_rename = ['openbareruimte_naam', 'id', 'landelijk_id', 'status_id']
        for r in l_rename:
            d_rename[r] = r + '_nummeraanduiding'
            d_rename[r + '_1'] = r + '_ligplaats'
            d_rename[r + '_2'] = r + '_standplaats'
            d_rename[r + '_3'] = r + '_verblijfsobject'
        df = df.rename(index=str, columns=d_rename)

        # Change dataset version, and save this version of the dataset.
        # self.set_version('columnFix')
        self.version = 'columnFix'
        self.save()


class HotlineDataset(MyDataset):
    """Create a dataset for the hotline data."""

    # Set the class attributes.
    name = 'hotline'
    table_name = 'bwv_hotline_melding'
    id_column = 'id'


class BbgaDataset(MyDataset):
    """Create a dataset for the BBGA data."""

    # Set the class attributes.
    name = 'bbga'


# Define HOME and DATA_PATH on a global level
HOME = str(Path.home())
DATA_PATH = f'{HOME}/Documents/woonfraude/data/'


def save_dataset(data, dataset_name, version):
    """Save a version of the given dataframe."""
    data.to_hdf(path_or_buf=f"{DATA_PATH}{dataset_name}_{version}.h5", key=dataset_name, mode='w')


def load_dataset(dataset_name, version):
    """Load a version of the dataframe from file. Rename it (pickling removes name)."""
    data = pd.read_hdf(path_or_buf=f"{DATA_PATH}{dataset_name}_{version}.h5", key=dataset_name, mode='r')
    return data
