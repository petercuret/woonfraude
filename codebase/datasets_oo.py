"""
datasets_oo.py

This code implements dataset-specific downloading, saving and data-transformation operations.

Written by Swaan Dekkers & Thomas Jongstra
"""

# Import statements
from pathlib import Path
import pandas as pd
import time
# from torch.utils.data import Dataset

# Import own modules
import config


class MyDataset(pd.Dataframe):
    """Dataset containing address data."""

    # Define class attributed (name, id_column), which have to get a value in all subclasses.
    name = None
    id_column = None

    def __init__(self):
        self.data = None
        self.version = None

    @classmethod
    def save(version):
        """Save a previously processed version of the dataset."""
        save_dataset(self.data, self.name, self.version)


    @classmethod
    def load(version):
        """Load a previously processed version of the dataset."""
        try:
            self.data = load_dataset(self.name, version)
            self.version = version
        except Exception:
            if version == 'download':
                _force_download()
            else:
                print(f"Sorry, version {version} of dataset {self.name} is not available on local storage.")


    @classmethod
    def download(force=False, limit: int = 9223372036854775807):
        """Download a copy of the dataset, or restore a previous version if available."""
        if force == True:
            self.forced_download()
        else:
            try:
                self.data = load(version='download')
                print("Loaded cached dataset from local storage. To force a download, \
                       use the 'download' method with the flag 'force' set to 'True'")
            except Exception:
                self._force_download()


    @classmethod
    def _force_download():
        """Download a new copy of the dataset from its source."""
        start = time.time()
        print("Starting data download...")
        self.data = download_dataset(self.name, limit)
        self.version = 'download'
        print("\n#### ...download done! Spent %.2f seconds.\n" % (time.time()-start))


class AdresDataset(MyDataset):
    """Create a dataset for the adres data."""

    # Set the class attributes.
    name = 'adres'
    id_column = 'adres_id'

    @classmethod
    def extract_leegstand():
        """Create a column indicating leegstand (no inhabitants on the address)."""
        self.data['leegstand'] = (self.data.inwnrs == 0)


class ZakenDataset(MyDataset):
    """Create a dataset for the zaken data."""

    ## Set the class attributes.
    name = 'zaken'
    id_column = 'zaak_id'


class StadiaDataset(MyDataset):
    """Create a dataset for the stadia data."""

    # Set the class attributes.
    name = 'stadia'
    id_column = 'stadium_id'


class BagDataset(MyDataset):
    """Create a dataset for the bag data."""

    # Set the class attributes.
    name = 'bag'
    id_column = 'id'


def download_dataset(table, limit=9223372036854775807):

    # Create a server connection.
    if table in ['import_adres', 'import_wvs', 'import_stadia', 'bwv_personen']:
        conn = psycopg2.connect(host = config.HOST,
                                dbname = config.DB,
                                user = config.USER,
                                password = config.PASSWORD)
    if table in ['bag_verblijfsobject', 'bag']:
        conn = psycopg2.connect(host = config.BAG_HOST,
                        dbname = config.BAG_DB,
                        user = config.BAG_USER,
                        password = config.BAG_PASSWORD)

    # Create query to download the specific table data from the server.
    if table in ['import_adres', 'import_wvs', 'import_stadia', 'bwv_personen', 'bag_verblijfsobject']:
        sql = f"select * from public.{table} limit {limit};"
    if table in ['bag']:
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

    # Name dataframe according to table name. Won't be saved after pickling.
    df.name = table

    # Return dataframe.
    return df


# Define HOME and DATA_PATH on a global level
HOME = str(Path.home())
DATA_PATH = f'{home}/Documents/woonfraude/data/'


def save_dataset(data, dataset_name, version):
    """Save a version of the given dataframe."""
    data.to_hdf(path_or_buf=f"{DATA_PATH}{dataset_name}_{version}.h5", key=dataset_name, mode='w')


def load_dataset(dataset_name, version):
    """Load a version of the dataframe from file. Rename it (pickling removes name)."""
    data = pd.read_hdf(path_or_buf=f"{DATA_PATH}{dataset_name}_{version}.h5", key=dataset_name, mode='r')
    return data