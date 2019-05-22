"""
core.py

This script implements several high-level functions, as well as data collection (download/store)
and data saving functions.

Written by Swaan Dekkers & Thomas Jongstra
"""


# Import statements
from pathlib import Path
import pandas.io.sql as sqlio
import pandas as pd
import psycopg2
import pickle
import time

# Import own modules
import config  # Load local passwords (config.py file expected in same folder as this file).
import clean
import enrich
import extract_features
import build_model

def download_data(table, limit=9223372036854775807):
    """
    Download data from wonen server, from specific table.

    Table options: "import_adres", "import_wvs", "import_stadia", "bwv_adres_periodes",
    "bwv_hotline_melding", "bwv_personen", "bwv_personen_huwelijk", e.a..

    """

    # Create a server connection.
    if table in ['import_adres', 'import_wvs', 'import_stadia', 'bwv_personen']:
        conn = psycopg2.connect(host = config.HOST,
                                dbname = config.DB,
                                user = config.USER,
                                password = config.PASSWORD)

    if table in ['bag_verblijfsobject']:
        conn = psycopg2.connect(host = config.BAG_HOST,
                        dbname = config.BAG_DB,
                        user = config.BAG_USER,
                        password = config.BAG_PASSWORD)

    # Create query to download specific table data from server.
    sql = f"select * from public.{table} limit {limit};"

    # Get data & convert to dataframe.
    df = sqlio.read_sql_query(sql, conn)

    # Close connection to server.
    conn.close()

    # Name dataframe according to table name. Won't be saved after pickling.
    df.name = table

    # Return dataframe.
    return df


def download_bag():
    """Download BAG data from multiple linked tables on the wonen server."""

    # Create query
sql = """
SELECT *
FROM public.bag_nummeraanduiding
FULL JOIN bag_ligplaats ON bag_nummeraanduiding.ligplaats_id = bag_ligplaats.id
FULL JOIN bag_standplaats ON bag_nummeraanduiding.standplaats_id = bag_standplaats.id
FULL JOIN bag_verblijfsobject ON bag_nummeraanduiding.verblijfsobject_id = bag_verblijfsobject.id;
"""

    # Create a server connection.
    conn = psycopg2.connect(host = config.BAG_HOST,
                            dbname = config.BAG_DB,
                            user = config.BAG_USER,
                            password = config.BAG_PASSWORD)

    # Get data & convert to dataframe.
    df = sqlio.read_sql_query(sql, conn)

    # Close connection to server.
    conn.close()

    # Name dataframe according to table name. Won't be saved after pickling.
    df.name = 'bag'

    # Return dataframe.
    return df


def save_dfs(dfs, version):
    """Save a version of the given dataframes to dir. Use the df names as their keys."""
    home = str(Path.home())
    path = f'{home}/Documents/woonfraude/data/'
    dfs[0].to_hdf(f"{path}data_{version}.h5", key=dfs[0].name, mode='w')
    if len(dfs) > 0:
        for df in dfs[1:]:
            df.to_hdf(f"{path}data_{version}.h5", key=df.name, mode='a')
    print("Dataframes saved as version \"%s\"." % version)


def load_dfs(version):
    """Load a version of the dataframes from dir. Rename them (pickling removes name)."""
    # Get keys of dfs in data file.
    home = str(Path.home())
    path = f'{home}/Documents/woonfraude/data/'
    keys = []
    with pd.HDFStore(f"{path}data_{version}.h5") as hdf:
        keys = hdf.keys()
    # Load dfs from data file.
    dfs = {}
    for key in keys:
        key = key[1:]  # Remove leading forward slash from key name
        dfs[key] = pd.read_hdf(f"{path}data_{version}.h5", key)
        dfs[key].name = key
    return dfs



def main(DOWNLOAD=False, FIX=False, ENRICH=False, ADD_LABEL=False, EXTRACT_FEATURES=False, SPLIT_DATA=False, BUILD_MODEL=False):

    # Downloads & saves tables to dataframes.
    if DOWNLOAD == True:
        start = time.time()
        print("\n######## Starting download...\n")
        adres = download_data('import_adres')
        zaken = download_data('import_wvs')
        stadia = download_data('import_stadia')
        personen = download_data("bwv_personen")
        bag = download_bag()
        bag = bag.add_suffix('@bag')
        # hotline_melding = download_data("bwv_hotline_melding", limit=100)
        # personen_huwelijk = download_data("bwv_personen_huwelijk", limit=100)
        # Name and save the dataframes.
        adres.name = 'adres'
        zaken.name = 'zaken'
        stadia.name = 'stadia'
        personen.name = 'personen'
        bag.name = 'bag'
        save_dfs([adres, zaken, stadia, personen, bag], '1')
        print("\n#### ...download done! Spent %.2f seconds.\n" % (time.time()-start))


    # Load and fix the dataframes.
    if FIX == True:
        start = time.time()
        print("\n######## Starting fix...\n")
        dfs = load_dfs('1')
        adres = dfs['adres']
        zaken = dfs['zaken']
        stadia = dfs['stadia']
        personen = dfs['personen']
        bag = dfs['bag']
        del dfs
        clean.fix_dfs(adres, zaken, stadia, personen, bag)
        zaken = clean.select_closed_cases(adres, zaken, stadia)
        zaken = clean.filter_categories(zaken)
        zaken.name = 'zaken'
        save_dfs([adres, zaken, stadia, personen, bag], '2')
        print("\n#### ...fix done! Spent %.2f seconds.\n" % (time.time()-start))

    if ENRICH == True:
        start = time.time()
        print("\n######## Starting BAG enrichment...\n")
        dfs = load_dfs('2')
        adres = dfs['adres']
        zaken = dfs['zaken']
        stadia = dfs['stadia']
        personen = dfs['personen']
        bag = dfs['bag']
        del dfs
        adres = enrich.adres_bag_enrich(adres, bag)
        save_dfs([adres, zaken, stadia, personen], '3')
        print("\n#### ...BAG enrichment done! Spent %.2f seconds.\n" % (time.time()-start))

    if ADD_LABEL == True:
        start = time.time()
        print("\n######## Starting to add label...\n")
        dfs = load_dfs('3')
        adres = dfs['adres']
        zaken = dfs['zaken']
        stadia = dfs['stadia']
        personen = dfs['personen']
        del dfs
        clean.add_binary_label_zaken(zaken, stadia)
        save_dfs([adres, zaken, stadia, personen], '4')
        print("\n#### ...adding label done! Spent %.2f seconds.\n" % (time.time()-start))


    if EXTRACT_FEATURES == True:
        start = time.time()
        print("\n######## Starting to extract features...\n")
        dfs = load_dfs('4')
        adres = dfs['adres']
        zaken = dfs['zaken']
        stadia = dfs['stadia']
        personen = dfs['personen']
        del dfs

        # Remove superfluous columns (e.g. columns with textual descriptions of codes etc.).
        adres_cat_remove = [# Remove because cols do not exists when melding is received
                            'wzs_update_datumtijd',
                            # Remove because cols do not add extra information.
                            'kmrs',
                            'straatcode',
                            'xref',
                            'yref',
                            'postcode',
                            'wzs_buurtcode_os_2015',
                            'wzs_buurtcombinatiecode_os_2015',
                            'wzs_stadsdeelcode_os_2015',
                            'sttnaam',
                            'hvv_dag_tek', # Empty column
                            'max_vestig_dtm', # Empty column
                            'wzs_22gebiedencode_os_2015', # Empty column
                            'wzs_22gebiedennaam_os_2015', # Empty column
                            'sdl_naam',
                            'pvh_cd',
                            'sbv_code',
                            'sbw_code',
                            'wzs_wijze_verrijking_geo',
                            'wzs_22gebiedencode_2015',
                            'brt_naam',
                            'wzs_buurtnaam_os_2015',
                            'wzs_buurtcombinatienaam_os_2015',
                            'wzs_rayonnaam_os_2015',
                            'wzs_rayoncode_os_2015',
                            'wzs_stadsdeelnaam_os_2015',
                            'wzs_alternatieve_buurtennaam_os_2015',
                            'wzs_alternatieve_buurtencode_os_2015',
                            'hsltr',
                            'wzs_geom',
                            'brt_code',
                            'brtcombi_code',
                            'brtcombi_naam',
                            'sdl_code',
                            'wzs_22gebiedennaam_2015',
                            'wzs_id',
                            'a_dam_bag',
                            'landelijk_bag']
        zaken_cat_remove = [# Remove because cols do not exists when melding is received
                            'einddatum',
                            'mededelingen',
                            'afs_code',
                            'afs_oms',
                            'afg_code_afs',
                            'wzs_update_datumtijd',
                            # Remove because cols do not add extra information.
                            'kamer_aantal',
                            'wvs_nr',
                            'beh_oms',
                            'wzs_id',
                            # Remove cols because they contains an ID
                            'zaak_id']
        adres.drop(columns=adres_cat_remove, inplace=True)
        zaken.drop(columns=zaken_cat_remove, inplace=True)

        # Combine adres and zaken dfs.
        df = zaken.merge(adres, on='adres_id', how='left')

        # Extract leegstand feature.
        df = extract_features.extract_leegstand(df)

        # Add person features.
        df = extract_features.add_person_features(df, personen)

        # Extract date features.
        df = extract_features.extract_date_features(df)

        # Extract features from columns based on word occurrence and one-hot encoding.
        adres_cat_use =  ['toev',
                          'pvh_omschr',
                          'sbw_omschr',
                          'sbv_omschr']
        zaken_cat_use = ['afg_code_beh',
                         'beh_code',
                         'eigenaar',
                         'categorie']
        bag_cat_use = ['status_coordinaat_code@bag',
                       'type_woonobject_omschrijving@bag',
                       'eigendomsverhouding_id@bag',
                       'financieringswijze_id@bag',
                       'gebruik_id@bag',
                       'ligging_id@bag',
                       'reden_opvoer_id@bag',
                       'status_id@bag',
                       'toegang_id@bag',
                       ]
        df = extract_features.process_df_categorical_columns_hot(df, adres_cat_use + zaken_cat_use + bag_cat_use)

        # Rescale features.
        df = extract_features.scale_data(df, ['inwnrs'])

        # Remove adres_id, since this is not a feature we want our algorihtm to try and learn from.
        df.drop(columns='adres_id', inplace=True)

        # Name and save resulting dataframe.
        df.name = 'df'
        save_dfs([df], '5')
        print("\n#### ...extracting features done! Spent %.2f seconds.\n" % (time.time()-start))


    if SPLIT_DATA == True:
        start = time.time()

        print('Loading data...')
        dfs = load_dfs('5')
        df = dfs['df']
        del dfs
        print('Done!')

        print('Splitting data...')
        X_train, X_test, y_train, y_test = build_model.split_data_train_test(df)
        X_train.name = 'X_train'
        X_test.name = 'X_test'
        y_train.name = 'y_train'
        y_test.name = 'y_test'
        save_dfs([X_train, X_test, y_train, y_test], '6')
        print('Done!')

        print("\n#### ...splitting data done! Spent %.2f seconds.\n" % (time.time()-start))


    if BUILD_MODEL == True:
        start = time.time()

        print('Loading data...')
        dfs = load_dfs('6')
        X_train = dfs['X_train']
        X_test = dfs['X_test']
        y_train = dfs['y_train']
        y_test = dfs['y_test']
        del dfs
        print('Done!')

        # TO BE FINISHED!


if __name__ == "__main__":
    main()