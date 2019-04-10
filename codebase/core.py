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
import clean
import config  # Load local passwords (config.py file expected in same folder as this file).
import extract_features
import build_model

def download_data(table, limit=9223372036854775807):
    """
    Download data from wonen server, from specific table.

    Table options: "import_adres", "import_wvs", "import_stadia", "bwv_adres_periodes",
    "bwv_hotline_melding", "bwv_personen", "bwv_personen_huwelijk", e.a..

    """

    # Open right server connection.
    conn = psycopg2.connect(host = config.HOST,
                            dbname = config.DB,
                            user = config.USER,
                            password = config.PASSWORD)

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



def main(DOWNLOAD=False, FIX=False, ADD_LABEL=False, EXTRACT_FEATURES=False, SPLIT_DATA=False, BUILD_MODEL=False):

    # Downloads & saves tables to dataframes.
    if DOWNLOAD == True:
        start = time.time()
        print("\n######## Starting download...\n")
        adres = download_data('import_adres')
        zaken = download_data('import_wvs')
        stadia = download_data('import_stadia')
        personen = download_data("bwv_personen")
        # hotline_melding = download_data("bwv_hotline_melding", limit=100)
        # personen_huwelijk = download_data("bwv_personen_huwelijk", limit=100)
        # Name and save the dataframes.
        adres.name = 'adres'
        zaken.name = 'zaken'
        stadia.name = 'stadia'
        personen.name = 'personen'
        save_dfs([adres, zaken, stadia, personen], '1')
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
        del dfs
        clean.fix_dfs(adres, zaken, stadia, personen)
        zaken = clean.select_closed_cases(adres, zaken, stadia)
        zaken = clean.filter_categories(zaken)
        zaken.name = 'zaken'
        save_dfs([adres, zaken, stadia, personen], '2')
        print("\n#### ...fix done! Spent %.2f seconds.\n" % (time.time()-start))


    if ADD_LABEL == True:
        start = time.time()
        print("\n######## Starting to add label...\n")
        dfs = load_dfs('2')
        adres = dfs['adres']
        zaken = dfs['zaken']
        stadia = dfs['stadia']
        personen = dfs['personen']
        del dfs
        clean.add_binary_label_zaken(zaken, stadia)
        save_dfs([adres, zaken, stadia, personen], '3')
        print("\n#### ...adding label done! Spent %.2f seconds.\n" % (time.time()-start))


    if EXTRACT_FEATURES == True:
        start = time.time()
        print("\n######## Starting to extract features...\n")
        dfs = load_dfs('3')
        adres = dfs['adres']
        zaken = dfs['zaken']
        stadia = dfs['stadia']
        personen = dfs['personen']
        del dfs

        # Remove superfluous columns (e.g. columns with textual descriptions of codes etc.).
        adres_cat_remove = [# Remove because cols do not exists when melding is received
                            'wzs_update_datumtijd',
                            # Remove because cols do not add extra information.
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
                            'afg_code_beh',
                            'afs_code',
                            'afs_oms',
                            'afg_code_afs',
                            'wzs_update_datumtijd',
                            # Remove because cols do not add extra information.
                            'beh_oms',
                            'wzs_id']
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
        adres_cat_use =  ['postcode',
                          'pvh_omschr',
                          'sbw_omschr',
                          'sbv_omschr',
                          'wzs_buurtcode_os_2015',
                          'wzs_buurtcombinatiecode_os_2015',
                          'wzs_stadsdeelcode_os_2015',
                          'sttnaam',
                          'toev']
        zaken_cat_use = ['beh_code',
                         'eigenaar',
                         'categorie']
        df = extract_features.process_df_categorical_columns_hot(df, adres_cat_use + zaken_cat_use)

        # Rescale features.
        df = extract_features.scale_data(df, ['inwnrs', 'kmrs'])

        # Remove adres_id, since this is not a feature we want our algorihtm to try and learn from.
        df.drop(columns='adres_id', inplace=True)

        # Name and save resulting dataframe.
        df.name = 'df'
        save_dfs([df, stadia], '4')
        print("\n#### ...extracting features done! Spent %.2f seconds.\n" % (time.time()-start))


    if SPLIT_DATA == True:
        start = time.time()

        print('Loading data...')
        dfs = load_dfs('4')
        df = dfs['df']
        stadia = dfs['stadia']
        del dfs
        print('Done!')

        print('Splitting data...')
        X_train_org, X_dev, X_test, y_train_org, y_dev, y_test = build_model.split_data_train_dev_test(df)
        X_train_org.name = 'X_train_org'
        X_dev.name = 'X_dev'
        X_test.name = 'X_test'
        y_train_org.name = 'y_train_org'
        y_dev.name = 'y_dev'
        y_test.name = 'y_test'
        save_dfs([X_train_org, X_dev, X_test, y_train_org, y_dev, y_test, stadia], '5')
        print('Done!')

        print("\n#### ...splitting data done! Spent %.2f seconds.\n" % (time.time()-start))


    if BUILD_MODEL == True:
        start = time.time()

        print('Loading data...')
        dfs = load_dfs('5')
        X_train_org = dfs['X_train_org']
        X_dev = dfs['X_dev']
        X_test = dfs['X_test']
        y_train_org = dfs['y_train_org']
        y_dev = dfs['y_dev']
        y_test = dfs['y_test']
        del dfs
        print('Done!')

        # subset_size = 0.05
        # X_train = X_train_org.head(int(len(X_train_org)*subset_size))
        # del X_train_org
        # y_train = y_train_org.head(int(len(y_train_org)*subset_size))
        # del y_train_org
        # X_train, y_train = build_model.augment_data(X_train, y_train)
        # X_dev = X_dev.head(int(len(X_dev)*subset_size))
        # y_dev = y_dev.head(int(len(y_dev)*subset_size))
        # y_test = y_test.head(int(len(y_test)*subset_size))

        # Select all positive samples
        X_train_positives = X_train_org[y_train_org == True]
        y_train_positives = y_train_org[y_train_org == True]

        # Select all negative samples
        X_train_negatives = X_train_org[y_train_org == False]
        y_train_negatives = y_train_org[y_train_org == False]

        # Splits negative data into several sets
        import numpy as np
        print('Splitting up negative samples.')
        n_splits = 8
        X_train_negatives_sets = np.split(X_train_negatives, n_splits)
        y_train_negatives_sets = np.split(y_train_negatives, n_splits)
        del X_train_negatives, y_train_negatives

        # Combine each subset of the negative samples with all positive samples.
        # Train a model on each of these new combined training sets.
        for i in range(n_splits):
            print(f"Training model on split {i}.")
            X_train = pd.concat([X_train_negatives_sets[i], X_train_positives])
            y_train = pd.concat([y_train_negatives_sets[i], y_train_positives])
            model, precision, recall, f1, conf, report = build_model.run_random_forest(X_train, y_train, X_dev, y_dev)

        print("\n#### ...training models done! Spent %.2f seconds.\n" % (time.time()-start))


if __name__ == "__main__":
    main()