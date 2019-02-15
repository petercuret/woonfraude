"""
core.py

This script implements several high-level functions, as well as data collection (download/store)
and data saving functions.

Written by Swaan Dekkers & Thomas Jongstra
"""


# Import statements
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


def save_dfs(dfs, version, path="E:\\woonfraude\\data\\"):
    """Save a version of the given dataframes to dir. Use the df names as their keys."""
    dfs[0].to_hdf(f"{path}data_{version}.h5", key=dfs[0].name, mode='w')
    if len(dfs) > 0:
        for df in dfs[1:]:
            df.to_hdf(f"{path}data_{version}.h5", key=df.name, mode='a')
    print("Dataframes saved as version \"%s\"." % version)


def load_dfs(version, path="E:\\woonfraude\\data\\"):
    """Load a version of the dataframes from dir. Rename them (pickling removes name)."""
    # Get keys of dfs in data file.
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
        adres = download_data('adres')
        zaken = download_data('zaken')
        stadia = download_data('stadia')
        # adres_periodes = download_data("adres_periodes", limit=100)
        # hotline_melding = download_data("hotline_melding", limit=100)
        # hotline_bevinding = download_data("hotline_bevinding", limit=100)
        # personen = download_data("personen", limit=100)
        # personen_huwelijk = download_data("personen_huwelijk", limit=100)
        # Name and save the dataframes.
        adres.name = 'adres'
        zaken.name = 'zaken'
        stadia.name = 'stadia'
        save_dfs([adres, zaken, stadia], '1')
        print("\n#### ...download done! Spent %.2f seconds.\n" % (time.time()-start))


    # Load and fix the dataframes.
    if FIX == True:
        start = time.time()
        print("\n######## Starting fix...\n")
        dfs = load_dfs('1')
        adres = dfs['adres']
        zaken = dfs['zaken']
        stadia = dfs['stadia']
        clean.fix_dfs(adres, zaken, stadia)
        save_dfs([adres, zaken, stadia], '2')
        print("\n#### ...fix done! Spent %.2f seconds.\n" % (time.time()-start))


    if ADD_LABEL == True:
        start = time.time()
        print("\n######## Starting to add label...\n")
        dfs = load_dfs('2')
        adres = dfs['adres']
        zaken = dfs['zaken']
        stadia = dfs['stadia']
        clean.add_binary_label_zaken(zaken, stadia)
        save_dfs([adres, zaken, stadia], '3')
        print("\n#### ...adding label done! Spent %.2f seconds.\n" % (time.time()-start))


    if EXTRACT_FEATURES == True:
        start = time.time()
        print("\n######## Starting to extract features...\n")
        dfs = load_dfs('3')
        adres = dfs['adres']
        zaken = dfs['zaken']
        stadia = dfs['stadia']
        # Combine adres and zaken dfs. Remove columns which are not available when cases are opened.
        df = extract_features.prepare_data(adres, zaken)
        # Extract date features.
        df = extract_features.extract_date_features(df)
        # Extract features from columns based on word occurrence and one-hot encoding.
        adres_cat_use =  ['postcode', 'sdl_code', 'brt_code', 'pvh_cd', 'pvh_omschr', 'sbw_omschr',
                          'sbv_omschr', 'wzs_buurtcode_os_2015', 'wzs_buurtcombinatiecode_os_2015',
                          'wzs_rayoncode_os_2015', 'wzs_stadsdeelcode_os_2015', 'sttnaam', 'hsltr',
                          'toev', 'brt_naam', 'wzs_wijze_verrijking_geo', 'wzs_22gebiedencode_2015']
        adres_cat_remove = ['wzs_buurtnaam_os_2015', 'wzs_buurtcombinatienaam_os_2015',
                            'wzs_rayonnaam_os_2015', 'wzs_stadsdeelnaam_os_2015',
                            'wzs_alternatieve_buurtennaam_os_2015',
                            'wzs_alternatieve_buurtencode_os_2015', 'wzs_geom', 'brtcombi_code',
                            'brtcombi_naam', 'sdl_naam', 'wzs_22gebiedennaam_2015']
        zaken_cat_use = ['beh_code', 'eigenaar', 'categorie']
        zaken_cat_remove = ['beh_oms']
        df = extract_features.process_df_categorical_columns_hot(df, adres_cat_use + zaken_cat_use)
        # Remove superfluous columns (e.g. columns with textual descriptions of codes)
        df.drop(columns=adres_cat_remove + zaken_cat_remove, inplace=True)
        df.name = 'df'
        save_dfs([df, stadia], '4')
        print("\n#### ...extracting features done! Spent %.2f seconds.\n" % (time.time()-start))


    if SPLIT_DATA == True:
        start = time.time()

        print('Loading data...')
        dfs = load_dfs('4')
        df = dfs['df']
        stadia = dfs['stadia']
        print('Done!')

        print('Splitting data...')
        X_train_org, X_dev, X_test, y_train_org, y_dev, y_test = build_model.split_data(df)
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
        # y_test = dfs['y_test']
        del dfs
        print('Done!')

        subset_size = 0.05
        X_train = X_train_org.head(int(len(X_train_org)*subset_size))
        del X_train_org
        y_train = y_train_org.head(int(len(y_train_org)*subset_size))
        del y_train_org
        # X_train, y_train = build_model.augment_data(X_train, y_train)
        # X_dev = X_dev.head(int(len(X_dev)*subset_size))
        # y_dev = y_dev.head(int(len(y_dev)*subset_size))
        # y_test = y_test.head(int(len(y_test)*subset_size))

        print('Training model...')
        model, precision, recall, f1, conf, report = build_model.run_linear_svc(X_train, y_train, X_dev, y_dev)
        print('Training done!')

        print(f"Precisions: {precision}\nRecall: {recall}\nF1: {f1}\n")
        print(conf)
        print("\n#### ...building model done! Spent %.2f seconds.\n" % (time.time()-start))


if __name__ == "__main__":
    main()