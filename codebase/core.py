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
import pickle  # vervangen door PytTables? (http://www.pytables.org)
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


def main(DOWNLOAD=False, FIX=False, ADD_LABEL=False, EXTRACT_FEATURES=False, BUILD_MODEL=False):

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
        name_dfs(adres, zaken, stadia)
        save_dfs(adres, zaken, stadia, '1')
        print("\n#### ...download done! Spent %.2f seconds.\n" % (time.time()-start))


    # Load and fix the dataframes.
    if FIX == True:
        start = time.time()
        print("\n######## Starting fix...\n")
        adres, zaken, stadia = load_dfs('1')
        clean.fix_dfs(adres, zaken, stadia)
        save_dfs(adres, zaken, stadia, '2')
        print("\n#### ...fix done! Spent %.2f seconds.\n" % (time.time()-start))


    if ADD_LABEL == True:
        start = time.time()
        print("\n######## Starting to add label...\n")
        adres, zaken, stadia = load_dfs('2')
        clean.add_binary_label_zaken(zaken, stadia)
        save_dfs(adres, zaken, stadia, '3')
        print("\n#### ...adding label done! Spent %.2f seconds.\n" % (time.time()-start))


    if EXTRACT_FEATURES == True:
        start = time.time()
        print("\n######## Starting to extract features...\n")
        adres, zaken, stadia = load_dfs('3')
        # Impute missing values
        extract_features.impute_missing_values(adres)  # Verplaatsen naar clean.py?
        extract_features.impute_missing_values(zaken)  # Verplaatsen naar clean.py?
        # Extract date features
        extract_features.extract_date_features(adres)
        extract_features.extract_date_features(zaken)
        # Extract features from columns based on word occurrence and one-hot encoding.
        extract_features.process_df_text_columns(adres, ['beh_oms'])
        extract_features.process_df_categorical_columns(adres, ['sbw_omschr', 'sbv_omschr'])

        extract_features.process_df_text_columns(zaken, ['beh_oms'])
        extract_features.process_df_categorical_columns(zaken, ['eigenaar'])

        save_dfs(adres, zaken, stadia, '4')
        print("\n#### ...extracting features done! Spent %.2f seconds.\n" % (time.time()-start))


    if BUILD_MODEL == True:
        start = time.time()
        adres, zaken, stadia = load_dfs('4')
        print("\n######## Starting to build model...\n")
        # Combine adres & zaken dataframes
        df = build_model.prepare_data(adres, zaken)
        X_train_org, X_dev, X_test, y_train_org, y_dev, y_test = build_model.split_data()
        X_train, y_train = augment_data(X_train_org, y_train_org)
        knn, precision, recall, f1, conf = run_knn(X_train, y_train, X_dev, y_dev, n_neighbors=11)
        print(f"Precisions: {precision}\nRecall: {recall}\nF1: {f1}\n")
        print(conf)
        print("\n#### ...building model done! Spent %.2f seconds.\n" % (time.time()-start))


if __name__ == "__main__":
    main()