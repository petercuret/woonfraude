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


def main(DOWNLOAD=False, FIX=False, ADD_LABEL=False):

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

    adres, zaken, stadia = load_dfs('3')


if __name__ == "__main__":
    main()