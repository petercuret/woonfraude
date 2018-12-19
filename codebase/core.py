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

# Import own modules
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