"""
clean.py

This script implements functions to download BWV data from the wonen servers, to clean this data,
and to categorize it.

Download:
    - Download any tables in the bwv set.

Cleaning:
    - Removing entries with incorrect dates
    - Transforming column data to the correct type

Output: Cleaned and categorized/labeled BWV data
        - ~48k adresses
        - ~67k zaken
        - ~150k stadia

Written by Swaan Dekkers & Thomas Jongstra
"""


# Import statements
import pandas as pd
import time
import re

# Import own modules
import core

# Turn off pandas chained assignment warnings.
pd.options.mode.chained_assignment = None  # default='warn'


def lower_strings(df, cols=None):
    """Convert all strings in given columns to lowercase. Type remains Object (Pandas standard)."""
    if cols == None:  # By default, select all eligible columns perform string-lowering on.
        cols = df.columns
        cols = [col for col in cols if df[col].dtype == object]
    for col in cols:
        df[col] = df[col].str.lower()
    print("Lowered strings of cols %s in df %s!" % (cols, df.name))


def fix_dates(df, cols):
    """Convert columns containing dates to datetime objects)."""
    df[cols] = df[cols].apply(pd.to_datetime, errors='coerce')
    print(f"Dataframe \"%s\": Fixed dates!" % df.name)


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


def drop_duplicates(df, cols):
    """Drop duplicates in dataframe based on given column values. Print results in terminal."""
    before = df.shape[0]
    df.drop_duplicates(subset = cols)
    after = df.shape[0]
    duplicates = before - after
    print(f"Dataframe \"%s\": Dropped %s duplicates!" % (df.name, duplicates))


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


def add_binary_label_zaken(zaken, stadia):
    """Create a binary label defining whether there was woonfraude."""

    # Only set "woonfraude" label to True when the zaken_mask and/or stadia_mask is True.
    zaken['woonfraude'] = False  # Set default value to false
    zaken_mask = zaken.loc[zaken['afs_oms'].str.contains('zl woning is beschikbaar gekomen',
                                                         regex=True, flags=re.IGNORECASE) == True]
    stadia_mask = stadia.loc[stadia['sta_oms'].str.contains('rapport naar han', regex=True,
                                                            flags=re.IGNORECASE) == True]
    zaken_ids_1 = zaken_mask['zaak_id'].tolist()
    zaken_ids_2 = stadia_mask['zaak_id'].tolist()
    zaken_ids = list(set(zaken_ids_1 + zaken_ids_2))  # Get uniques
    zaken.loc[zaken['zaak_id'].isin(zaken_ids), 'woonfraude'] = True

    # Print results
    print(f"Dataframe \"zaken\": added column \"woonfraude\" (binary label)")


def impute_missing_values(df):
    """Impute missing values in each column (using column averages)."""
    # Compute averages per column (not for date columns)
    averages = dict(df.mean())
    # Impute missing values by using column averages
    df.fillna(value=averages, inplace=True)


def fix_dfs(adres, zaken, stadia):
    """Fix adres, zaken en stadia dataframes."""

    # Adres
    drop_duplicates(adres, "adres_id")
    fix_dates(adres, ['hvv_dag_tek', 'max_vestig_dtm', 'wzs_update_datumtijd'])
    lower_strings(adres)
    impute_missing_values(adres)

    # Zaken
    drop_duplicates(zaken, "zaak_id")
    fix_dates(zaken, ['begindatum','einddatum', 'wzs_update_datumtijd'])
    clean_dates(zaken)
    lower_strings(zaken)  # This needs to be done before add_column (we match lowercase strings)
    add_column(df=zaken, new_col='categorie', match_col='beh_oms',
               csv_path='E:/woonfraude/data/aanvulling_beh_oms.csv')
    impute_missing_values(zaken)

    # Stadia
    fix_dates(stadia, ['begindatum', 'peildatum', 'einddatum', 'date_created',
                      'date_modified', 'wzs_update_datumtijd'])
    clean_dates(stadia)
    stadia['zaak_id'] = stadia['adres_id'].astype(int).astype(str) + '_' + stadia['wvs_nr'].astype(int).astype(str)
    stadia['stadium_id'] = stadia['zaak_id'] + '_' + stadia['sta_nr'].astype(int).astype(str)
    drop_duplicates(stadia, "stadium_id")
    lower_strings(stadia) # This needs to be done before add_column (we match lowercase strings)
    add_column(df=stadia, new_col='label', match_col='sta_oms',
               csv_path='E:/woonfraude/data/aanvulling_sta_oms.csv')
    impute_missing_values(stadia)

