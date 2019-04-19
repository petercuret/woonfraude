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
from pathlib import Path
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


def impute_missing_values(df):
    """Impute missing values in each column (using column averages)."""

    # Compute averages per column (only for numeric columns, so not for dates or strings)
    averages = dict(df._get_numeric_data().mean())

    # Also compute averages for datetime columns
    for col in df.select_dtypes(include=['datetime64[ns]']):
        # Get underlying Unix timestamps for all non-null values.
        unix = df[col][df[col].notnull()].view('int64')
        # Compute the average unix timestamp
        mean_unix = unix.mean()
        # Convert back to datetime
        mean_datetime = pd.to_datetime(mean_unix)
        # Put value in averages column
        averages[col] = mean_datetime

    # Impute missing values by using the column averages.
    df.fillna(value=averages, inplace=True)


def impute_missing_values_mode(df, cols):
    """Impute the mode value (most frequent) in empty values. Usable for fixing bool columns."""

    # Make a dict to save the column modes.
    modes = {}

    # Loop over all given columns.
    for col in cols:
        mode = df[col].mode()[0]  # Compute the column mode.
        modes[col] = mode  # Add to dictionary.

    # Impute missing values by using the columns modes.
    df.fillna(value=modes, inplace=True)


def select_closed_cases(adres, zaken, stadia):
    """Only select cases (zaken) that have 100% certainly been closed."""

    # Select closed zoeklicht cases.
    zaken['mask'] = zaken.afs_oms == 'zl woning is beschikbaar gekomen'
    zaken['mask'] += zaken.afs_oms == 'zl geen woonfraude'
    zl_zaken = zaken.loc[zaken['mask']]


    # Indicate which stadia are indicative of closed cases.
    stadia['mask'] = stadia.sta_oms == 'rapport naar han'
    stadia['mask'] += stadia.sta_oms == 'bd naar han'

    # Indicate which stadia are from before 2013. Cases linked to these stadia should be
    # disregarded. Before 2013, 'rapport naar han' and 'bd naar han' were used inconsistently.
    timestamp_2013 =  pd.Timestamp('2013-01-01')
    stadia['before_2013'] = stadia.begindatum < timestamp_2013

    # Create groups linking cases to their stadia.
    zaak_groups = stadia.groupby('zaak_id').groups

    # Select all closed cases based on "rapport naar han" and "bd naar han" stadia.
    keep_ids = []
    for zaak_id, stadia_ids in zaak_groups.items():
        zaak_stadia = stadia.loc[stadia_ids]
        if sum(zaak_stadia['mask']) >= 1 and sum(zaak_stadia['before_2013']) == 0:
            keep_ids.append(zaak_id)
    rap_zaken = zaken[zaken.zaak_id.isin(keep_ids)]

    # Combine all selected cases.
    selected_zaken = pd.concat([zl_zaken, rap_zaken], sort=True)

    # Print results.
    print(f'Selected {len(selected_zaken)} closed cases from a total of {len(zaken)} cases.')

    # Only return the relevant selection of cases.
    return selected_zaken


def filter_categories(zaken):
    """
    Remove cases (zaken) with categories 'woningkwaliteit' or 'afdeling vergunninen beheer'.

    These cases do not contain reliable samples.
    """
    filtered_zaken = zaken[~zaken.categorie.isin(['woningkwaliteit', 'afdeling vergunningen en beheer'])]
    return filtered_zaken


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


def fix_dfs(adres, zaken, stadia, personen, bag):
    """Fix adres, zaken en stadia dataframes."""

    # Get path to home directory
    home = str(Path.home())

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
               csv_path=f'{home}/Documents/woonfraude/data/aanvulling_beh_oms.csv')
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
               csv_path=f'{home}/Documents/woonfraude/data/aanvulling_sta_oms.csv')
    impute_missing_values(stadia)

    # Personen
    personen.drop_duplicates(subset='id', inplace=True)  # Remove duplicate persons.

    # BAG
    fix_dates(bag, ['begin_geldigheid@bag', 'date_modified@bag'])
    impute_missing_values(bag)
    impute_missing_values_mode(bag, ['status_coordinaat_code@bag', 'indicatie_geconstateerd@bag', 'indicatie_in_onderzoek@bag', 'woningvoorraad@bag'])
    bag.fillna(value={'type_woonobject_omschrijving': 'None',
                      'eigendomsverhouding_id@bag': 'None',
                      'financieringswijze_id@bag': -1,
                      'gebruik_id@bag': -1,
                      'reden_opvoer_id@bag': -1,
                      'status_id@bag': -1,
                      'toegang_id@bag': 'None'}, inplace=True)