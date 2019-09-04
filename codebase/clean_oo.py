"""
clean_oo.py

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
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import pandas as pd
import time
import re

# Import own modules
import core


class CleanTransformer(BaseEstimator, TransformerMixin):
    """Class for performing cleaning steps on dataframes within the sklearn pipeline."""

    def __init__(self,
                 id_column = None,
                 drop_duplicates: bool = True,
                 drop_columns: list = [],  # Contains list of columns to drop.
                 fix_date_columns: list = [],  # Contains list of date columns to fix.
                 clean_dates: bool = False,
                 lower_string_columns = True,  # Contains list of columns to lower strings in. If True, all string columns are lowered.
                 impute_missing_values: bool = True,  # Impute missing values in all numeric and timestamp columns using averages.
                 impute_missing_values_mode: list = [],  # Impute missing values for a list of specific columns using the mode.
                 impute_missing_values_custom: dict = {},  # impute missing values of each column defined in the key, with the value corresponding to the key.
                 fillna_columns: dict = {},  # Contains the following key-value pairs: key=column_name, value=value_to_be_imputed.
                ):
        self.id_column = id_column
        self.drop_duplicates = drop_duplicates
        self.drop_columns = drop_columns
        self.fix_date_columns = fix_date_columns
        self.clean_dates = clean_dates
        self.lower_string_columns = lower_string_columns
        self.impute_missing_values = impute_missing_values
        self.impute_missing_values_mode = impute_missing_values_mode
        self.impute_missing_values_custom = impute_missing_values_custom

        self.fillna_columns = fillna_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.drop_duplicates and self.id_column:
            drop_duplicates(X, self.id_column)
        if len(self.drop_columns) > 0:
            X.drop(columns=self.drop_columns, inplace=True)
        if len(self.fix_date_columns) > 0:
            fix_dates(X, self.fix_date_columns)
        if self.clean_dates:
            clean_dates(X)
        if self.lower_string_columns:
            lower_strings(X)
        if self.impute_missing_values:
            impute_missing_values(X)
        if len(self.impute_missing_values_mode) > 0:
            impute_missing_values_mode(X, self.impute_missing_values_mode)
        if len(self.impute_missing_values_custom) > 0:
            impute_missing_values_custom(X, self.impute_missing_values_custom)
        if len(self.fillna_columns) > 0:
            X.fillna(value=self.fillna_columns)
        return X

def drop_duplicates(df, cols):
    """Drop duplicates in dataframe based on given column values. Print results in terminal."""
    before = df.shape[0]
    df.drop_duplicates(subset = cols)
    after = df.shape[0]
    duplicates = before - after
    print(f"Dataframe \"%s\": Dropped %s duplicates!" % (df.name, duplicates))


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


def lower_strings(df, cols=True):
    """Convert all strings in given columns to lowercase. Type remains Object (Pandas standard)."""
    if cols == True:  # By default, select all eligible columns perform string-lowering on.
        cols = df.columns
        cols = [col for col in cols if df[col].dtype == object]
    for col in cols:
        df[col] = df[col].str.lower()
    print("Lowered strings of cols %s in df %s!" % (cols, df.name))


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
    print("Missing values in df %s have been imputed!" % (df.name))


# DEZE SCHRIJVEN MET STANDARD IMPUTER UIT SKLEARN! #
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
    print("Missing values (using mode) of cols %s in df %s have been imputed!" % (cols, df.name))


def impute_missing_values_custom(df, col_dict):
    """Impute the missing values of each column defined in the dict keys, with the corresponding dict value."""
    df.fillna(value=col_dict, inplace=True)
    print("Missing values (using custom strategy) of cols %s in df %s have been imputed!" % (str(list(col_dict.keys())), df.name))
