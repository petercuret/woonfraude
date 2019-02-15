"""
extract_features.py

This script aims to take the enriched BWV data, and extract higher level features for learning.
After running this script, the resulting features should be usable for creating prediction models.

Input: enriched BWV data, i.e. coupled with up-to-date BAG data (~38k entries @ 2018-11-21).
Output: extracted BWV features for model building.

Written by Swaan Dekkers & Thomas Jongstra
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

def prepare_data(adres, zaken):
    """Combine address and cases data"""
    # Add address information to each case (this duplicates 'wzs_id' and 'wzs_update_datumtijd').
    df = zaken.merge(adres, on='adres_id', how='left')
    # Remove all columns which would not yet be available when cases (zaken) are newly opened.
    df = df.drop(columns=['einddatum', 'afg_code_beh', 'afs_code', 'afs_oms', 'afg_code_afs', 'wzs_update_datumtijd_x', 'wzs_update_datumtijd_y', 'mededelingen', 'a_dam_bag', 'landelijk_bag'])
    # Also remove columns with more than 40% nonetype data.
    # df.drop(columns=['hsltr', 'toev'], inplace=True)
    # Temporarily remove the wzs_id columns (different values in adres and zaken tables)
    df.drop(columns=['wzs_id_x', 'wzs_id_y'], inplace=True)
    return df


def scale_data(df):
    # TODO: finish writing this function
    scaler = StandardScaler
    scaler.fit(df._get_numeric_data())
    scaler.transform(df._get_numeric_data())


##################################
##### Features based on text #####
##################################

def text_series_to_features(series):
    """Convert a series of text items (possibly containing multiple words) to a list of words and an occurrence matrix."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(series)
    features = vectorizer.get_feature_names()
    matrix = X.toarray()
    return features, matrix

def extract_text_col_features(df, col):
    """Extract text features from a single column in a df. Return an occurrence dataframe encoding based on these features."""
    features, matrix = text_series_to_features(df[col])
    features = [col + '#' + x for x in features]  # Prefix each feature with name of the originating raw feature
    col_features = pd.DataFrame(data=matrix, columns=features)
    return col_features

def process_df_text_columns_hot(df, cols):
    """Create encoded feature columns for the dataframe, based on the defined text columns."""
    all_col_features = []
    for col in cols:
        col_features = extract_text_col_features(df, col)
        all_col_features.append(col_features)
    df = pd.concat([df] + all_col_features, axis=1, sort=False)
    df.drop(columns=cols, inplace=True)
    return df

def process_df_categorical_columns_hot(df, cols):
    """Create HOT encoded feature columns for the dataframe, based on the defined categorical columns."""
    all_col_features = []
    for col in cols:
        print(f"Now extracting features from column: '{col}'.")
        col_features = pd.get_dummies(df[col], prefix=col, prefix_sep='#')
        all_col_features.append(col_features)
        print("Done!")
    df = pd.concat([df] + all_col_features, axis=1, sort=False)
    df.drop(columns=cols, inplace=True)
    return df

def process_df_categorical_columns_no_hot(df, cols):
    """Create a numerically encoded feature column in the df based on each defined categorical column."""
    all_col_features = []
    for col in cols:
        print(f"Now extracting features from column: '{col}'.")
        col_features = df.col.astype('category').cat.codes
        all_col_features.append(col_features)
        print("Done!")
    df = pd.concat([df] + all_col_features, axis=1, sort=False)
    df.drop(columns=cols, inplace=True)
    return df


###########################
##### Other Features  #####
###########################

def extract_date_features(df):
    """Expand datetime values into individual features."""
    for col in df.select_dtypes(include=['datetime64[ns]']):
        print(f"Now extracting features from column: '{col}'.")
        df[col + '_year'] = pd.DatetimeIndex(df[col]).year
        df[col + '_month'] = pd.DatetimeIndex(df[col]).month
        df[col + '_day'] = pd.DatetimeIndex(df[col]).day
        df[col + '_weekday'] = pd.DatetimeIndex(df[col]).weekday
        # Get underlying Unix timestamp:
        # https://stackoverflow.com/questions/15203623/convert-pandas-datetimeindex-to-unix-time
        df[col + '_unix'] = df[col].view('int64')
        df.drop(columns=[col], inplace=True)
        print("Done!")
    return df