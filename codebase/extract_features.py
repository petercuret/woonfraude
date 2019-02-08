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

# Import own module
import core

##################################
##### Features based on text #####
##################################

def text_series_to_features(series):
    """Convert a series of text items (possibly containing multiple words) to a list of words and an occurrence matrix."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(series)
    features = vectorizer.get_feature_names()
    matrix = X.toarray()
    return words, matrix

def extract_text_col_features(df, col):
    """Extract text features from a single column in a df. Return an occurrence dataframe encoding based on these features."""
    features, matrix = text_series_to_features(df[col])
    features = [col + '#' + x for x in features]  # Prefix each feature with name of the originating raw feature
    col_features = pd.DataFrame(data=matrix, columns=features)
    return col_features

def process_df_text_columns(df, cols):
    """Create encoded feature columns for the dataframe, based on the defined text columns."""
    for col in cols:
        col_features = extract_text_col_features(df, col)
        df = pd.concat([df, col_features], axis=1, sort=False)
        df.drop(columns=[col], inplace=True)
    return df

def process_df_categorical_columns(df, cols):
    """Create HOT encoded feature columns for the dataframe, based on the defined categorical columns."""
    for col in cols:
        col_features = pd.get_dummies(df[col], prefix=col, prefix_sep='#')
        df = pd.concat([df, col_features], axis=1, sort=False)
        df.drop(columns=[col], inplace=True)
    return df


###########################
##### Other Features  #####
###########################

def impute_missing_values(df):
    """Impute missing values in each column (using column averages)."""
    # Compute averages per column (not for date columns)
    averages = dict(df.mean())

    # Impute missing values by using column averages
    df.fillna(value=averages, inplace=True)

    return df


def extract_date_features(df):
    """Expand datetime values into individual features."""
    for col in df.select_dtypes(include=['datetime64[ns]']):
        df[col + '_year'] = pd.DatetimeIndex(df[col]).year
        df[col + '_month'] = pd.DatetimeIndex(df[col]).month
        df[col + '_day'] = pd.DatetimeIndex(df[col]).day
        df[col + '_weekday'] = pd.DatetimeIndex(df[col]).weekday
        # Get underlying Unix timestamp: https://stackoverflow.com/questions/15203623/convert-pandas-datetimeindex-to-unix-time
        df[col + '_unix'] = pd.DatetimeIndex(df[col]).astype(np.int64) // 10**9
        df.drop(columns=[col], inplace=True)

    for col in df.select_dtypes(include=['float64']):
        pass

    return df