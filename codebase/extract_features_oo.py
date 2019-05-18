"""
extract_features_oo.py

This script aims to take the enriched BWV data, and extract higher level features for learning.
After running this script, the resulting features should be usable for creating prediction models.

Input: enriched BWV data, i.e. coupled with up-to-date BAG data (~38k entries @ 2018-11-21).
Output: extracted BWV features for model building.

Written by Swaan Dekkers & Thomas Jongstra
"""

# Imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math


###################
# AANDACHTSPUNTEN #
###################
#
# Waar kunnen de functies "extract_leegstand", "add_person_features" en "scale_data" het beste landen?
#
###################
###################


class FeatureExtractionTransformer(BaseEstimator, TransformerMixin):
    """Class for performing feature extraction steps on dataframes within the sklearn pipeline."""

    def __init__(self,
                 text_features_cols_hot: list = [],  # List should contain names of text columns to extract features from.
                 categorical_cols_hot: list = [],  # List should contain names of categorical columnsto extract features from, using HOT encoding.
                 categorical_cols_no_hot: list = [], # List should contain names of categorical columns to extract features from, not using HOT encoding.
                 extract_date_features: bool = False,  # Boolean indicating whether features should be extracted from all date columns.
                ):
        self.text_features_cols_hot = text_features_cols_hot
        self.categorical_cols_hot = categorical_cols_hot
        self.categorical_cols_no_hot = categorical_cols_no_hot
        self.extract_date_features = extract_date_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.text_features_cols_hot:
            X = extract_hot_text_features_cols(X, self.text_features_cols_hot)
        if self.categorical_cols_hot:
            X = extract_hot_categorical_cols(X, self.categorical_cols_hot)
        if self.categorical_cols_no_hot:
            X = extract_no_hot_categorical_cols(X, self.categorical_cols_no_hot)
        if self.extract_date_features:
            X = extract_date_features(X)


##################################
##### Features based on text #####
##################################

# def text_series_to_features(series):
#     """Convert a series of text items (possibly containing multiple words) to a list of words and an occurrence matrix."""
#     vectorizer = CountVectorizer()
#     X = vectorizer.fit_transform(series)
#     features = vectorizer.get_feature_names()
#     matrix = X.toarray()
#     return features, matrix

def extract_text_features_col(df, col):
    """Extract text features from a single column in a df. Return an occurrence dataframe encoding based on these features."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[col])
    features = vectorizer.get_feature_names()
    matrix = X.toarray()
    features = [col + '#' + x for x in features]  # Prefix each feature with name of the originating raw feature
    col_features = pd.DataFrame(data=matrix, columns=features)
    return col_features

def extract_hot_text_features_cols(df, cols):
    """Create encoded feature columns for the dataframe, based on the defined text columns."""
    all_col_features = []
    for col in cols:
        col_features = extract_text_features_col(df[col])
        all_col_features.append(col_features)
    df = pd.concat([df] + all_col_features, axis=1, sort=False)
    df.drop(columns=cols, inplace=True)
    return df

def extract_hot_categorical_cols(df, cols):
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

def extract_no_hot_categorical_cols(df, cols):
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
        # df[col + '_year'] = pd.DatetimeIndex(df[col]).year
        df[col + '_month'] = pd.DatetimeIndex(df[col]).month
        df[col + '_day'] = pd.DatetimeIndex(df[col]).day
        df[col + '_weekday'] = pd.DatetimeIndex(df[col]).weekday
        # Get underlying Unix timestamp:
        # https://stackoverflow.com/questions/15203623/convert-pandas-datetimeindex-to-unix-time
        # df[col + '_unix'] = df[col].view('int64')
        df.drop(columns=[col], inplace=True)
        print("Done!")
    return df


def extract_leegstand(df):
    """Create a column indicating leegstand (no inhabitants on the address)."""
    df['leegstand'] = (df.inwnrs == 1)
    return df


###########################
##### Feature Scaling #####
###########################

def scale_data(df, cols):
    """Scale data using the sklearn StandardScaler for the defined columns."""

    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df