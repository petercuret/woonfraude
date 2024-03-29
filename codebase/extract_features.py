####################################################################################################
"""
extract_features.py

This module implements a transformer class to extract higher level features from raw data,
which can be used when training models.

Written by Swaan Dekkers & Thomas Jongstra

"""
####################################################################################################

#############
## Imports ##
#############

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import math


#####################################
## Features Extraction Transformer ##
#####################################

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
        if self.text_features_cols_hot != []:
            X = extract_text_features_cols_hot(X, self.text_features_cols_hot)
        if self.categorical_cols_hot != []:
            X = extract_categorical_cols_hot(X, self.categorical_cols_hot)
        if self.categorical_cols_no_hot != []:
            X = extract_categorical_cols_no_hot(X, self.categorical_cols_no_hot)
        if self.extract_date_features:
            X = extract_date_features(X)
        return X


############################
## Features based on text ##
############################

def extract_text_features_col(df, col):
    """Extract text features from a single column in a df. Return an occurrence dataframe encoding based on these features."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[col])
    features = vectorizer.get_feature_names()
    matrix = X.toarray()
    features = [col + '#' + x for x in features]  # Prefix each feature with name of the originating raw feature
    col_features = pd.DataFrame(data=matrix, columns=features)
    return col_features

def extract_text_features_cols_hot(df, cols):
    """Create encoded feature columns for the dataframe, based on the defined text columns."""
    all_col_features = []
    for col in cols:
        col_features = extract_text_features_col(df[col])
        all_col_features.append(col_features)
    df = pd.concat([df] + all_col_features, axis=1, sort=False)
    return df

def extract_categorical_cols_hot(df, cols):
    """Create HOT encoded feature columns for the dataframe, based on the defined categorical columns."""
    all_col_features = []
    for col in cols:
        print(f"Now extracting features from column: '{col}'.")
        col_features = pd.get_dummies(df[col], prefix=col, prefix_sep='#')
        all_col_features.append(col_features)
        print("Done!")
    df = pd.concat([df] + all_col_features, axis=1, sort=False)
    return df

def extract_categorical_cols_no_hot(df, cols):
    """Create a numerically encoded feature column in the df based on each defined categorical column."""
    all_col_features = []
    for col in cols:
        print(f"Now extracting features from column: '{col}'.")
        col_features = df.col.astype('category').cat.codes
        all_col_features.append(col_features)
        print("Done!")
    df = pd.concat([df] + all_col_features, axis=1, sort=False)
    return df


#####################
## Other Features  ##
#####################

def extract_date_features(df):
    """Expand datetime values into individual features."""
    for col in df.select_dtypes(include=['datetime64[ns]']):
        print(f"Now extracting features from column: '{col}'.")
        df[col + '_month'] = pd.DatetimeIndex(df[col]).month
        df[col + '_day'] = pd.DatetimeIndex(df[col]).day
        df[col + '_weekday'] = pd.DatetimeIndex(df[col]).weekday
        df.drop(columns=[col], inplace=True)
        print("Done!")
    return df