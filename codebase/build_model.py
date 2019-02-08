"""
build_model.py

This script takes the extracted BWV features and builds a prediction model.

Input: extracted BWV features.
Output: prediction model for prediction social housing rental fraud.

Written by Swaan Dekkers & Thomas Jongstra
"""

# Source this script from collect_data_and_make_model.ipynb.

# Imports
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Samplers for handling data imbalance
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import BaseSampler


def prepare_data(adres, zaken):
    """Combine address and cases data"""
    # Add address information to each case (this duplicates 'wzs_id' and 'wzs_update_datumtijd').
    df = zaken.merge(adres, on='adres_id', how='left')
    # Remove all columns which would not yet be available when cases (zaken) are newly opened.
    df = df.drop(columns=['einddatum', 'afg_code_beh', 'afs_code', 'afs_oms', 'afg_code_afs', 'wzs_update_datumtijd_x', 'wzs_update_datumtijd_y', 'mededelingen'])
    # Also remove columns with more than 40% nonetype data.
    df.drop(columns=['hsltr', 'toev'], inplace=True)
    return df


def split_data(df):
	# Creating sets for model building and testing
	# 1. Training set (70%) - for building the model
	# 2. Development set a.k.a. hold-out set (15%) - for optimizing model parameters
	# 3. Test set (15%) - For testing the performance of the tuned model

	# Split data into features (X) and labels (y).
	X = df.drop('woonfraude', axis=1)
	y = df.woonfraude
	print('Original dataset shape %s' % Counter(y_org))

	n = X.shape[0]

	# Create train, dev and test sets using the feature values of the examples.
	X_shuffled = X.sample(frac=1, random_state=0)
	X_train_org, X_dev, X_test = np.split(X_shuffled, [int(n*.7), int(n*.85)])

	# Create train, dec and test sets of the corresponding example labels.
	y_shuffled = y.sample(frac=1, random_state=0)
	y_train_org, y_dev, y_test = np.split(y_shuffled, [int(n*.7), int(n*.85)])

	print('Training set shape %s' % Counter(y_train_org))
	print('Development set shape %s' % Counter(y_dev))
	print('Testing set shape %s' % Counter(y_test))

	return X_train_org, X_dev, X_test, y_train_org, y_dev, y_test


def augment_data(X_train_org, y_train_org, sampler='ADASYN'):
    """Synthesize more positive samples using one of various techniques (ADASYN, SMOTE, etc.)"""

    # Set random seed.
    random_seed = 42

    # Pick sampler based on variable setting.
    if sampler == 'ADASYN':
        samp = ADASYN(random_state=random_seed)
    if sampler == 'SMOTE':
        samp = SMOTE(random_state=random_seed)
    if sampler == 'BorderlineSMOTE':
        samp = BorderlineSMOTE(random_state=random_seed)
    if sampler == 'SVMSMOTE':
        samp = SVMSMOTE(random_state=random_seed)
    if sampler == 'SMOTENC':
        samp = SMOTENC(random_state=random_seed, categorical_features=[2])
    if sampler == 'RandomOverSampler':
        samp = RandomOverSampler(random_state=random_seed)
    if sampler == 'BaseSampler':
        samp = BaseSampler(random_state=random_seed)

    # The resulting X_train and y_train are numpy arrays.
    X_train, y_train = samp.fit_resample(X_train_org, y_train_org)

    # Turn X_train and y_train into Pandas dataframes again.
    X_train = pd.DataFrame(X_train, columns = X_train_org.columns)
    y_train = pd.Series(y_train)

    # Show counts
    print('Resampled training dataset shape %s' % Counter(y_train))

    return X_train, y_train


def evaluate_performance(y_pred, y_dev):
    """Compute and return the prediction performance."""
    precision = precision_score(y_dev, y_pred)
    recall = recall_score(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred)
    conf = confusion_matrix(y_dev, y_pred) / len(y_pred)
    return precision, recall, f1, conf


def run_knn(X_train, y_train, X_dev, y_dev, n_neighbors=11):
    """Run a KNN model. Return results"""

    # Build KNN model using several neighbors.
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit model on training data.
    knn.fit(X_train, y_train)

    # Create predictions.
    y_pred = knn.predict(X_dev)

    # Compute performance statistics.
    precision, recall, f1, conf = evaluate_performance(y_pred=y_pred, y_dev=y_dev)

    return knn, precision, recall, f1, conf


def run_lasso(X_train, y_train, X_dev, y_dev):
    """Run a lasso model. Return results"""

    # Fit lasso model on training data.
    reg = LassoCV(cv=5, random_state=0).fit(X, y)

    # Create predictions.
    y_pred = reg.predict(X_dev) > 0.12

    # Compute performance statistics.
    precision, recall, f1, conf = evaluate_performance(y_pred=y_pred, y_dev=y_dev)

    return reg, precision, recall, f1, conf


# def import_features():
# 	# Import de output van extract_features.py
# 	df = pd.read_pickle('../../data/df_adres_features.pkl')
# 	df = df.fillna('')
# 	df = df[df['landelijk_bag']!='']

# 	print(df.head())
# 	return df

# def split_data():
# 	# Split dataset into train and test data, make sure to do this random as the data is organised based on date
# 	pass

# def train_model():
# 	# train moodel on train set
# 	pass

# def test_model():
# 	# test model
# 	pass

# def main():
# 	import_features()
# 	split_data()
# 	train_model()
# 	test_model()

# if __name__ == "__main__":
#     main()

