"""
build_model.py

This script takes the extracted BWV features and builds a prediction model.

Input: extracted BWV features.
Output: prediction model for prediction social housing rental fraud.

Written by Swaan Dekkers & Thomas Jongstra
"""

# Source this script from collect_data_and_make_model.ipynb.

# General imports
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Import ML Methods
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LassoCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# Import samplers for handling data imbalance
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
from imblearn.over_sampling import RandomOverSampler

# Import functions to evaluate algorithm performance
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix
from imblearn.metrics import classification_report_imbalanced


def split_data(df):
    # Creating sets for model building and testing
    # 1. Training set (70%) - for building the model
    # 2. Development set a.k.a. hold-out set (15%) - for optimizing model parameters
    # 3. Test set (15%) - For testing the performance of the tuned model

    # Split data into features (X) and labels (y).
    X = df.drop('woonfraude', axis=1)
    y = df.woonfraude
    print('Original dataset shape %s' % Counter(y))

    # Compute the amount of datapoints.
    n = X.shape[0]

    # Create train, dev and test sets using the feature values of the examples.
    X_shuffled = X.sample(frac=1, random_state=0)
    X_train_org, X_dev, X_test = np.split(X_shuffled, [int(n*.7), int(n*.85)])

    # Create train, dec and test sets of the corresponding example labels.
    y_shuffled = y.sample(frac=1, random_state=0)
    y_train_org, y_dev, y_test = np.split(y_shuffled, [int(n*.7), int(n*.85)])


    # Print some information about the train/dev/test set sizes.
    print('Training set shape %s' % Counter(y_train_org))
    print('Development set shape %s' % Counter(y_dev))
    print('Testing set shape %s' % Counter(y_test))

    return X_train_org, X_dev, X_test, y_train_org, y_dev, y_test


def undersample(X_train_org, y_train_org):
    # TODO: finish writing this function
    pass


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
    report = classification_report_imbalanced(y_true=y_dev, y_pred=y_pred)
    print(report)
    return precision, recall, f1, conf, report


def run_knn(X_train, y_train, X_dev, y_dev, n_neighbors=11):
    """Run a KNN model. Return results"""

    # Build KNN model using several neighbors.
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit model on training data.
    knn.fit(X_train, y_train)

    # Create predictions.
    y_pred = knn.predict(X_dev)

    # Compute and show performance statistics.
    precision, recall, f1, conf, report = evaluate_performance(y_pred=y_pred, y_dev=y_dev)

    return knn, precision, recall, f1, conf, report


def run_lasso(X_train, y_train, X_dev, y_dev):
    """Run a lasso model. Return results"""

    # Fit lasso model on training data.
    reg = LassoCV(cv=5, random_state=0).fit(X_train, y_train)

    # Create predictions.
    y_pred = reg.predict(X_dev) > 0.12

    # Compute and show performance statistics.
    precision, recall, f1, conf, report = evaluate_performance(y_pred=y_pred, y_dev=y_dev)

    return reg, precision, recall, f1, conf, report


def run_linear_svc(X_train, y_train, X_dev, y_dev):
    """Run linear support vector classification. Return results."""

    # Fit model to training data.
    clf = LinearSVC(random_state=0, tol=1e-5, max_iter=1000)
    clf.fit(X_train, y_train)

    # Create predictions.
    y_pred = clf.predict(X_dev)

    # Compute and show performance statistics.
    precision, recall, f1, conf, report = evaluate_performance(y_pred=y_pred, y_dev=y_dev)

    return clf, precision, recall, f1, conf, report


def run_gaussian_naive_bayes(X_train, y_train, X_dev, y_dev):
    """Run gaussian naive bayes. Return results."""

    # Fit model to training data.
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Create predictions.
    y_pred = gnb.predict(X_dev)

    # Compute and show performance statistics.
    precision, recall, f1, conf, report = evaluate_performance(y_pred=y_pred, y_dev=y_dev)

    return gnb, precision, recall, f1, conf, report

# def import_features():
#     # Import de output van extract_features.py
#     df = pd.read_pickle('../../data/df_adres_features.pkl')
#     df = df.fillna('')
#     df = df[df['landelijk_bag']!='']

#     print(df.head())
#     return df

# def split_data():
#     # Split dataset into train and test data, make sure to do this random as the data is organised based on date
#     pass

# def train_model():
#     # train moodel on train set
#     pass

# def test_model():
#     # test model
#     pass

# def main():
#     import_features()
#     split_data()
#     train_model()
#     test_model()

# if __name__ == "__main__":
#     main()

