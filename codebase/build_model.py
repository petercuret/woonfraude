"""
build_model.py

This script takes the enriched BWV data and builds a prediction model.

Input: enriched BWV data (~38k entries @ 2018-11-21)
Output: a prediction model for prediction housing fraud

Written by Swaan Dekkers & Thomas Jongstra
"""

# Source this script from collect_data_and_make_model.ipynb.

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def import_features():
	# Import de output van extract_features.py
	df = pd.read_pickle('../../data/df_adres_features.pkl')  
	df = df.fillna('')
	df = df[df['landelijk_bag']!='']

	print(df.head())
	return df

def split_data():
	# Split dataset into train and test data, make sure to do this random as the data is organised based on date
	pass

def train_model():
	# train moodel on train set
	pass

def test_model():
	# test model 
	pass

def main():
	import_features()
	split_data()
	train_model()
	test_model()

if __name__ == "__main__":
    main()

