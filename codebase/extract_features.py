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
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler


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


def add_person_features(df, personen):
    """Add features relating to persons to addresses. Currently adds nr of persons per address."""

    # Compute age of people in years (float)
    today = pd.to_datetime('today')
    # Set all dates within range allowed by Pandas (584 years?)
    personen['geboortedatum'] = pd.to_datetime(personen['geboortedatum'], errors='coerce')
    # Get the most frequent birthdate (mode).
    geboortedatum_mode = personen['geboortedatum'].mode()[0]
    # Compute the age (result is a TimeDelta).
    personen['leeftijd'] = today - personen['geboortedatum']
    # Convert the age to an approximation in years ("smearin out" the leap years).
    personen['leeftijd'] = personen['leeftijd'].apply(lambda x: x.days / 365.25)


    # Find the matching address ids between the adres/zaken df and the personen df.
    adres_ids = df.adres_id
    personen_adres_ids = personen.ads_id_wa
    intersect = set(adres_ids).intersection(set(personen_adres_ids))

    # Iterate over all matching address ids and find all people at each address.
    inhabitant_locs = {}
    print("Now looping over all address ids that have a link with one or more inhabitants...")
    for i, adres_id in enumerate(intersect):
        if i % 1000 == 0:
            print(i)
        inhabitant_locs[adres_id] = personen_adres_ids[personen_adres_ids == adres_id]

    # Create a new column in the dataframe showing the amount of people at each address.
    # TODO: this step currently takes a few minutes to complete, should still be optimized.
    df['aantal_personen'] = 0
    df['leeftijd_jongste_persoon'] = -1.
    df['leeftijd_oudste_persoon'] = -1.
    df['aantal_kinderen'] = 0
    df['percentage_kinderen'] = -1.
    df['aantal_mannen'] = 0
    df['percentage_mannen'] = -1.
    df['gemiddelde_leeftijd'] = -1.
    df['stdev_leeftijd'] = -1.
    df['aantal_achternamen'] = 0
    df['percentage_achternamen'] = -1.
    for i in range(1,8):
        df[f'gezinsverhouding_{i}'] = 0
        df[f'percentage_gezinsverhouding_{i}'] = 0.
    print("Now looping over all rows in the main dataframe in order to add person information...")
    for i in df.index[:10]:
        if i % 1000 == 0:
            print(i)
        row = df.iloc[i]
        adres_id = row['adres_id']
        try:
            # Get the inhabitants for the current address
            inhab_locs = inhabitant_locs[adres_id].keys()
            inhab = personen.loc[inhab_locs]

            # Totaal aantal personen (int)
            aantal_personen = len(inhab)
            df.at[i, 'aantal_personen'] = aantal_personen

            # Leeftijd jongste persoon (float)
            leeftijd_jongste_persoon = min(inhab['leeftijd'])
            df.at[i, 'leeftijd_jongste_persoon'] = leeftijd_jongste_persoon

            # Leeftijd oudste persoon (float)
            leeftijd_oudste_persoon = max(inhab['leeftijd'])
            df.at[i, 'leeftijd_oudste_persoon'] = leeftijd_oudste_persoon

            # Aantal kinderen ingeschreven op adres (int/float)
            aantal_kinderen = sum(inhab['leeftijd'] < 18)
            df.at[i, 'aantal_kinderen'] = aantal_kinderen
            df.at[i, 'percentage_kinderen'] = aantal_kinderen / aantal_personen

            # Aantal mannen (int/float)
            aantal_mannen = sum(inhab.geslacht == 'M')
            df.at[i, 'aantal_mannen'] = aantal_mannen
            df.at[i, 'percentage_mannen'] = aantal_mannen / aantal_personen

            # Gemiddelde leeftijd (float)
            gemiddelde_leeftijd = inhab.leeftijd.mean()
            df.at[i, 'gemiddelde_leeftijd'] = gemiddelde_leeftijd

            # Standardeviatie van leeftijd (float). Set to 0 when the sample size is 1.
            stdev_leeftijd = inhab.leeftijd.std()
            df.at[i, 'stdev_leeftijd'] = stdev_leeftijd if aantal_personen > 1 else 0

            # Aantal verschillende achternamen (int/float)
            aantal_achternamen = inhab.naam.nunique()
            df.at[i, 'aantal_achternamen'] = aantal_achternamen
            df.at[i, 'percentage_achternamen'] = aantal_achternamen / aantal_personen

            # Gezinsverhouding (frequency count per klasse) (int/float)
            gezinsverhouding = inhab.gezinsverhouding.value_counts()
            for key in gezinsverhouding.keys():
                val = gezinsverhouding[key]
                df.at[i, f'gezinsverhouding_{key}'] = val
                df.at[i, f'percentage_gezinsverhouding_{key}'] = val / aantal_personen

        except KeyError:
            pass
    print("...done!")

    return df

###########################
##### Feature Scaling #####
###########################

def scale_data(df, cols):
    """Scale data using the sklearn StandardScaler for the defined columns."""

    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df