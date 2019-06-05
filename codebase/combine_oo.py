"""
enrich_oo.py

This script aims to take the cleaned BWV data, and enrich it with up-to-date BAG data.
In this process, all BWV entries that cannot be coupled with new BAG data are removed.
After running this script, the resulting data should be ready for feature extraction.

Input: cleaned BWV data (~48k entries @ 2018-11-21).
Output: enriched BWV data, i.e. coupled with up-to-date BAG data (~38k entries @ 2018-11-21)
        unenriched BWV data (~10k entries, no match found with BAG code)

Written by Swaan Dekkers & Thomas Jongstra
"""


# Import statements
from pathlib import Path
import pandas.io.sql as sqlio
import pandas as pd
import psycopg2
import numpy as np


class AdresCombineTransformer(BaseEstimator, TransformerMixin):
    """Class for combining/integrating multiple dataframes into a single dataframe."""

    def __init__(self,
                 zaken_features = None,  # If this contains the adres dataframe, it will be merged onto the adres dataframe.
                 bag_features = None,  # If this contains the bag dataframe, it will be merged onto the adres dataframe.
                 person_features = None, # If this contains the person dataframe, it will be merged onto the adres dataframe.
                 select_closed_cases_zaken_stadia = None  # If this contains a list containing the zaken and stadia dataframes, then all closed cases will be removed.
                ):
            self.adres_features = adres_features
            self.bag_features = bag_features
            self.person_features = person_features
            self.select_closed_cases_zaken_stadia = select_closed_cases_zaken_stadia

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.adres_features != None:
            X = self.zaken_features.merge(X, on='adres_id', how='left')
        if self.bag_features != None:
            X = adres_bag_enrich(X, self.bag_features)
        if self.person_features != None:
            X = add_person_features(X, self.person_features)
        if self.select_closed_cases_zaken_stadia != None:
            zaken, stadia = self.select_closed_cases_zaken_stadia
            X = select_closed_cases(X, zaken, stadia)


def adres_bag_enrich(adres, bag):
    """Enrich the adres data with information from the BAG data."""
    # bag = prepare_bag(bag)
    # adres = prepare_adres(adres)
    adres = match_bwv_bag(adres, bag)
    # bag = replace_string_nan_bag(bag)
    # adres = replace_string_nan_adres(adres)
    impute_values_for_bagless_addresses(adres)
    adres.name = 'adres'
    return adres


def impute_values_for_bagless_addresses(adres):
    """Impute values for adresses where no BAG-match could be found."""
    clean.impute_missing_values(adres)
    clean.impute_missing_values_mode(adres, ['status_coordinaat_code@bag', 'indicatie_geconstateerd@bag', 'indicatie_in_onderzoek@bag', 'woningvoorraad@bag'])
    adres.fillna(value={'type_woonobject_omschrijving': 'None',
                      'eigendomsverhouding_id@bag': 'None',
                      'financieringswijze_id@bag': -1,
                      'gebruik_id@bag': -1,
                      'reden_opvoer_id@bag': -1,
                      'status_id@bag': -1,
                      'toegang_id@bag': 'None'}, inplace=True)


def match_bwv_bag(adres, bag):
    # Merge dataframes on adres dataframe
    new_df = pd.merge(adres, bag,  how='left', left_on=['sttnaam','hsnr'], right_on = ['_openbare_ruimte_naam@bag', '_huisnummer@bag'])

    # Find id's that have a direct match and that have multiple matches
    g = new_df.groupby('adres_id')
    df_direct = g.filter(lambda x: len(x) == 1)
    df_multiple = g.filter(lambda x: len(x) > 1)

    # Make multiplematch more specific to construct perfect match
    df_multiple = df_multiple[(df_multiple['hsltr'] == df_multiple['_huisletter@bag']) & (df_multiple['toev'] == df_multiple['_huisnummer_toevoeging@bag'])]

    # Concat df_direct and df_multiple
    df_result = pd.concat([df_direct, df_multiple])

    # Because of the seperation of an object, there are two matching objects. Keep the oldest object with definif point
    df_result = df_result.sort_values(['adres_id', 'status_coordinaat_code@bag'])
    df_result = df_result.drop_duplicates(subset='adres_id', keep='first')

    # Add adresses without match
    final_df = pd.merge(adres, df_result,  how='left', on='adres_id', suffixes=('', '_y'))
    final_df.drop(list(final_df.filter(regex='_y$')), axis=1, inplace=True)

    # Drop unwanted bag columns
    bag_remove = ['einde_geldigheid@bag',               # Only 2 entries in column.
                  'verhuurbare_eenheden@bag',           # Only ~2k entries in column.
                  'geometrie@bag',                      # Needs a lot of processing before being useful.
                  'bron_id@bag',                        # Only 2 entries in column.
                  'locatie_ingang_id@bag',              # Only 2 entries in column.
                  'reden_afvoer_id@bag',                # Only a few entries in column.
                  '_gebiedsgerichtwerken_id@bag',       # Superfluous (gebied).
                  '_grootstedelijkgebied_id@bag',       # Superfluous (grootstedelijkgebied).
                  'buurt_id@bag',                       # Superfluous (buurt).
                  '_openbare_ruimte_naam@bag',          # Superfluous (straatnaam).
                  '_huisnummer@bag',                    # Superfluous (huisnummer).
                  '_huisletter@bag',                    # Superfluous (huisletter).
                  '_huisnummer_toevoeging@bag',         # Superfluous (huisnummer toevoeging).
                  'vervallen@bag',                      # Superfluous (all values in col are equal).
                  'mutatie_gebruiker@bag',              # Superfluous (all values in col are equal).
                  'document_mutatie@bag',               # Not available at time of signal.
                  'date_modified@bag',                  # Not available at time of signal.
                  'document_nummer@bag',                # Not needed? (Swaan?)
                  'status_coordinaat_omschrijving@bag', # Not needed? (Swaan?)
                  'type_woonobject_code@bag',           # Not needed? (Swaan?)
                  'id@bag',                             # Not needed.
                  'landelijk_id@bag'                    # Not needed.
                  ]
    final_df.drop(columns=bag_remove, inplace=True)

    return final_df


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
    df['aantal_vertrokken_personen'] = -1
    df['aantal_overleden_personen'] = -1
    df['aantal_niet_uitgeschrevenen'] = -1
    df['leegstand'] = True
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
    for i in df.index:
        if i % 1000 == 0:
            print(i)
        row = df.iloc[i]
        adres_id = row['adres_id']
        try:
            # Get the inhabitants for the current address.
            inhab_locs = inhabitant_locs[adres_id].keys()
            inhab = personen.loc[inhab_locs]

            # Check whether any registered inhabitants have left Amsterdam or have passed away.
            aantal_vertrokken_personen = sum(inhab["vertrekdatum_adam"].notnull())
            aantal_overleden_personen = sum(inhab["overlijdensdatum"].notnull())
            aantal_niet_uitgeschrevenen = len(inhab[inhab["vertrekdatum_adam"].notnull() | inhab["overlijdensdatum"].notnull()])
            df['aantal_vertrokken_personen'] = aantal_vertrokken_personen
            df['aantal_overleden_personen'] = aantal_overleden_personen
            df['aantal_niet_uitgeschrevenen'] = aantal_niet_uitgeschrevenen
            # If there are more inhabitants than people that are incorrectly still registered, then there is no 'leegstand'.
            if len(inhab) > aantal_niet_uitgeschrevenen:
                df['leegstand'] = False

            # Totaal aantal personen (int).
            aantal_personen = len(inhab)
            df.at[i, 'aantal_personen'] = aantal_personen

            # Leeftijd jongste persoon (float).
            leeftijd_jongste_persoon = min(inhab['leeftijd'])
            df.at[i, 'leeftijd_jongste_persoon'] = leeftijd_jongste_persoon

            # Leeftijd oudste persoon (float).
            leeftijd_oudste_persoon = max(inhab['leeftijd'])
            df.at[i, 'leeftijd_oudste_persoon'] = leeftijd_oudste_persoon

            # Aantal kinderen ingeschreven op adres (int/float).
            aantal_kinderen = sum(inhab['leeftijd'] < 18)
            df.at[i, 'aantal_kinderen'] = aantal_kinderen
            df.at[i, 'percentage_kinderen'] = aantal_kinderen / aantal_personen

            # Aantal mannen (int/float).
            aantal_mannen = sum(inhab.geslacht == 'M')
            df.at[i, 'aantal_mannen'] = aantal_mannen
            df.at[i, 'percentage_mannen'] = aantal_mannen / aantal_personen

            # Gemiddelde leeftijd (float).
            gemiddelde_leeftijd = inhab.leeftijd.mean()
            df.at[i, 'gemiddelde_leeftijd'] = gemiddelde_leeftijd

            # Standardeviatie van leeftijd (float). Set to 0 when the sample size is 1.
            stdev_leeftijd = inhab.leeftijd.std()
            df.at[i, 'stdev_leeftijd'] = stdev_leeftijd if aantal_personen > 1 else 0

            # Aantal verschillende achternamen (int/float).
            aantal_achternamen = inhab.naam.nunique()
            df.at[i, 'aantal_achternamen'] = aantal_achternamen
            df.at[i, 'percentage_achternamen'] = aantal_achternamen / aantal_personen

            # Gezinsverhouding (frequency count per klasse) (int/float).
            gezinsverhouding = inhab.gezinsverhouding.value_counts()
            for key in gezinsverhouding.keys():
                val = gezinsverhouding[key]
                df.at[i, f'gezinsverhouding_{key}'] = val
                df.at[i, f'percentage_gezinsverhouding_{key}'] = val / aantal_personen

        except (KeyError, ValueError) as e:
            pass

    print("...done!")

    return df


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

    # Remove temporary mask
    selected_zaken.drop(columns=['mask'], inplace=True)

    # Print results.
    print(f'Selected {len(selected_zaken)} closed cases from a total of {len(zaken)} cases.')

    # Only return the relevant selection of cases.
    return selected_zaken