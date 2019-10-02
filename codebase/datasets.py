####################################################################################################
# datasets.py                                                                                      #
#                                                                                                  #
# This code implements dataset-specific downloading, saving and data-transformation operations.    #
#                                                                                                  #
# Written by Swaan Dekkers & Thomas Jongstra                                                       #
####################################################################################################

#############
## Imports ##
#############

from pathlib import Path
import pandas.io.sql as sqlio
import pandas as pd
import numpy as np
import requests
import psycopg2
import time
import os
import re

# Import own modules.
import config, clean

# Define HOME and DATA_PATH on a global level.
HOME = os.path.abspath('E:\\Jasmine')  # Home path for old VAO.
# USERNAME = os.path.basename(HOME)
# HOME = os.path.join('/data', USERNAME)  # Set home for new VAO.
# DATA_PATH = os.path.join(HOME, 'Documents/woonfraude/data/')
DATA_PATH = os.path.abspath('E:\\Jasmine\\woonfraude\\data')


###################
## Dataset class ##
###################

class MyDataset():
    """
    Generic dataset template implementing generic functionalities.
    All other dataset classes inherit from this one.
    """

    # Define class attributed (name, id_column), which have to get a value in all subclasses.
    name = None
    table_name = None
    id_column = None


    def __init__(self):
        self._data = None
        self._version = None


    def save(self):
        """Save a previously processed version of the dataset."""
        print(f"Saving version '{self.version}' of dataframe '{self.name}'.")
        save_dataset(self.data, self.name, self.version)


    def load(self, version):
        """Load a previously processed version of the dataset."""
        try:
            self.data = load_dataset(self.name, version)
            self.version = version
            print(f"Version '{self.version}' of dataset '{self.name}' loaded!")
        except FileNotFoundError as e:
            print(f"Sorry, version '{version}' of dataset '{self.name}' is not available on local storage.")
            if version == 'download':
                print("The software will now download the dataset instead.")
                self._force_download()
            else:
                print("Please try loading another version, or creating the version you need.")


    def download(self, force=False, limit: int = 9223372036854775807):
        """Download a copy of the dataset, or restore a previous version if available."""
        if force == True:
            self._force_download()
        else:
            try:
                self.data = load(version='download')
                print("Loaded cached dataset from local storage. To force a download, \
                       use the 'download' method with the flag 'force' set to 'True'")
            except Exception:
                self._force_download()


    def _force_download(self, limit=9223372036854775807):
        """Force a dataset download."""
        self.data = download_dataset(self.name, self.table_name, limit)
        self.version = 'download'
        save_dataset(self.data, self.name, self.version)  # cache dataset locally


def download_dataset(dataset_name, table_name, limit=9223372036854775807):
        """Download a new copy of the dataset from its source."""

        start = time.time()
        print(f"#### Starting download of dataset '{dataset_name}'...")

        if dataset_name == 'bbga':
            # Download BBGA file, interpret as dataframe, and return.
            url = "https://api.data.amsterdam.nl/dcatd/datasets/G5JpqNbhweXZSw/purls/LXGOPUQQfAXBbg"
            res = requests.get(url)
            df = pd.read_csv(res.content)
            return df

        # Create a server connection based on the needed tables. The datasets are available on two different servers.
        if table_name in ['import_adres', 'import_wvs', 'import_stadia']:
            conn = psycopg2.connect(host = config.HOST_1,
                                    dbname = config.DB_1,
                                    user = config.USER_1,
                                    password = config.PASSWORD_1)

        if table_name in ['bwv_personen', 'bwv_hotline_melding', 'bwv_adres_periodes']:
            conn = psycopg2.connect(host = config.HOST_2,
                                    dbname = config.DB_2,
                                    user = config.USER_2,
                                    password = config.PASSWORD_2)

        # if table_name in ['bag_nummeraanduiding']:
        #     conn = psycopg2.connect(host = config.BAG_HOST,
        #                     dbname = config.BAG_DB,
        #                     user = config.BAG_USER,
        #                     password = config.BAG_PASSWORD)

        # Create query to download the specific table data from the server.
        sql = f"select * from public.{table_name} limit {limit};"

        # if table_name in ['bag_nummeraanduiding']:
        #     sql = """
        #     SELECT *
        #     FROM public.bag_nummeraanduiding
        #     FULL JOIN bag_ligplaats ON bag_nummeraanduiding.ligplaats_id = bag_ligplaats.id
        #     FULL JOIN bag_standplaats ON bag_nummeraanduiding.standplaats_id = bag_standplaats.id
        #     FULL JOIN bag_verblijfsobject ON bag_nummeraanduiding.verblijfsobject_id = bag_verblijfsobject.id;
        #     """

        # Get data & convert to dataframe.
        df = sqlio.read_sql_query(sql, conn)

        # Close connection to server.
        conn.close()

        # Name dataframe according to table name. Beware: name will be removed by pickling.
        df.name = dataset_name

        if dataset_name == 'bag':
            df = apply_bag_colname_fix(df)

        print("\n#### ...download done! Spent %.2f seconds.\n" % (time.time()-start))
        return df


def apply_bag_colname_fix(df):
    """Fix BAG columns directly after download."""

    # Rename duplicate columns using a suffix _idx.
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols

    # Rename huisnummer, huisletter and huisnummer_toevoeging columns to specify their table origin.
    df = df.rename(index=str, columns={'huisnummer': 'huisnummer_nummeraanduiding',
                                       'huisletter': 'huisletter_nummeraanduiding',
                                       'huisnummer_toevoeging': 'huisnummer_toevoeging_nummeraanduiding'})

    # Rename columns (two copies).
    d_rename1 = {}
    l_rename1 = ['_huisnummer', '_huisletter', '_huisnummer_toevoeging']
    for r in l_rename1:
        d_rename1[r] = r + '_ligplaats'
        d_rename1[r + '_1'] = r + '_standplaats'
        d_rename1[r + '_2'] = r + '_verblijfsobject'
    df = df.rename(index=str, columns=d_rename1)

    # Rename columns (three copies).
    d_rename2 = {}
    l_rename2 = ['indicatie_geconstateerd', 'indicatie_in_onderzoek', 'geometrie',
                '_gebiedsgerichtwerken_id', '_grootstedelijkgebied_id', 'buurt_id']
    for r in l_rename2:
        d_rename2[r] = r + '_ligplaats'
        d_rename2[r + '_1'] = r + '_standplaats'
        d_rename2[r + '_2'] = r + '_verblijfsobject'
    df = df.rename(index=str, columns=d_rename2)

    # Rename columns (four copies).
    d_rename3 = {}
    l_rename3 = ['document_mutatie', 'document_nummer', 'begin_geldigheid', 'einde_geldigheid',
                'mutatie_gebruiker', 'id', 'landelijk_id', 'vervallen', 'date_modified',
                '_openbare_ruimte_naam', 'bron_id', 'status_id']
    for r in l_rename3:
        d_rename3[r] = r + '_nummeraanduiding'
        d_rename3[r + '_1'] = r + '_ligplaats'
        d_rename3[r + '_2'] = r + '_standplaats'
        d_rename3[r + '_3'] = r + '_verblijfsobject'
    df = df.rename(index=str, columns=d_rename3)

    return df


class AdresDataset(MyDataset):
    """Create a dataset for the adres data."""

    # Set the class attributes.
    name = 'adres'
    table_name = 'import_adres'
    id_column = 'adres_id'


    def extract_leegstand(self):
        """Create a column indicating leegstand (no inhabitants on the address)."""
        self.data['leegstand'] = ~self.data.inwnrs.notnull()
        self.version += '_leegstand'
        self.save()


    def enrich_with_woning_id(self):
        """Add woning ids to the adres dataframe."""
        adres_periodes = download_dataset('bwv_adres_periodes', 'bwv_adres_periodes')
        self.data = self.data.merge(adres_periodes[['ads_id', 'wng_id']], how='left', left_on='adres_id', right_on='ads_id')
        self.version += '_woningId'
        self.save()


    def prepare_bag(self, bag):
        # To int
        bag['huisnummer_nummeraanduiding'] = bag['huisnummer_nummeraanduiding'].astype(int)
        bag['huisnummer_nummeraanduiding'] = bag['huisnummer_nummeraanduiding'].replace(0, -1)

        # Fillna and replace ''
        bag['huisletter_nummeraanduiding'] = bag['huisletter_nummeraanduiding'].replace('', 'None')

        # bag['_openbare_ruimte_naam@bag'] = bag['_openbare_ruimte_naam@bag'].fillna('None')
        bag['_openbare_ruimte_naam_nummeraanduiding'] = bag['_openbare_ruimte_naam_nummeraanduiding'].replace('', 'None')

        # bag['_huisnummer_toevoeging@bag'] = bag['_huisnummer_toevoeging@bag'].fillna('None')
        bag['huisnummer_toevoeging_nummeraanduiding'] = bag['huisnummer_toevoeging_nummeraanduiding'].replace('', 'None')
        return bag


    def prepare_adres(self, adres):
        # To int
        adres['hsnr'] = adres['hsnr'].astype(int)
        adres['hsnr'] = adres['hsnr'].replace(0, -1)

        return adres


    def replace_string_nan_adres(self, adres):
        adres['hsnr'] = adres['hsnr'].replace(-1, np.nan)
        adres['sttnaam'] = adres['sttnaam'].replace('None', np.nan)
        adres['hsltr'] = adres['hsltr'].replace('None', np.nan)
        adres['toev'] = adres['toev'].replace('None', np.nan)
        adres['huisnummer_nummeraanduiding'] = adres['huisnummer_nummeraanduiding'].replace(-1, np.nan)
        adres['huisletter_nummeraanduiding'] = adres['huisletter_nummeraanduiding'].replace('None', np.nan)
        adres['_openbare_ruimte_naam_nummeraanduiding'] = adres['_openbare_ruimte_naam_nummeraanduiding'].replace('None', np.nan)
        adres['huisnummer_toevoeging_nummeraanduiding'] = adres['huisnummer_toevoeging_nummeraanduiding'].replace('None', np.nan)
        return adres


    def match_bwv_bag(self, adres, bag):
        # Merge dataframes on adres dataframe.
        new_df = pd.merge(adres, bag,  how='left', left_on=['sttnaam','hsnr'], right_on = ['_openbare_ruimte_naam_nummeraanduiding', 'huisnummer_nummeraanduiding'])

        # Find id's that have a direct match and that have multiple matches.
        g = new_df.groupby('adres_id')
        df_direct = g.filter(lambda x: len(x) == 1)
        df_multiple = g.filter(lambda x: len(x) > 1)

        # Make multiplematch more specific to construct perfect match.
        df_multiple = df_multiple[(df_multiple['hsltr'] == df_multiple['huisletter_nummeraanduiding']) & (df_multiple['toev'] == df_multiple['huisnummer_toevoeging_nummeraanduiding'])]

        # Concat df_direct and df_multiple.
        df_result = pd.concat([df_direct, df_multiple])

        # Because of the seperation of an object, there are two matching objects. Keep the oldest object with definif point.
        df_result = df_result.sort_values(['adres_id', 'status_coordinaat_code'])
        df_result = df_result.drop_duplicates(subset='adres_id', keep='first')

        # Add adresses without match.
        final_df = pd.merge(adres, df_result,  how='left', on='adres_id', suffixes=('', '_y'))
        final_df.drop(list(final_df.filter(regex='_y$')), axis=1, inplace=True)

        # Set the name of the final adres dataframe again.
        final_df.name = 'adres'

        return final_df


    def impute_values_for_bagless_addresses(self, adres):
        """Impute values for adresses where no BAG-match could be found."""
        clean.impute_missing_values(adres)
        # clean.impute_missing_values_mode(adres, ['status_coordinaat_code@bag'])
        adres.fillna(value={'huisnummer_nummeraanduiding': 0,
                            'huisletter_nummeraanduiding': 'None',
                            '_openbare_ruimte_naam_nummeraanduiding': 'None',
                            'huisnummer_toevoeging_nummeraanduiding': 'None',
                            'type_woonobject_omschrijving': 'None',
                            'eigendomsverhouding_id': 'None',
                            'financieringswijze_id': -1,
                            'gebruik_id': -1,
                            'reden_opvoer_id': -1,
                            'status_id_verblijfsobject': -1,
                            'toegang_id': 'None'}, inplace=True)
        return adres


    def enrich_with_bag(self, bag):
        """Enrich the adres data with information from the BAG data. Uses the bag dataframe as input."""
        bag = self.prepare_bag(bag)
        self.data = self.prepare_adres(self.data)
        self.data = self.match_bwv_bag(self.data, bag)
        self.data = self.replace_string_nan_adres(self.data)
        self.data = self.impute_values_for_bagless_addresses(self.data)
        self.version += '_bag'
        self.save()
        print("The adres dataset is now enriched with BAG data.")


    def enrich_with_personen_features(self, personen):
        """Add aggregated features relating to persons to the address dataframe. Uses the personen dataframe as input."""

        # Create simple handle to the adres data.
        adres = self.data

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

        # Find the matching address ids between the adres df and the personen df.
        adres_ids = adres.adres_id
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
        adres['aantal_personen'] = 0
        adres['aantal_vertrokken_personen'] = -1
        adres['aantal_overleden_personen'] = -1
        adres['aantal_niet_uitgeschrevenen'] = -1
        adres['leegstand'] = True
        adres['leeftijd_jongste_persoon'] = -1.
        adres['leeftijd_oudste_persoon'] = -1.
        adres['aantal_kinderen'] = 0
        adres['percentage_kinderen'] = -1.
        adres['aantal_mannen'] = 0
        adres['percentage_mannen'] = -1.
        adres['gemiddelde_leeftijd'] = -1.
        adres['stdev_leeftijd'] = -1.
        adres['aantal_achternamen'] = 0
        adres['percentage_achternamen'] = -1.
        for i in range(1,8):
            adres[f'gezinsverhouding_{i}'] = 0
            adres[f'percentage_gezinsverhouding_{i}'] = 0.
        print("Now looping over all rows in the adres dataframe in order to add person information...")
        for i in adres.index:
            if i % 1000 == 0:
                print(i)
            row = adres.iloc[i]
            adres_id = row['adres_id']
            try:
                # Get the inhabitants for the current address.
                inhab_locs = inhabitant_locs[adres_id].keys()
                inhab = personen.loc[inhab_locs]

                # Check whether any registered inhabitants have left Amsterdam or have passed away.
                aantal_vertrokken_personen = sum(inhab["vertrekdatum_adam"].notnull())
                aantal_overleden_personen = sum(inhab["overlijdensdatum"].notnull())
                aantal_niet_uitgeschrevenen = len(inhab[inhab["vertrekdatum_adam"].notnull() | inhab["overlijdensdatum"].notnull()])
                adres['aantal_vertrokken_personen'] = aantal_vertrokken_personen
                adres['aantal_overleden_personen'] = aantal_overleden_personen
                adres['aantal_niet_uitgeschrevenen'] = aantal_niet_uitgeschrevenen
                # If there are more inhabitants than people that are incorrectly still registered, then there is no 'leegstand'.
                if len(inhab) > aantal_niet_uitgeschrevenen:
                    adres['leegstand'] = False

                # Totaal aantal personen (int).
                aantal_personen = len(inhab)
                adres.at[i, 'aantal_personen'] = aantal_personen

                # Leeftijd jongste persoon (float).
                leeftijd_jongste_persoon = min(inhab['leeftijd'])
                adres.at[i, 'leeftijd_jongste_persoon'] = leeftijd_jongste_persoon

                # Leeftijd oudste persoon (float).
                leeftijd_oudste_persoon = max(inhab['leeftijd'])
                adres.at[i, 'leeftijd_oudste_persoon'] = leeftijd_oudste_persoon

                # Aantal kinderen ingeschreven op adres (int/float).
                aantal_kinderen = sum(inhab['leeftijd'] < 18)
                adres.at[i, 'aantal_kinderen'] = aantal_kinderen
                adres.at[i, 'percentage_kinderen'] = aantal_kinderen / aantal_personen

                # Aantal mannen (int/float).
                aantal_mannen = sum(inhab.geslacht == 'M')
                adres.at[i, 'aantal_mannen'] = aantal_mannen
                adres.at[i, 'percentage_mannen'] = aantal_mannen / aantal_personen

                # Gemiddelde leeftijd (float).
                gemiddelde_leeftijd = inhab.leeftijd.mean()
                adres.at[i, 'gemiddelde_leeftijd'] = gemiddelde_leeftijd

                # Standardeviatie van leeftijd (float). Set to 0 when the sample size is 1.
                stdev_leeftijd = inhab.leeftijd.std()
                adres.at[i, 'stdev_leeftijd'] = stdev_leeftijd if aantal_personen > 1 else 0

                # Aantal verschillende achternamen (int/float).
                aantal_achternamen = inhab.naam.nunique()
                adres.at[i, 'aantal_achternamen'] = aantal_achternamen
                adres.at[i, 'percentage_achternamen'] = aantal_achternamen / aantal_personen

                # Gezinsverhouding (frequency count per klasse) (int/float).
                gezinsverhouding = inhab.gezinsverhouding.value_counts()
                for key in gezinsverhouding.keys():
                    val = gezinsverhouding[key]
                    adres.at[i, f'gezinsverhouding_{key}'] = val
                    adres.at[i, f'percentage_gezinsverhouding_{key}'] = val / aantal_personen

            except (KeyError, ValueError) as e:
                pass

        print("...done!")

        self.data = adres
        self.version += '_personen'
        self.save()
        print("The adres dataset is now enriched with personen data.")


    def add_hotline_features(self, hotline):
        """Add the hotline features to the adres dataframe."""
        # Create a temporary merged df using the adres and hotline dataframes.
        merge = self.data.merge(hotline, on='wng_id', how='left')
        # Create a group for each adres_id
        adres_groups = merge.groupby(by='adres_id')
        # Count the number of hotline meldingen per group/adres_id.
        # 'id' should be the primary key of hotline df, so it is usable for hotline entry counting.
        hotline_counts = adres_groups['id'].agg(['count'])
        # Rename column
        hotline_counts.columns = ['aantal_hotline_meldingen']
        # Enrich the 'adres' dataframe with the computed hotline counts.
        self.data = self.data.merge(hotline_counts, on='adres_id', how='left')
        self.version += '_hotline'
        self.save()
        print("The adres dataset is now enriched with hotline data.")


class ZakenDataset(MyDataset):
    """Create a dataset for the zaken data."""

    ## Set the class attributes.
    name = 'zaken'
    table_name = 'import_wvs'
    id_column = 'zaak_id'


    def add_categories(self):
        """Add categories to the zaken dataframe."""
        clean.lower_strings(self.data)
        add_column(df=self.data, new_col='categorie', match_col='beh_oms',
                   csv_path=os.path.join(HOME, 'woonfraude/data/aanvulling_beh_oms.csv'))
        self.version += '_categories'
        self.save()


    def filter_categories(self):
        """
        Remove cases (zaken) with categories 'woningkwaliteit' or 'afdeling vergunninen beheer'.
        These cases do not contain reliable samples.
        """
        self.data = self.data[~self.data.categorie.isin(['woningkwaliteit', 'afdeling vergunningen en beheer'])]
        self.version += '_filterCategories'
        self.save()


    def keep_finished_cases(self, stadia):
        """Only keep cases (zaken) that have 100% certainly been finished. Uses stadia dataframe as input."""

        # Create simple handle to the zaken data.
        zaken = self.data

        # Select finished zoeklicht cases.
        zaken['mask'] = zaken.afs_oms == 'zl woning is beschikbaar gekomen'
        zaken['mask'] += zaken.afs_oms == 'zl geen woonfraude'
        zl_zaken = zaken.loc[zaken['mask']]

        # Indicate which stadia are indicative of finished cases.
        stadia['mask'] = stadia.sta_oms == 'rapport naar han'
        stadia['mask'] += stadia.sta_oms == 'bd naar han'

        # Indicate which stadia are from before 2013. Cases linked to these stadia should be
        # disregarded. Before 2013, 'rapport naar han' and 'bd naar han' were used inconsistently.
        timestamp_2013 =  pd.Timestamp('2013-01-01')
        stadia['before_2013'] = stadia.begindatum < timestamp_2013

        # Create groups linking cases to their stadia.
        zaak_groups = stadia.groupby('zaak_id').groups

        # Select all finished cases based on "rapport naar han" and "bd naar han" stadia.
        keep_ids = []
        for zaak_id, stadia_ids in zaak_groups.items():
            zaak_stadia = stadia.loc[stadia_ids]
            if sum(zaak_stadia['mask']) >= 1 and sum(zaak_stadia['before_2013']) == 0:
                keep_ids.append(zaak_id)
        rap_zaken = zaken[zaken.zaak_id.isin(keep_ids)]

        # Combine all finished cases.
        finished_cases = pd.concat([zl_zaken, rap_zaken], sort=True)

        # Remove temporary mask
        finished_cases.drop(columns=['mask'], inplace=True)

        # Print results.
        print(f'Selected {len(finished_cases)} finished cases from a total of {len(zaken)} cases.')

        # Only keep the sleection of finished of cases.
        self.data = finished_cases
        self.version += '_finishedCases'
        self.save()


    def add_binary_label_zaken(self, stadia):
        """Create a binary label defining whether there was woonfraude."""

        # Only set "woonfraude" label to True when the zaken_mask and/or stadia_mask is True.
        self.data['woonfraude'] = False # Set default value to false
        zaken_mask = self.data.loc[self.data['afs_oms'].str.contains('zl woning is beschikbaar gekomen',
                                                             regex=True, flags=re.IGNORECASE) == True]
        stadia_mask = stadia.loc[stadia['sta_oms'].str.contains('rapport naar han', regex=True,
                                                                flags=re.IGNORECASE) == True]
        zaken_ids_zaken = zaken_mask['zaak_id'].tolist()
        zaken_ids_stadia = stadia_mask['zaak_id'].tolist()
        zaken_ids = list(set(zaken_ids_zaken + zaken_ids_stadia))  # Get uniques
        self.data.loc[self.data['zaak_id'].isin(zaken_ids), 'woonfraude'] = True  # Add woonfraude label

        # Add woonfraude class labels.
        # zaken.loc[zaken['zaak_id'].isin(zaken_ids_zaken), 'woonfraude'] = 'zoeklicht'
        # zaken.loc[zaken['zaak_id'].isin(zaken_ids_stadia), 'woonfraude'] = 'rapport_handhaving'

        # Print results
        print(f'Dataframe "zaken": added column "woonfraude" (binary label)')


class StadiaDataset(MyDataset):
    """Create a dataset for the stadia data."""

    # Set the class attributes.
    name = 'stadia'
    table_name = 'import_stadia'
    id_column = 'stadium_id'


    def add_zaak_stadium_ids(self):
        """Add necessary id's to the dataset."""
        self.data['zaak_id'] = self.data['adres_id'].astype(int).astype(str) + '_' + self.data['wvs_nr'].astype(int).astype(str)
        self.data['stadium_id'] = self.data['zaak_id'] + '_' + self.data['sta_nr'].astype(int).astype(str)
        self.version += '_ids'
        self.save()


    def add_labels(self):
        """Add labels to the zaken dataframe."""
        clean.lower_strings(self.data)
        add_column(df=self.data, new_col='label', match_col='sta_oms',
                   csv_path=os.path.join(HOME, 'woonfraude/data/aanvulling_sta_oms.csv'))
        self.version += '_labels'
        self.save()


class PersonenDataset(MyDataset):
    """Create a dataset for the stadia data."""

    # Set the class attributes.
    name = 'personen'
    table_name = 'bwv_personen'
    id_column = 'id'


class BagDataset(MyDataset):
    """Create a dataset for the bag data."""

    # Set the class attributes.
    name = 'bag'
    table_name = 'bag_nummeraanduiding'
    id_column = 'id_nummeraanduiding'

    def bag_fix(self):
        """Apply specific fixes for the BAG dataset."""

        # Merge columns (three copies).
        l_merge = ['_gebiedsgerichtwerken_id', 'indicatie_geconstateerd', 'indicatie_in_onderzoek',
                   '_grootstedelijkgebied_id', 'buurt_id']
        for m in l_merge:
            self.data[m] = self.data[m + '_ligplaats'].combine_first(self.data[m + '_verblijfsobject'])
            self.data[m] = self.data[m].combine_first(self.data[m + '_standplaats'])
            self.data.drop(columns=[m + '_ligplaats', m + '_standplaats', m + '_verblijfsobject'], inplace=True)

        # Merge columns (four copies). (almost always none, except for columns 0 & 3)
        l_merge2 = ['document_mutatie', 'document_nummer', 'begin_geldigheid', 'einde_geldigheid']
        for m in l_merge2:
            self.data[m] = self.data[m + '_nummeraanduiding'].combine_first(self.data[m + '_verblijfsobject'])
            self.data[m] = self.data[m].combine_first(self.data[m + '_standplaats'])
            self.data[m] = self.data[m].combine_first(self.data[m + '_ligplaats'])
            self.data.drop(columns=[m + '_nummeraanduiding', m + '_ligplaats', m + '_standplaats', m + '_verblijfsobject'], inplace=True)

        # Drop columns
        self.data.drop(columns=['_openbare_ruimte_naam_ligplaats','_openbare_ruimte_naam_standplaats', 'mutatie_gebruiker_nummeraanduiding',
                                'mutatie_gebruiker_ligplaats', 'mutatie_gebruiker_standplaats', 'mutatie_gebruiker_verblijfsobject',
                                '_huisnummer_ligplaats', '_huisnummer_standplaats', '_huisletter_ligplaats', '_huisletter_standplaats',
                                '_huisnummer_toevoeging_ligplaats', '_huisnummer_toevoeging_standplaats', '_huisnummer_toevoeging_verblijfsobject',
                                '_huisnummer_verblijfsobject', '_openbare_ruimte_naam_verblijfsobject', 'date_modified_ligplaats',
                                'date_modified_standplaats', 'date_modified_verblijfsobject'],
                                inplace=True)

        # Change dataset version, and save this version of the dataset.
        self.version += '_columnFix'
        self.save()


class HotlineDataset(MyDataset):
    """Create a dataset for the hotline data."""

    # Set the class attributes.
    name = 'hotline'
    table_name = 'bwv_hotline_melding'
    id_column = 'id'


class BbgaDataset(MyDataset):
    """Create a dataset for the BBGA data."""

    # Set the class attributes.
    name = 'bbga'


######################
## Helper functions ##
######################

def add_column(df, new_col, match_col, csv_path, key='lcolumn', val='ncolumn'):
    """Add a new column to dataframe based on the match_column, and the mapping in the csv.

    df: dataframe to be augmented.
    new_col: name of new dataframe column.
    match_col: colum to match with the csv variable 'key'.
    csv_path: path to the csv file which is used for augmentation.
    key: name of column in csv file containing keys.
    val: name of column in csv file containing values.
    """

    # Load csv file.
    df_label = pd.read_csv(csv_path)

    # Transform csv string data to lowercase.
    df_label[key] = df_label[key].str.lower()
    df_label[val] = df_label[val].str.lower()

    # Create a dict mapping: key -> val, based on the csv data.
    label_dict = dict(zip(df_label[key], df_label[val]))

    # Create a new dataframe column. If match_col matches with 'key', set the value to 'val'.
    df[new_col] = df[match_col].apply(lambda x: label_dict.get(x))

    # Print information about performed operation to terminal.
    print(f"Dataframe \"%s\": added column \"%s\"!" % (df.name, new_col))


def save_dataset(data, dataset_name, version):
    """Save a version of the given dataframe."""
    dataset_path = os.path.join(DATA_PATH, f'{dataset_name}_{version}.h5')
    data.to_hdf(path_or_buf=dataset_path, key=dataset_name, mode='w')


def load_dataset(dataset_name, version):
    """Load a version of the dataframe from file. Rename it (pickling removes name)."""
    dataset_path = os.path.join(DATA_PATH, f'{dataset_name}_{version}.h5')
    data = pd.read_hdf(path_or_buf=dataset_path, key=dataset_name, mode='r')
    data.name = dataset_name # Set the dataframe name again after loading (it is lost when saving).
    return data
