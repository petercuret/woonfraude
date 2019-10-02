####################################################################################################
"""
zaken_dataset.py

This module implements several classes to perform dataset-specific downloading, saving and
data-transformation operations.

Written by Swaan Dekkers & Thomas Jongstra
"""
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
import clean, datasets

# Define HOME and DATA_PATH on a global level.
HOME = os.path.abspath('E:\\Jasmine')
DATA_PATH = os.path.abspath('E:\\Jasmine\\woonfraude\\data')


########################
## ZakenDataset class ##
########################

class ZakenDataset(datasets.MyDataset):
    """Create a dataset for the zaken data."""

    ## Set the class attributes.
    name = 'zaken'
    table_name = 'import_wvs'
    id_column = 'zaak_id'


    def add_categories(self):
        """Add categories to the zaken dataframe."""
        clean.lower_strings(self.data)
        datasets.add_column(df=self.data, new_col='categorie', match_col='beh_oms',
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
