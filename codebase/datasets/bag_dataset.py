####################################################################################################
"""
bag_dataset.py

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
import datasets

# Define HOME and DATA_PATH on a global level.
HOME = Path.home()  # Home path for old VAO.
# USERNAME = os.path.basename(HOME)
# HOME = os.path.join('/data', USERNAME)  # Set home for new VAO.
DATA_PATH = os.path.join(HOME, 'Documents/woonfraude/data/')


######################
## BagDataset class ##
######################

class BagDataset(datasets.MyDataset):
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