####################################################################################################
"""
stadia_dataset.py

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
HOME = Path.home()  # Home path for old VAO.
# USERNAME = os.path.basename(HOME)
# HOME = os.path.join('/data', USERNAME)  # Set home for new VAO.
DATA_PATH = os.path.join(HOME, 'Documents/woonfraude/data/')


#########################
## StadiaDataset class ##
#########################

class StadiaDataset(datasets.MyDataset):
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
        datasets.add_column(df=self.data, new_col='label', match_col='sta_oms',
                   csv_path=os.path.join(HOME, 'Documents/woonfraude/data/aanvulling_sta_oms.csv'))
        self.version += '_labels'
        self.save()