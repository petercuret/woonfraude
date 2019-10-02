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
HOME = os.path.abspath('E:\\Jasmine')
DATA_PATH = os.path.abspath('E:\\Jasmine\\woonfraude\\data')


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