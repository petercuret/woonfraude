####################################################################################################
"""
hotline_dataset.py

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


##########################
## HotlineDataset class ##
##########################

class HotlineDataset(datasets.MyDataset):
    """Create a dataset for the hotline data."""

    # Set the class attributes.
    name = 'hotline'
    table_name = 'bwv_hotline_melding'
    id_column = 'id'