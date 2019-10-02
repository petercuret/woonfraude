####################################################################################################
"""
bbga_dataset.py

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
HOME = os.path.abspath('E:\\Jasmine')
DATA_PATH = os.path.abspath('E:\\Jasmine\\woonfraude\\data')


#######################
## BbgaDataset class ##
#######################

class BbgaDataset(datasets.MyDataset):
    """Create a dataset for the BBGA data."""

    # Set the class attributes.
    name = 'bbga'