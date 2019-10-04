####################################################################################################
"""
personen_dataset.py

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


###########################
## PersonenDataset class ##
###########################

class PersonenDataset(datasets.MyDataset):
    """Create a dataset for the stadia data."""

    # Set the class attributes.
    name = 'personen'
    table_name = 'bwv_personen'
    id_column = 'id'