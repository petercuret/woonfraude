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


#######################
## BbgaDataset class ##
#######################

class BbgaDataset(datasets.MyDataset):
    """Create a dataset for the BBGA data."""

    # Set the class attributes.
    name = 'bbga'