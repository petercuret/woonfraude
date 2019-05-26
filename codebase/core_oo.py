"""
core_oo.py

This script implements several high-level functions, as well as data collection (download/store)
and data saving functions.

Written by Swaan Dekkers & Thomas Jongstra
"""


# Import statements
from pathlib import Path
import pandas.io.sql as sqlio
import pandas as pd
import psycopg2
import pickle
import time

# Import own modules
import config  # Load local passwords (config.py file expected in same folder as this file).
import datasets_oo
import clean_oo
import extract_features_oo
import combine_oo


adres = AdresDataset.download()
zaken = ZakenDataset.download()
stadia = StadiaDataset.download()
personen = PersonenDataset.download()
bag = BagDataset.download()