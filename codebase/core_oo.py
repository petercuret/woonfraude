"""
core_oo.py

This script implements several high-level functions, as well as data collection (download/store)
and data saving functions.

Written by Swaan Dekkers & Thomas Jongstra
"""


# Import statements
from sklearn.pipeline import Pipeline
from pathlib import Path
import pandas.io.sql as sqlio
import pandas as pd
import psycopg2
import pickle
import time

# Import own modules
import config  # Load local passwords (config.py file expected in same folder as this file).
from datasets_oo import *
from clean_oo import CleanTransformer
from  extract_features_oo import FeatureExtractionTransformer
import combine_oo


# NIET VERGETEN!!!!!!!!!!!!
# adres_id moet verwijderd worden voordat learning plaatsvindt!
# Anders kan direct op het id geleerd worden.


# Download (or load cached versions of) the datasets.
# zaken = ZakenDataset.load('download')
# stadia = StadiaDataset.load('download')
# personen = PersonenDataset.load('download')
# bag = BagDataset.load('download')


adres_remove = [# Remove because cols do not exists when melding is received
                    'wzs_update_datumtijd',
                    # Remove because cols do not add extra information.
                    'kmrs',
                    'straatcode',
                    'xref',
                    'yref',
                    'postcode',
                    'wzs_buurtcode_os_2015',
                    'wzs_buurtcombinatiecode_os_2015',
                    'wzs_stadsdeelcode_os_2015',
                    'sttnaam',
                    'hvv_dag_tek', # Empty column
                    'max_vestig_dtm', # Empty column
                    'wzs_22gebiedencode_os_2015', # Empty column
                    'wzs_22gebiedennaam_os_2015', # Empty column
                    'sdl_naam',
                    'pvh_cd',
                    'sbv_code',
                    'sbw_code',
                    'wzs_wijze_verrijking_geo',
                    'wzs_22gebiedencode_2015',
                    'brt_naam',
                    'wzs_buurtnaam_os_2015',
                    'wzs_buurtcombinatienaam_os_2015',
                    'wzs_rayonnaam_os_2015',
                    'wzs_rayoncode_os_2015',
                    'wzs_stadsdeelnaam_os_2015',
                    'wzs_alternatieve_buurtennaam_os_2015',
                    'wzs_alternatieve_buurtencode_os_2015',
                    'hsltr',
                    'wzs_geom',
                    'brt_code',
                    'brtcombi_code',
                    'brtcombi_naam',
                    'sdl_code',
                    'wzs_22gebiedennaam_2015',
                    'wzs_id',
                    'a_dam_bag',
                    'landelijk_bag']

# adres = AdresDataset.load('download')
# adresPipeline = Pipeline(steps=[
#     ('clean', CleanTransformer(
#         id_column=adres.id_column,
#         drop_duplicates=True,
#         drop_columns=adres_remove,
#         fix_date_columns=['hvv_dag_tek', 'max_vestig_dtm', 'wzs_update_datumtijd'],
#         lower_string_columns=True,
#         impute_missing_values=True,
#         fillna_columns=True)
#     ),
#     ('extract', FeatureExtractionTransformer(
#         text_features_cols_hot=[],
#         categorical_cols_hot=['toev', 'pvh_omschr', 'sbw_omschr', 'sbv_omschr'],
#         categorical_cols_no_hot=[],
#         ))
#     ])