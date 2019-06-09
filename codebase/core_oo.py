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


# NOG TOEVOEGEN BIJ OMZETTING NAAR "PREPARE" NOTEBOOK:
#   - STAPPEN OM PIPELINE UIT TE VOEREN
#   - FEATURE SCALER IN PIPELINES
#   - TRAIN/TEST SPLIT
#   - MODEL TRAINING TEST!

# TODO (optioneel): datasets_oo.enrich_with_woning_id download lokaal laten cachen


#######################################################
# Download (or load cached versions of) the datasets.
adresDataset = AdresDataset()
adresDataset.load('download')
adresDataset.extract_leegstand()
adresDataset.enrich_with_woning_id()
# adresDataset.load('download_leegstand_woningId')

zakenDataset = ZakenDataset()
zakenDataset.load('download')
zakenDataset.filter_categories()
zakenDataset.load('download_filterCategories')

stadiaDataset = StadiaDataset()
stadiaDataset.load('download')
stadiaDataset.add_zaak_stadium_ids()
# stadiaDataset.load('download_ids')

personenDataset = PersonenDataset()
personenDataset.load('download')

BagDataset = BagDataset()
BagDataset.load('download')
BagDataset.bag_fix()
# BagDataset.load('download_columnFix')

HotlineDataset = HotlineDataset()
hotlineDataset.load('download')

# Get path to home directory
HOME = str(Path.home())
#######################################################


#######################################################
# Clean zaken dataset
zakenPipeline = Pipeline(steps=[
    ('clean', CleanTransformer(
        id_column=zakenDataset.id_column,
        drop_duplicates=True,
        fix_date_columns=['begindatum','einddatum', 'wzs_update_datumtijd'],
        clean_dates=True,
        lower_string_columns=True,
        add_columns=[{'new_col': 'categorie', 'match_col':'beh_oms',
                      'csv_path': 'f{HOME}/Documents/woonfraude/data/aanvulling_beh_oms.csv'}],
        impute_missing_values=True,
        fillna_columns=False)
    ),
    ('extract', FeatureExtractionTransformer(
        categorical_cols_hot=['afg_code_beh', 'beh_code', 'eigenaar', 'categorie'])
    )
    ])
#######################################################


#######################################################
# Clean stadia dataset
stadiaPipeline = Pipeline(steps=[
    ('clean', CleanTransformer(
        id_column=stadiaDataset.id_column,
        drop_duplicates=True,
        fix_date_columns=['begindatum', 'peildatum', 'einddatum', 'date_created',
                          'date_modified', 'wzs_update_datumtijd'],
        clean_dates=True,
        lower_string_columns=True,
        add_columns=[{'new_col': 'label', 'match_col':'sta_oms',
                      'csv_path': 'f{HOME}/Documents/woonfraude/data/aanvulling_sta_oms.csv'}],
        impute_missing_values=True)
    )])
#######################################################


#######################################################
# Clean personen dataset
personenPipeline = Pipeline(steps=[
    ('clean', CleanTransformer(
        id_column=personenDataset.id_column,
        drop_duplicates=True,
        lower_string_columns=True)
    )])
#######################################################


#######################################################
# Clean BAG dataset
bagPipeline = Pipeline(steps=[
    ('clean', CleanTransformer(
        id_column=bag.id_column,
        drop_duplicates=True,
        drop_columns=bag_remove,
        fix_date_columns=[],
        lower_string_columns=True,
        impute_missing_values=True,
        impute_missing_values_mode=['status_coordinaat_code@bag', 'indicatie_geconstateerd@bag',
                                    'indicatie_in_onderzoek@bag', 'woningvoorraad@bag'],
        fillna_columns={'_huisnummer@bag': 0
                         '_huisletter@bag': 'None',
                         '_openbare_ruimte_naam@bag': 'None',
                         '_huisnummer_toevoeging@bag': 'None',
                         'type_woonobject_omschrijving@bag': 'None',
                         'eigendomsverhouding_id@bag': 'None',
                         'financieringswijze_id@bag': -1,
                         'gebruik_id@bag': -1,
                         'reden_opvoer_id@bag': -1,
                         'status_id@bag': -1,
                         'toegang_id@bag': 'None'})
    ),
    ('extract', FeatureExtractionTransformer(
        categorical_cols_hot=['status_coordinaat_code@bag', 'type_woonobject_omschrijving@bag',
                              'eigendomsverhouding_id@bag', 'financieringswijze_id@bag',
                              'gebruik_id@bag', 'ligging_id@bag', 'reden_opvoer_id@bag',
                              'status_id@bag', 'toegang_id@bag'])
    )
    ])
#######################################################


#######################################################
# Clean hotline dataset
hotlinePipeline = Pipeline(steps=[
    ('clean', CleanTransformer(
        id_column=hotlineDataset.id_column,
        drop_duplicates=True,
        lower_string_columns=True,
        impute_missing_values=True)
    )])
#######################################################


#######################################################
# Clean adres dataset
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

bag_remove = ['einde_geldigheid@bag',               # Only 2 entries in column.
              'verhuurbare_eenheden@bag',           # Only ~2k entries in column.
              'geometrie@bag',                      # Needs a lot of processing before being useful.
              'bron_id@bag',                        # Only 2 entries in column.
              'locatie_ingang_id@bag',              # Only 2 entries in column.
              'reden_afvoer_id@bag',                # Only a few entries in column.
              '_gebiedsgerichtwerken_id@bag',       # Superfluous (gebied).
              '_grootstedelijkgebied_id@bag',       # Superfluous (grootstedelijkgebied).
              'buurt_id@bag',                       # Superfluous (buurt).
              # ONDERSTAANDE 4 KOLOMMEN KONDEN EERDER NIET WEG IVM MATCH MET ADRES DATAFRAME.
              # DEZE MOETEN NU WEL WEG, DAAROM WORDT NU HIER ALLES WEGGEHAALD.
              '_openbare_ruimte_naam@bag',          # Superfluous (straatnaam).
              '_huisnummer@bag',                    # Superfluous (huisnummer).
              '_huisletter@bag',                    # Superfluous (huisletter).
              '_huisnummer_toevoeging@bag',         # Superfluous (huisnummer toevoeging).
              'vervallen@bag',                      # Superfluous (all values in col are equal).
              'mutatie_gebruiker@bag',              # Superfluous (all values in col are equal).
              'document_mutatie@bag',               # Not available at time of signal.
              'date_modified@bag',                  # Not available at time of signal.
              'document_nummer@bag',                # Not needed? (Swaan?)
              'status_coordinaat_omschrijving@bag', # Not needed? (Swaan?)
              'type_woonobject_code@bag',           # Not needed? (Swaan?)
              'id@bag',                             # Not needed.
              'landelijk_id@bag'                    # Not needed.
              ]

# Hier de extract stap weghalen? Deze past waarschijnlijk beter na het combinen v/d datasets.
adresPipeline = Pipeline(steps=[
    ('clean', CleanTransformer(
        id_column=adres.id_column,
        drop_duplicates=True,
        drop_columns=adres_remove + bag_remove,
        fix_date_columns=['hvv_dag_tek', 'max_vestig_dtm', 'wzs_update_datumtijd'],
        lower_string_columns=True,
        impute_missing_values=True,
        fillna_columns={'hsnr': 0, 'sttnaam': 'None', 'hsltr': 'None', 'toev': 'None'})
    ),
    ('extract', FeatureExtractionTransformer(
        categorical_cols_hot=['toev', 'pvh_omschr', 'sbw_omschr', 'sbv_omschr'],
        ))
    ])
#######################################################


#######################################################
# Combine datasets
adresDataset.enrich_with_bag(bagDataset.data)
adresDataset.enrich_with_personen_features(personenDataset.data)
adresDataset.add_hotline_features(hotlineDataset.data)

# Remove adres_id, since this is not a feature we want our algorihtm to try and learn from.
adresDataset.data.drop(columns='adres_id', inplace=True)
#######################################################


#######################################################
# Scale features?
#######################################################