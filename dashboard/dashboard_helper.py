####################################################################################################
# dashboard_helper.py                                                                              #
#                                                                                                  #
# This script provides the following functions to the dashboard,                                   #
# hence creating a link between the codebase and the dashboard:                                    #
#                                                                                                  #
# - Creating a selection of the most recent ICTU signals.                                          #
# - Loading a pre-trained prediction model.                                                        #
# - Performing inference on a list of ICTU signals, using a loaded pre-trained model.              #
#                                                                                                  #
# Written by Swaan Dekkers & Thomas Jongstra                                                       #
####################################################################################################

##################
## Manage Paths ##
##################

# Load environment variables.
MAIN_PATH = os.getenv("WOONFRAUDE_PATH")
DATA_PATH = os.getenv("WOONFRAUDE_DATA_PATH")
CODEBASE_PATH = os.path.abspath(os.path.join(MAIN_PATH, 'codebase'))
NOTEBOOK_PATH = os.path.abspath(os.path.join(MAIN_PATH, 'notebooks'))
DASHBOARD_PATH = os.path.abspath(os.path.join(MAIN_PATH, 'dashboard'))

# Add system paths.
sys.path.insert(1, CODEBASE_PATH)


#############
## Imports ##
#############

from sqlalchemy import create_engine
import papermill as pm
import datetime
import pickle
import copy
import sys
import os

# Add the parent paths to sys.path, so our own modules on the root dir can also be imported.
# SCRIPT_PATH = os.path.abspath(__file__)
# SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
# WOONFRAUDE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, os.path.pardir))
# CODEBASE_PATH = os.path.abspath(os.path.join(WOONFRAUDE_PATH, 'codebase'))
# NOTEBOOK_PATH = os.path.abspath(os.path.join(WOONFRAUDE_PATH, 'notebooks'))
# sys.path.append(WOONFRAUDE_PATH)
# sys.path.append(CODEBASE_PATH)
# sys.path.append(NOTEBOOK_PATH)

# Import own modules.
from datasets import *

# Import config file.
import config

################################
## Dashboard helper functions ##
################################

def load_data():
    """Load the final pre-processed and enriched version of the zaken dataset."""
    zakenDataset = ZakenDataset()
    zakenDataset.load('final')
    return zakenDataset


def load_pre_trained_model():
    """
    Load a pre-trained machine learning model, which can calculate the statistical
    chance of housing fraud for a list of addresses.
    """
    model_path = os.path.join(WOONFRAUDE_DATA_PATH, 'best_random_forest_regressor_temp.pickle')
    model = pickle.load(open(model_path, "rb"))
    return model


def get_recent_signals(zakenDataset, n=100):
    """Create a list the n most recent ICTU signals from our data."""
    signals = zakenDataset.data.sample(100)  # INSTEAD OF PICKING THE MOST RECENT SIGNALS, WE TEMPORARILY RANDOMLY SAMPLE THEM FOR OUR MOCK UP!
    signals.drop(columns=['woonfraude'], inplace=True)  # For the final list of signals, this step should not be necessary either.
    return signals


def create_signals_predictions(model, signals):
    """Create predictions for signals using a given model."""

    # Remove adres_id column, if it is there (should not be used for predictions).
    try:
        signals.drop(columns=['adres_id'], inplace=True)
    except:
        pass

    # Remove non-numeric columns.
    signals = signals._get_numeric_data()

    # Try to remove columns containing only NaN values.
    try:
        signals.drop(columns=['hoofdadres', 'begin_geldigheid'], inplace=True)
    except:
        pass

    # Create predictions.
    predictions = model.predict(signals)
    return predictions


def process_recent_signals():
    """Create a list of recent signals and their computed fraud predictions."""
    zakenDataset = load_data()
    model = load_pre_trained_model()
    recent_signals = get_recent_signals(zakenDataset)
    recent_signals_for_predictions = copy.deepcopy(recent_signals)
    model = load_pre_trained_model()
    predictions = create_signals_predictions(model, recent_signals_for_predictions)
    recent_signals['woonfraude'] = predictions
    recent_signals['fraude_kans'] = recent_signals['woonfraude'].astype(int)  # Temporarily create a fraude_kans column to be compatible with the dashboard.
    return recent_signals



def process_for_tableau():
    """
    !!! TEMPORARY SOLUTION FOR PILOT !!!
    Create a list of prediction values for ALL data points, and write the results to a databsae.
    Tableau will be able to look at the results in this database. However, only a small selection
    of the data points will be used by Tableau (the open cases).
    The predictions for already closed cases are not needed (and since the model is trained
    on these cases, the predictions might make the model look better than it actually is).
    Nevertheless, since it is hard to make a selection of all open cases beforehand,
    and since Tableau only uses the prediction values of the open cases, we made the
    (quick-and-dirty) decision to temporarily generate predictions for ALL cases,
    so we can start with pilot :)
    """

    # Create a folder structure for the papermill output.
    now = datetime.datetime.now()
    day_string = f'{str(now)[0:10]}'
    output_folder = os.path.abspath(os.path.join(NOTEBOOK_PATH, 'papermill_output'))
    output_folder_run = os.path.abspath(os.path.join(output_folder, day_string))
    if not Path(output_folder).exists():
        os.mkdir(f'{output_folder}')
    if not Path(output_folder_run).exists():
        os.mkdir(f'{output_folder_run}')

    # Run data preparation step (master_prepare.ipynb) using Papermill.
    _ = pm.execute_notebook(os.path.abspath(os.path.join(NOTEBOOK_PATH, 'master_prepare_tableau.ipynb')),
                            f'{output_folder_run}/master_prepare - output.ipynb')

    # Load data & model.
    zakenDataset = load_data()
    model = load_pre_trained_model()

    # Get list of columns expected by model. Remove any columns in the data that do not match this list.
    data = copy.deepcopy(zakenDataset.data)
    data = data[model.feature_names]

    # Create predictions.
    predictions = create_signals_predictions(model, data)
    zakenDataset.data['woonfraude'] = predictions

    # Convert predictions to a model fitting the database.
    zakenDataset.data['wvs_nr'] = zakenDataset.data.zaak_id.apply(lambda x: x.split('_')[1])
    zakenDataset.data.rename(columns={'woonfraude': 'fraud_prediction'}, inplace=True)
    predictions_tableau =  zakenDataset.data[['adres_id', 'wvs_nr', 'fraud_prediction']]

    # Create a database engine.
    engine = create_engine(f'postgresql+psycopg2://{config.USER_2}:{config.PASSWORD_2}@{config.HOST_2}:{config.PORT_2}/{config.DB_2}')

    # Commit predictions to the database, to be used by Tableau.
    predictions_tableau.to_sql('bwv_jasmine', engine, schema='public', index=False, if_exists='replace')
    # predictions_tableau.to_sql('bwv_jasmine_history', engine, schema='public', index=False, if_exists='append')