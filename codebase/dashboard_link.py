"""
dashboard_link.py

This script provides the following functions to the dashboard,
hence creating a link between the codebase and the dashboard:

- Creating a selection of the most recent ICTU signals.
- Loading a pre-trained prediction model.
- Performing inference on a list of ICTU signals, using a loaded pre-trained model.

Written by Swaan Dekkers & Thomas Jongstra
"""

# Import public modules.
import pickle
import copy
import sys
import os

# Add the parent paths to sys.path, so our own modules on the root dir can also be imported.
SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PARENT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, os.path.pardir))
CODEBASE_PATH = os.path.abspath(os.path.join(PARENT_PATH, 'codebase'))
sys.path.append(PARENT_PATH)
sys.path.append(CODEBASE_PATH)

# Import own modules.
from datasets_oo import *


#######################################################
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
    model_path = os.path.join(os.path.join(PARENT_PATH, 'data'), 'best_random_forest_classifier_temp.pickle')
    model = pickle.load(open(model_path, "rb"))
    return model



def get_recent_signals(zakenDataset, n=100):
    """Create a list the n most recent ICTU signals from our data."""
    signals = zakenDataset.data.sample(100)  # INSTEAD OF PICKING THE MOST RECENT SIGNALS, WE TEMPORARILY RANDOMLY SAMPLE THEM FOR OUR MOCK UP!
    signals.drop(columns=['woonfraude'], inplace=True)  # For the final list of signals, this step should not be necessary either.
    return signals


def create_signals_predictions(model, signals):
    """Create predictions for signals using a given model."""

    # Remove adres_id column (should not be used for predictions).
    signals.drop(columns=['adres_id'], inplace=True)
    # Remove non-numeric columns.
    signals = signals._get_numeric_data()
    # Remove columns containing only NaN values.
    signals.drop(columns=['hoofdadres', 'begin_geldigheid'], inplace=True)
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