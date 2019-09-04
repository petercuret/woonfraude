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
import sys
import os

# Add the parent paths to sys.path, so our own modules on the root dir can also be imported.
SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PARENT_PATH = os.path.join(SCRIPT_DIR, os.path.pardir)
sys.path.append(PARENT_PATH)

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
    model = pickle.load(open("best_random_forest_classifier_temp.pickle", "rb"))
    return model


def get_recent_signals(zakenDataset, n=100):
    """Create a list the n most recent ICTU signals from our data."""
    recent_signals = zakenDataset.data.sample(100)  # INSTEAD OF PICKING THE MOST RECENT SIGNALS, WE TEMPORARILY RANDOMLY SAMPLE THEM FOR OUR MOCK UP!
    recent_signals.drop(columns=['woonfraude'], inplace=True)
    return recent_signals


def get_recent_meldingen_predictions():
    zakenDataset = load_data()
    model = load_pre_trained_model()
    recent_signals = get_recent_signals(zakenDataset)
    model = load_pre_trained_model()
    predictions = model.transform(df)
    df['woonfraude_predicted'] = predictions  # Check if this works. We probably should maps "predictions" to a Pandas Series to get this to work.