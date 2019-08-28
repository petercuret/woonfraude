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
import os
import sys

# Add the parent paths to sys.path, so our own modules on the root dir can also be imported.
SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PARENT_PATH = os.path.join(SCRIPT_DIR, os.path.pardir)
sys.path.append(PARENT_PATH)


def get_recent_signals(n=100):
    """Create a list the n most recent ICTU signals from our data."""
    pass


def load_pre_trained_model(path):
    """
    Load a pre-trained machine learning model, which can calculate the statistical
    chance of housing fraud for a list of addresses.
    """
    pass


def get_recent_meldingen_predictions():
    df = get_recent_signals()
    model = load_pre_trained_model('model.pickle')
    predictions = model.transform(df)  # Check if this works
    df['woonfraude_predicted'] = predictions  # Check if this works. We probably should maps "predictions" to a Pandas Series to get this to work.