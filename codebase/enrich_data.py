"""
enrich_data.py

This script aims to take the cleaned BWV data, and enrich it with up-to-date BAG data.
In this process, all BWV entries that cannot be coupled with new BAG data are removed.
After running this script, the resulting data should be ready for feature extraction.

Input: cleaned BWV data (~48k entries @ 2018-11-21).
Output: enriched BWV data, i.e. coupled with up-to-date BAG data (~38k entries @ 2018-11-21).

Written by Swaan Dekkers & Thomas Jongstra
"""

# Source this script from collect_data_and_make_model.ipynb.
# An example of the needed resulting data format can be found in data/base/df_0.pkl
# BAG data for all addresses in Amsterdam is saved in df_adres_cleaned.pkl & df_adres_cleaned.csv