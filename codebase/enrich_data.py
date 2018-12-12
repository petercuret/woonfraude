"""
enrich_data.py

This script aims to take the cleaned BWV data, and enrich it with up-to-date BAG data.
In this process, all BWV entries that cannot be coupled with new BAG data are removed.
After running this script, the resulting data should be ready for feature extraction.

Input: cleaned BWV data (~48k entries @ 2018-11-21).
Output: enriched BWV data, i.e. coupled with up-to-date BAG data (~38k entries @ 2018-11-21)
        unenriched BWV data (~10k entries, no match found with BAG code)

Written by Swaan Dekkers & Thomas Jongstra
"""

# Source this script from collect_data_and_make_model.ipynb.
# An example of the needed resulting data format can be found in data/base/df_0.pkl
# BAG data for all addresses in Amsterdam is saved in df_adres_cleaned.pkl & df_adres_cleaned.csv

from pathlib import Path
import clean_bwv

def main():
    """Add BAG data to cleaned BWV data."""

    # Load pre-cleaned adres/zaken/stadia tables.
    adres, zaken, stadia = clean_bwv.load_dfs(3)

    # Load BAG data
    bag_path = Path("E:/woonfraude/data/base/csv/ALL.csv")
    df = pd.read_csv(bag_path, sep=',', encoding="latin-1")

    # Pre-process adres data for join with BAG data.
    adres['landelijk_bag'] = pd.to_numeric(adres['landelijk_bag'],
                                           errors='coerce', downcast='integer')

    # Join adres data to bag data.
    adres = adres.merge(bag, on='landelijk_bag', how='left')

    # Save data to new pickle files.
    clean_bwv.save_dfs(adres, zaken, stadia, 4)

if __name__ == "__main__":
    main()