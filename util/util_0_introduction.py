# The functions used here are based on code written in "0 Introduction.ipynb",
#  look at the notebook for extra explanation.

import pandas as pd
import openpyxl  # Dependency of pandas that's not automatically installed


def get_preprocessed_dataset():
    # Dataset
    df = pd.read_excel('headline-data/Dataverwerking.xlsx', sheet_name='Verwerking')

    # Column names
    df.columns = [x.strip() for x in df.columns]

    # Test ID's
    for i in range(1, len(df)):
        if pd.isna(df.loc[i, 'Test']):
            df.loc[i, 'Test'] = df.loc[i - 1, 'Test']

    return df