import argparse

import numpy as np
import pandas as pd
import requests

def apply_drift(df):
    drifted_df = df.copy()

    drift_values = np.random.randint(-1, 8, size=len(drifted_df))
    drifted_df['age'] = drifted_df['age'] + drift_values
    drifted_df['age'] = drifted_df['age'].clip(lower=0)

    drifted_df['hours-per-week'] = np.round(drifted_df['hours-per-week'] * 0.73).astype('int64')

    mask = (
            (drifted_df['race'] == 'White') &
            (np.random.rand(len(drifted_df)) < 0.13)
    )
    drifted_df.loc[mask, 'race'] = 'Black'

    mask = (
            drifted_df['marital-status'].isin(['Married-civ-spouse', 'Never-married']) &
            (np.random.rand(len(drifted_df)) < 0.17)
    )
    drifted_df.loc[mask, 'marital-status'] = 'Divorced'

    mask = np.random.rand(len(drifted_df)) < 0.73
    drifted_df.loc[mask, 'sex'] = drifted_df.loc[mask, 'sex'].apply(
        lambda x: 'Female' if x == 'Male' else 'Male'
    )

    return drifted_df

def main():
    parser = argparse.ArgumentParser(description="Generate prediction for Flask server.")
    parser.add_argument(
        "number",
        type=int,
        help="Prediction samples number."
    )
    parser.add_argument(
        "-l",
        "--labeled",
        action="store_true",
        help="Add labels for predictions."
    )

    args = parser.parse_args()

    df = pd.read_csv('../app/data/adult.csv').sample(args.number)

    if not args.labeled:
        df = df.drop(['salary'], axis=1)
    drifted_df = apply_drift(df)
    for index, row in drifted_df.iterrows():
        data = row.to_json()
        requests.post('http://127.0.0.1:8000/api/predict',
                            headers={'Content-Type': 'application/json', 'accept': 'application/json'}, data=data)

if __name__ == '__main__':
    main()