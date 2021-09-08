import os
from typing import Dict, List, Union, Optional

import pandas as pd


def get_df_column_details(df: pd.DataFrame) -> pd.DataFrame:
    col_list = list(df.columns)
    n_rows = df.shape[0]
    df_details = pd.DataFrame(
        {
            "feature": [col for col in col_list],
            "unique_vals": [df[col].nunique() for col in col_list],
            "pct_unique": [
                round(100 * df[col].nunique() / n_rows, 4) for col in col_list
            ],
            "null_vals": [df[col].isnull().sum() for col in col_list],
            "pct_null": [
                round(100 * df[col].isnull().sum() / n_rows, 4) for col in col_list
            ],
        }
    )
    df_details = df_details.sort_values(by="unique_vals")
    df_details = df_details.reset_index(drop=True)
    return df_details


def get_df_of_data_portal_data(
    file_name: str,
    url: str,
    raw_file_path: Union[str, None] = None,
    force_repull: bool = False,
) -> pd.DataFrame:
    if raw_file_path is None:
        raw_file_dir = os.path.join(
            os.path.expanduser("~"), "projects", "cook_county_real_estate", "data_raw"
        )
        raw_file_path = os.path.join(raw_file_dir, file_name)
    else:
        raw_file_dir = os.path.dirname(raw_file_path)
    os.makedirs(raw_file_dir, exist_ok=True)

    if not os.path.isfile(raw_file_path) or force_repull:
        df = pd.read_csv(url, low_memory=False)
        df.to_parquet(raw_file_path, compression="gzip")
    else:
        df = pd.read_parquet(raw_file_path)
    return df
