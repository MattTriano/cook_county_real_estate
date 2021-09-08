import os
from typing import Dict, List

import pandas as pd

def get_df_column_details(df: pd.DataFrame) -> pd.DataFrame:
    col_list = list(df.columns)
    n_rows = df.shape[0]
    df_details = pd.DataFrame({
        "feature": [col for col in col_list],
        "unique_vals": [df[col].nunique() for col in col_list],
        "pct_unique": [round(100 * df[col].nunique()/n_rows, 4) for col in col_list],
        "null_vals": [df[col].isnull().sum() for col in col_list],
        "pct_null": [round(100 * df[col].isnull().sum() / n_rows, 4) for col in col_list]    
    })
    df_details = df_details.sort_values(by="unique_vals")
    df_details = df_details.reset_index(drop=True)
    return df_details