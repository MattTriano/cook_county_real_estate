import os
from typing import Dict, List, Union, Optional

import pandas as pd
import geopandas as gpd


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


def get_gdf_of_data_portal_data(
    file_name: str,
    url: str,
    raw_file_path: Union[str, None] = None,
    force_repull: bool = False,
) -> gpd.GeoDataFrame:
    if raw_file_path is None:
        raw_file_dir = os.path.join(
            os.path.expanduser("~"), "projects", "cook_county_real_estate", "data_raw"
        )
        raw_file_path = os.path.join(raw_file_dir, file_name)
    else:
        raw_file_dir = os.path.dirname(raw_file_path)
    os.makedirs(raw_file_dir, exist_ok=True)
    if not os.path.isfile(raw_file_path) or force_repull:
        gdf = gpd.read_file(url)
        gdf.to_parquet(raw_file_path, compression="gzip")
    else:
        gdf = gpd.read_parquet(raw_file_path)
    return gdf


def get_raw_cc_real_estate_sales_data(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> pd.DataFrame:
    df = get_df_of_data_portal_data(
        file_name="cc_real_estate_sales.parquet.gzip",
        url="https://datacatalog.cookcountyil.gov/api/views/93st-4bxh/rows.csv?accessType=DOWNLOAD",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return df


def clean_cc_real_estate_sales_arms_length_col(df: pd.DataFrame) -> pd.DataFrame:
    if 9 in df["Arms' length"].unique():
        arms_length_map = {0: "no", 1: "yes", 9: "unknown"}
        df["Arms' length"] = df["Arms' length"].map(arms_length_map)
    df["Arms' length"] = df["Arms' length"].astype("category")
    return df


def clean_cc_real_estate_sales_deed_type_col(df: pd.DataFrame) -> pd.DataFrame:
    if "Warranty" not in df["Deed type"].unique():
        deed_type_map = {
            "W": "Warranty",
            "O": "Other",
            "o": "Other",
            "T": "Trustee",
            "Y": "Trustee",
        }
        df["Deed type"] = df["Deed type"].map(deed_type_map)
    df["Deed type"] = df["Deed type"].astype("category")
    return df


def clean_cc_real_estate_sales_date_cols(
    df: pd.DataFrame, date_cols: Union[List, None] = None
) -> pd.DataFrame:
    if date_cols is None:
        date_cols = df.head(2).filter(regex="[Dd][Aa][Tt][Ee]$").columns
    for date_col in date_cols:
        df[date_col] = pd.to_datetime(
            df[date_col], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
        )
    return df


def clean_cc_real_estate_sales_data(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> pd.DataFrame:
    cc_sales_df = get_raw_cc_real_estate_sales_data(
        raw_file_path=raw_file_path, force_repull=force_repull
    )
    cc_sales_df = clean_cc_real_estate_sales_arms_length_col(df=cc_sales_df)
    cc_sales_df = clean_cc_real_estate_sales_deed_type_col(df=cc_sales_df)
    cc_sales_df = clean_cc_real_estate_sales_date_cols(df=cc_sales_df)
    cc_sales_df = cc_sales_df.convert_dtypes()
    return cc_sales_df


def get_clean_cc_real_estate_sales_data(
    clean_file_path: Union[str, bool] = None,
    raw_file_path: Union[str, bool] = None,
    force_reclean: bool = False,
    force_repull: bool = False,
):
    if clean_file_path is None:
        file_dir = os.path.join(
            os.path.expanduser("~"), "projects", "cook_county_real_estate", "data_clean"
        )
        clean_file_path = os.path.join(file_dir, "cc_real_estate_sales.parquet.gzip")
    if os.path.isfile(clean_file_path) and not force_reclean and not force_repull:
        df = pd.read_parquet(file_path)
        return df
    elif force_reclean and not force_repull:
        df = clean_cc_real_estate_sales_data(raw_file_path=raw_file_path)
    else:
        df = clean_cc_real_estate_sales_data(
            raw_file_path=raw_file_path, force_repull=force_repull
        )
    df.to_parquet(clean_file_path, compression="gzip")
    return df


def get_raw_cc_residential_neighborhood_geodata(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    gdf = get_gdf_of_data_portal_data(
        file_name="cc_residential_neighborhood_boundaries.parquet.gzip",
        url="https://datacatalog.cookcountyil.gov/api/geospatial/wyzt-dzf8?method=export&format=Shapefile",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return gdf


def clean_cc_residential_neighborhood_geodata(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    gdf = get_raw_cc_residential_neighborhood_geodata(
        raw_file_path=raw_file_path, force_repull=force_repull
    )
    gdf["triad_code"] = gdf["triad_code"].astype("category")
    gdf["triad_name"] = gdf["triad_name"].astype("category")
    gdf["township_c"] = gdf["township_c"].astype("category")
    gdf["township_n"] = gdf["township_n"].astype("category")
    gdf["nbhd"] = gdf["nbhd"].astype("category")
    gdf = gdf.convert_dtypes()
    return gdf
