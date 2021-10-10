from typing import Dict, List, Union, Optional

import pandas as pd

from utils import get_df_of_data_portal_data


def get_raw_cc_property_locations_data(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> pd.DataFrame:
    df = get_df_of_data_portal_data(
        file_name="cc_property_locations.parquet.gzip",
        url="https://datacatalog.cookcountyil.gov/api/views/c49d-89sn/rows.csv?accessType=DOWNLOAD",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return df


def clean_cc_property_locations_ohare_noise_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Indicator for properties near O'Hare airport flight paths."""
    df["ohare_noise"] = df["ohare_noise"].astype("boolean")
    return df


def clean_cc_property_locations_floodplain_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Indicator for properties within FEMA-defined floodplains."""
    df["floodplain"] = df["floodplain"].astype("boolean")
    return df


def clean_cc_property_locations_withinmr100_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Indicator for properties within 100 feet of a major road.
    Roads taken from OpenStreetMap."""
    df["withinmr100"] = df["withinmr100"].astype("boolean")
    return df


def clean_cc_property_locations_withinmr101300_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Indicator for properties between 101-300 feet of a major road.
    Roads taken from OpenStreetMap."""
    df["withinmr101300"] = df["withinmr101300"].astype("boolean")
    return df


def clean_cc_property_locations_drop_cols(
    df: pd.DataFrame,
) -> pd.DataFrame:
    drop_cols = [
        "indicator_has_latlon",
        "indicator_has_address",
    ]
    df = df.drop(columns=drop_cols)
    return df


def clean_cc_property_locations_data(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    df = get_raw_cc_residential_property_characteristics_data(
        raw_file_path=raw_file_path, force_repull=force_repull
    )
    df = df.convert_dtypes()
    df = clean_cc_property_locations_drop_cols(df)
    df = clean_cc_property_locations_ohare_noise_col(df)
    df = clean_cc_property_locations_floodplain_col(df)
    df = clean_cc_property_locations_withinmr100_col(df)
    df = clean_cc_property_locations_withinmr101300_col(df)
    return df
