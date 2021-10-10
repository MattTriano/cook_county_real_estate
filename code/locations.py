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


def clean_cc_property_locations_property_city_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """City of the property"""
    df["property_city"] = df["property_city"].str.replace(
        " GR VILL$", " GROVE VILLAGE", regex=True
    )
    df["property_city"] = df["property_city"].str.replace(
        " PK$", " PARK", regex=True
    )
    df["property_city"] = df["property_city"].str.replace(
        " HTS$", " HEIGHTS", regex=True
    )
    df["property_city"] = df["property_city"].str.replace(
        " HGT$", " HEIGHTS", regex=True
    )
    df["property_city"] = df["property_city"].str.replace(
        " HLS$", " HILLS", regex=True
    )
    df["property_city"] = df["property_city"].str.replace(
        "WESTERN SPRG", "WESTERN SPRINGS"
    )
    df["property_city"] = df["property_city"].str.replace(
        "OAKBROOK", "OAK BROOK"
    )
    df["property_city"] = df["property_city"].str.replace(
        "BERKLEY", "BERKELEY"
    )
    df["property_city"] = df["property_city"].str.replace(
        "BERKLEY", "BERKELEY"
    )
    df["property_city"] = df["property_city"].str.replace(
        "LAGRANGE", "LA GRANGE"
    )
    df["property_city"] = df["property_city"].str.replace(
        "SUMMIT ARGO", "SUMMIT"
    )
    df["property_city"] = df["property_city"].str.replace(
        "SUMMIT ARGO", "SUMMIT"
    )
    df["property_city"] = df["property_city"].str.replace(
        "^ARGO$", "SUMMIT", regex=True
    )
    df["property_city"] = df["property_city"].str.replace(
        "^ORLAND$", "ORLAND PARK", regex=True
    )
    df["property_city"] = df["property_city"].str.replace(
        "^SCHILLET PARK$", "SCHILLER PARK", regex=True
    )
    df["property_city"] = df["property_city"].astype("category")
    return df


def clean_cc_property_locations_fs_flood_risk_direction_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    flood_risk_map = {
        -1: "Descreasing",
        0: "Stationary",
        1: "Increasing",
    }
    if "Stationary" not in df["fs_flood_risk_direction"].unique():
        df["fs_flood_risk_direction"] = df["fs_flood_risk_direction"].map(
            flood_risk_map
        )
    df["fs_flood_risk_direction"] = df["fs_flood_risk_direction"].astype(
        "category"
    )
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
    df = clean_cc_property_locations_property_city_col(df)
    df = clean_cc_property_locations_fs_flood_risk_direction_col(df)
    return df
