from typing import Dict, List, Union, Optional

import pandas as pd
from pandas.api.types import CategoricalDtype

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
    """The property's flood risk direction represented in a numeric value
    based on the change in risk for the location from 2020 to 2050 for the
    climate model realization of the RCP 4.5 mid emissions scenario.
    -1 = descreasing, 0 = stationary, 1 = increasing.
    Data provided by First Street and academics at UPenn."""
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


def clean_cc_property_locations_fs_flood_factor_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """The property's First Street Flood Factor, a numeric integer from 1-10
    (where 1 = minimal and 10 = extreme) based on flooding risk to the
    building footprint. Flood risk is defined as a combination of cumulative
    risk over 30 years and flood depth. Flood depth is calculated at the
    lowest elevation of the building footprint (large."""
    flood_factor_risk_map = CategoricalDtype(
        categories=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ordered=True
    )
    df["fs_flood_factor"] = df["fs_flood_factor"].astype(flood_factor_risk_map)
    return df


def clean_cc_property_locations_commissioner_dist_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Cook County Commissioner district."""
    df["commissioner_dist"] = df["commissioner_dist"].astype("category")
    return df


def clean_cc_property_locations_senate_dist_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Illinois state senate district."""
    df["senate_dist"] = df["senate_dist"].astype("category")
    return df


def clean_cc_property_locations_township_name_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Township name."""
    df["township_name"] = df["township_name"].astype("category")
    return df


def clean_cc_property_locations_township_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Township number."""
    df["township"] = df["township"].astype("category")
    return df


def clean_cc_property_locations_puma_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """PUMA (Public Use Microdata Area) code of the PUMA containing the
    property, 2018 defitition."""
    df["puma"] = df["puma"].astype("category")
    return df


def clean_cc_property_locations_ward_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """City of Chicago ward number."""
    df["ward"] = df["ward"].astype("category")
    return df


def clean_cc_property_locations_ssa_no_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """City of Chicago Special Service Area number."""
    df["ssa_no"] = df["ssa_no"].astype("category")
    return df


def clean_cc_property_locations_ssa_name_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """City of Chicago Special Service Area name."""
    df["ssa_name"] = df["ssa_name"].astype("category")
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
    df = clean_cc_property_locations_fs_flood_factor_col(df)
    df = clean_cc_property_locations_commissioner_dist_col(df)
    df = clean_cc_property_locations_senate_dist_col(df)
    df = clean_cc_property_locations_township_name_col(df)
    df = clean_cc_property_locations_township_col(df)
    df = clean_cc_property_locations_puma_col(df)
    df = clean_cc_property_locations_ward_col(df)
    df = clean_cc_property_locations_ssa_no_col(df)
    df = clean_cc_property_locations_ssa_name_col(df)
    return df
