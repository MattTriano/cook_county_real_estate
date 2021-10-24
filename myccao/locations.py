from typing import Dict, List, Union, Optional

import geopandas as gpd
import pandas as pd
from pandas.api.types import CategoricalDtype

from utils import get_df_of_data_portal_data, get_gdf_of_data_portal_data


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


def clean_cc_property_locations_reps_dist_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Illinois state representative district"""
    df["reps_dist"] = df["reps_dist"].astype("category")
    return df


def clean_cc_property_locations_mailing_state_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Mailing state of property owner"""
    df["mailing_state"] = df["mailing_state"].astype("category")
    return df


def clean_cc_property_locations_school_hs_district_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Name of the high school district the property falls within, including
    CPS catchment zones."""
    df["school_hs_district"] = df["school_hs_district"].astype("category")
    return df


def clean_cc_property_locations_school_elem_district_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Name of the elementary/middle school district the property falls
    within, including CPS catchment zones."""
    df["school_elem_district"] = df["school_elem_district"].astype("category")
    return df


def clean_cc_property_locations_municipality_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Municipality name."""
    df["municipality"] = df["municipality"].astype("category")
    return df


def clean_cc_property_locations_nbhd_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """3-digit assessor neighborhood, only unique when combined with township
    number."""
    df["nbhd"] = df["nbhd"].astype(str).str.zfill(3).astype("category")
    return df


def clean_cc_property_locations_tif_agencynum_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Tax Increment Financing (TIF) district."""
    df["tif_agencynum"] = df["tif_agencynum"].astype("category")
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
    df = get_raw_cc_property_locations_data(
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
    df = clean_cc_property_locations_reps_dist_col(df)
    df = clean_cc_property_locations_mailing_state_col(df)
    df = clean_cc_property_locations_school_hs_district_col(df)
    df = clean_cc_property_locations_school_elem_district_col(df)
    df = clean_cc_property_locations_municipality_col(df)
    df = clean_cc_property_locations_nbhd_col(df)
    df = clean_cc_property_locations_tif_agencynum_col(df)
    return df


def get_raw_chicago_building_footprints_geodata(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    """data dictionary:
    https://data.cityofchicago.org/api/assets/003C600C-3A66-4605-8E7E-2477AAE95E16"""
    gdf = get_gdf_of_data_portal_data(
        file_name="cc_chicago_building_footprints.parquet.gzip",
        url="https://data.cityofchicago.org/api/geospatial/hz9b-7nh8?method=export&format=Shapefile",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return gdf


def clean_chicago_building_footprint_bldg_create_date_col(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Date footprint created."""
    gdf["BLDG_CREATE_DATE"] = gdf["date_bld_2"] + " " + gdf["time_bld_2"]
    gdf["BLDG_CREATE_DATE"] = gdf["BLDG_CREATE_DATE"].str[:19]
    return gdf


def clean_chicago_building_footprint_bldg_active_date_col(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Date footprint given ACTIVE status."""
    gdf["BLDG_ACTIVE_DATE"] = gdf["date_bldg_"] + " " + gdf["time_bldg_"]
    gdf["BLDG_ACTIVE_DATE"] = gdf["BLDG_ACTIVE_DATE"].str[:19]
    return gdf


def clean_chicago_building_footprint_bldg_end_date_col(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Date footprint given DEMOLISHED status (Demolished buildings are
    removed from the BUILDINGS layer and moved to a ‘DEMOLISHED’ layer."""
    gdf["BLDG_END_DATE"] = gdf["date_bld_3"] + " " + gdf["time_bld_3"]
    gdf["BLDG_END_DATE"] = gdf["BLDG_END_DATE"].str[:19]
    return gdf


def clean_chicago_building_footprint_condition_as_of_date_col(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Per documentation: Not actively maintained."""
    gdf["CONDITION_AS_OF_DATE"] = gdf["date_condi"] + " " + gdf["time_condi"]
    gdf["CONDITION_AS_OF_DATE"] = gdf["CONDITION_AS_OF_DATE"].str[:19]
    return gdf


def clean_chicago_building_footprint_demolished_date_col(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Per documentation: N/A."""
    gdf["DEMOLISHED_DATE"] = gdf["date_demol"] + " " + gdf["time_demol"]
    gdf["DEMOLISHED_DATE"] = gdf["DEMOLISHED_DATE"].str[:19]
    return gdf


def clean_chicago_building_footprint_edit_date_col(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Per documentation: Internal Use Only."""
    gdf["EDIT_DATE"] = gdf["date_edit_"] + " " + gdf["time_edit_"]
    gdf["EDIT_DATE"] = gdf["EDIT_DATE"].str[:19]
    return gdf


def clean_chicago_building_footprint_bldg_condition_col(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Per documentation: Not actively maintained."""
    gdf["BLDG_CONDITION"] = gdf["bldg_condi"].copy()
    gdf["BLDG_CONDITION"] = gdf["BLDG_CONDITION"].str.replace(
        "UNNHABITABLE", "UNINHABITABLE"
    )
    gdf["BLDG_CONDITION"] = gdf["BLDG_CONDITION"].astype("category")
    return gdf


def clean_chicago_building_footprint_geodata(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    gdf = get_raw_chicago_building_footprints_geodata(
        raw_file_path=raw_file_path, force_repull=force_repull
    )
    gdf = gdf.convert_dtypes()
    gdf = clean_chicago_building_footprint_bldg_create_date_col(gdf)
    gdf = clean_chicago_building_footprint_bldg_active_date_col(gdf)
    gdf = clean_chicago_building_footprint_bldg_end_date_col(gdf)
    gdf = clean_chicago_building_footprint_condition_as_of_date_col(gdf)
    gdf = clean_chicago_building_footprint_demolished_date_col(gdf)
    gdf = clean_chicago_building_footprint_edit_date_col(gdf)
    gdf = clean_chicago_building_footprint_bldg_condition_col(gdf)
    return gdf
