import os
from typing import Dict, List, Union, Optional

import missingno as msno
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
                round(100 * df[col].isnull().sum() / n_rows, 4)
                for col in col_list
            ],
        }
    )
    df_details = df_details.sort_values(by="unique_vals")
    df_details = df_details.reset_index(drop=True)
    return df_details


def plot_highly_missingness_correlated_cols(
    col: str, plot_df: pd.DataFrame, i: int = 0, positive_corr: bool = True
) -> None:
    plot_df = plot_df.copy()
    plot_df = plot_df.sort_values(by=col)
    plot_df = plot_df.reset_index(drop=True)

    plot_notnull_df = plot_df.loc[plot_df[col].notnull()].copy()
    plot_deets_df = get_df_column_details(plot_notnull_df)
    plot_deets_df = plot_deets_df.sort_values(
        by="pct_null", ascending=positive_corr
    )
    plot_deets_df = plot_deets_df.reset_index(drop=True)

    view_cols = [col]
    other_cols = list(plot_deets_df["feature"][i : i + 49])
    other_cols = [c for c in other_cols if c != col]
    view_cols.extend(other_cols)

    msno.matrix(plot_df[view_cols])


def get_df_of_data_portal_data(
    file_name: str,
    url: str,
    raw_file_path: Union[str, None] = None,
    force_repull: bool = False,
) -> pd.DataFrame:
    if raw_file_path is None:
        raw_file_dir = os.path.join(
            os.path.expanduser("~"),
            "projects",
            "cook_county_real_estate",
            "data_raw",
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
            os.path.expanduser("~"),
            "projects",
            "cook_county_real_estate",
            "data_raw",
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


def clean_cc_real_estate_sales_arms_length_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
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
) -> pd.DataFrame:
    if clean_file_path is None:
        file_dir = os.path.join(
            os.path.expanduser("~"),
            "projects",
            "cook_county_real_estate",
            "data_clean",
        )
        clean_file_path = os.path.join(
            file_dir, "cc_real_estate_sales.parquet.gzip"
        )
    if (
        os.path.isfile(clean_file_path)
        and not force_reclean
        and not force_repull
    ):
        df = pd.read_parquet(clean_file_path)
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
    return gdf


def get_clean_cc_residential_neighborhood_geodata(
    clean_file_path: Union[str, bool] = None,
    raw_file_path: Union[str, bool] = None,
    force_reclean: bool = False,
    force_repull: bool = False,
) -> gpd.GeoDataFrame:
    if clean_file_path is None:
        file_dir = os.path.join(
            os.path.expanduser("~"),
            "projects",
            "cook_county_real_estate",
            "data_clean",
        )
        clean_file_path = os.path.join(
            file_dir, "cc_residential_neighborhood_boundaries.parquet.gzip"
        )
    if (
        os.path.isfile(clean_file_path)
        and not force_reclean
        and not force_repull
    ):
        gdf = gpd.read_parquet(clean_file_path)
        return gdf
    elif force_reclean and not force_repull:
        gdf = clean_cc_residential_neighborhood_geodata(
            raw_file_path=raw_file_path
        )
    else:
        gdf = clean_cc_residential_neighborhood_geodata(
            raw_file_path=raw_file_path, force_repull=force_repull
        )
    gdf.to_parquet(clean_file_path, compression="gzip")
    return gdf


def get_raw_cc_residential_property_characteristics_data(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> pd.DataFrame:
    df = get_df_of_data_portal_data(
        file_name="cc_residential_property_characteristics.parquet.gzip",
        url="https://datacatalog.cookcountyil.gov/api/views/bcnq-qi2z/rows.csv?accessType=DOWNLOAD",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return df


def clean_cc_residential_property_characteristics_data(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    df = get_raw_cc_residential_property_characteristics_data(
        raw_file_path=raw_file_path, force_repull=force_repull
    )
    df = df.convert_dtypes()
    df["Property Class"] = df["Property Class"].astype("category")
    is_condo_mask = df["Property Class"] == 299
    condo_df = df.loc[is_condo_mask].copy()
    condo_df = condo_df.reset_index(drop=True)
    non_condo_df = df.loc[~is_condo_mask].copy()
    non_condo_df = non_condo_df.reset_index(drop=True)


def process_condo_property_characteristicts_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    null_cols = [
        "Building Square Feet",
        "Renovation",
        "Site Desireability",
        "Garage 1 Size",
        "Garage 1 Material",
        "Garage 1 Attachment",
        "Garage 1 Area",
        "Garage 2 Size",
        "Garage 2 Material",
        "Construction Quality",
        "Garage 2 Attachment",
        "Other Improvements",
        "Repair Condition",
        "Multi Code",
        "Number of Commercial Units",
        "Square root of improvement size",
        "Total Building Square Feet",
        "Multi-Family Indicator",
        "Improvement Size Squared",
        "Garage 2 Area",
        "Cathedral Ceiling",
        "Garage indicator",
        "Half Baths",
        "Type of Residence",
        "Apartments",
        "Wall Material",
        "Roof Material",
        "Rooms",
        "Bedrooms",
        "Basement",
        "Design Plan",
        "Central Heating",
        "Other Heating",
        "Central Air",
        "Fireplaces",
        "Attic Type",
        "Basement Finish",
    ]
    df = df.drop(columns=null_cols)
    return df


def dtypeset_simple_categorical_cols(
    df: pd.DataFrame, col_list: List
) -> pd.DataFrame:
    for col in col_list:
        df[col] = df[col].astype("category")
    return df


def clean_cc_residential_prop_chars_renovation_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "yes" not in df["Renovation"].unique():
        renovation_map = {1: "yes", 2: "no"}
        df["Renovation"] = df["Renovation"].map(renovation_map)
    df["Renovation"] = df["Renovation"].astype("category")
    return df


def clean_cc_residential_prop_chars_condo_class_factor_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "residential_condominium" not in df["Condo Class Factor"].unique():
        condo_class_factor_map = {
            200: "residential_land",
            299: "residential_condominium",
        }
        df["Condo Class Factor"] = df["Condo Class Factor"].map(
            condo_class_factor_map
        )
    df["Condo Class Factor"] = df["Condo Class Factor"].astype("category")
    return df


def clean_cc_residential_prop_chars_property_class_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    property_class_map = {
        200: "residential_land",
        201: "residential_garage",
        202: "one_story_residence__any_age__lt_1k_sq_ft",
        203: "one_story_residence__any_age__1k_to_1800_sq_ft",
        204: "one_story_residence__any_age__gt_1800_sq_ft",
        205: "two_plus_story_residence__gt_62_yrs_old__lt_2200_sq_ft",
        206: "two_plus_story_residence__gt_62_yrs_old__2201_to_4999_sq_ft",
        207: "two_plus_story_residence__lte_62_yrs_old__lte_2000_sq_ft",
        278: "two_plus_story_residence__lte_62_yrs_old__2001_to_3800_sq_ft",
        208: "two_plus_story_residence__lte_62_yrs_old__3801_to_4999_sq_ft",
        209: "two_plus_story_residence__any_age__gte_5000_sq_ft",
        210: "old_style_row_house__gt_62_yrs_old",
        211: "appartment_bldg_w_2_to_6_units__any_age",
        212: "mixed_use_bldg_w_lte_6_units__any_age__gte_20k_sq_ft",
        234: "split_level_residence_w_a_level_below_ground__any_age__any_sq_ft",
        241: "vacant_land_under_common_ownership_adjacent_to_residence",
        295: "individually_owned_townhowm_or_row_house__lte_62_years_old",
        299: "residential_condominium",
    }
    if "residential_condominium" not in df["Property Class"].unique():
        df["Property Class"] = df["Property Class"].map(property_class_map)
    df["Property Class"] = df["Property Class"].astype("category")
    return df
