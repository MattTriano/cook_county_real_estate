import os
from typing import Dict, List, Union, Optional

import missingno as msno
import pandas as pd
from pandas.api.types import CategoricalDtype
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
    if "Property Class Descr" not in list(df.columns):
        df["Property Class Descr"] = df["Property Class"].map(
            property_class_map
        )
    df["Property Class"] = df["Property Class"].astype("category")
    df["Property Class Descr"] = df["Property Class Descr"].astype("category")
    return df


def clean_cc_residential_prop_chars_neighborhood_code_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["Neighborhood Code"] = df["Neighborhood Code"].astype("string")
    df["Neighborhood Code"] = df["Neighborhood Code"].str.zfill(3)
    df["Neighborhood Code"] = df["Neighborhood Code"].astype("category")
    return df


def clean_cc_residential_prop_chars_town_code_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["Town Code"] = df["Town Code"].astype("string")
    df["Town Code"] = df["Town Code"].astype("category")
    return df


def clean_cc_residential_prop_chars_type_of_residence_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    type_of_residence_map = {
        1: "one story",
        2: "two story",
        3: "three story or higher",
        4: "split level",
        5: "1.5 story (one story w/ partial livable attic; 50% sq footage of 1st floor)",
        6: "1.6 story (one story w/ partial livable attic; 60% sq footage of 1st floor)",
        7: "1.7 story (one story w/ partial livable attic; 70% sq footage of 1st floor)",
        8: "1.8 story (one story w/ partial livable attic; 80% sq footage of 1st floor)",
        9: "1.9 story (one story w/ partial livable attic; 90% sq footage of 1st floor)",
    }
    if "one story" not in df["Type of Residence"].unique():
        df["Type of Residence"] = df["Type of Residence"].map(
            type_of_residence_map
        )
    df["Type of Residence"] = df["Type of Residence"].astype("category")
    return df


def clean_cc_residential_prop_chars_apartments_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    bad_apartment_value_mask = (df["Apartments"] < 0) | (df["Apartments"] > 6)
    df.loc[bad_apartment_value_mask, "Apartments"] = None
    df["Apartments"] = df["Apartments"].astype("Int8")
    return df


def clean_cc_residential_prop_chars_wall_material_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    wall_material_map = {
        1: "Wood",
        2: "Masonry",
        3: "Wood and Masonry",
        4: "Stucco",
    }
    if "Wood" not in df["Wall Material"].unique():
        df["Wall Material"] = df["Wall Material"].map(wall_material_map)
    df["Wall Material"] = df["Wall Material"].astype("category")
    return df


def clean_cc_residential_prop_chars_roof_material_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    roof_material_map = {
        1: "Shingle/Asphalt",
        2: "Tar and Gravel",
        3: "Slate",
        4: "Shake",
        5: "Tile",
        6: "Other",
    }
    if "Shingle/Asphalt" not in df["Roof Material"].unique():
        df["Roof Material"] = df["Roof Material"].map(roof_material_map)
    df["Roof Material"] = df["Roof Material"].astype("category")
    return df


def clean_cc_residential_prop_chars_rooms_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    bad_rooms_value_mask = df["Rooms"] > 100
    df.loc[bad_rooms_value_mask, "Rooms"] = None
    df["Rooms"] = df["Rooms"].astype("Int8")
    return df


def clean_cc_residential_prop_chars_bedrooms_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    bad_bedrooms_value_mask = df["Bedrooms"] > 50
    df.loc[bad_bedrooms_value_mask, "Bedrooms"] = None
    df["Bedrooms"] = df["Bedrooms"].astype("Int8")
    return df


def clean_cc_residential_prop_chars_basement_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    basement_map = {1: "Full", 2: "Slab", 3: "Partial", 4: "Crawl"}
    if "Full" not in df["Basement"].unique():
        df["Basement"] = df["Basement"].map(basement_map)
    df["Basement"] = df["Basement"].astype("category")
    return df


def clean_cc_residential_prop_chars_basement_finish_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    basement_finish_map = {
        1: "Formal rec room",
        2: "Apartment",
        3: "Unfinished",
    }
    if "Apartment" not in df["Basement Finish"].unique():
        df["Basement Finish"] = df["Basement Finish"].map(basement_finish_map)
    df["Basement Finish"] = df["Basement Finish"].astype("category")
    return df


def clean_cc_residential_prop_chars_central_heating_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    central_heating_map = {
        1: "Warm air",
        2: "Hot water steam",
        3: "Electric",
        4: "Other",
    }
    if "Warm air" not in df["Central Heating"].unique():
        df["Central Heating"] = df["Central Heating"].map(central_heating_map)
    df["Central Heating"] = df["Central Heating"].astype("category")
    return df


def clean_cc_residential_prop_chars_other_heating_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    other_heating_map = {
        1: "Floor furnace",
        2: "Unit heater",
        3: "Stove",
        4: "Solar",
        5: "none",
    }
    if "none" not in df["Other Heating"].unique():
        df["Other Heating"] = df["Other Heating"].map(other_heating_map)
    df["Other Heating"] = df["Other Heating"].astype("category")
    return df


def clean_cc_residential_prop_chars_central_air_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["Central Air"] = df["Central Air"].astype("boolean")
    return df


def clean_cc_residential_prop_chars_fireplaces_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["Fireplaces"] = df["Fireplaces"].astype("Int8")
    return df


def clean_cc_residential_prop_chars_attic_type_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    attic_type_map = {1: "Full", 2: "Partial", 3: "None"}
    if "None" not in df["Attic Type"].unique():
        df["Attic Type"] = df["Attic Type"].map(attic_type_map)
    df["Attic Type"] = df["Attic Type"].astype("category")
    return df


def clean_cc_residential_prop_chars_attic_finish_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    attic_finish_map = {
        0: "No Attic",
        1: "Living Area",
        2: "Apartment",
        3: "Unfinished",
    }
    if "No Attic" not in df["Attic Finish"].unique():
        df["Attic Finish"] = df["Attic Finish"].map(attic_finish_map)
    df["Attic Finish"] = df["Attic Finish"].astype("category")
    return df


def clean_cc_residential_prop_chars_half_baths_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["Half Baths"] = df["Half Baths"].astype("Int8")
    return df


def clean_cc_residential_prop_chars_full_baths_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Number of full bathrooms, defined as having a bath or shower. If this
    value is missing, the default value is set to 1."""
    df["Full Baths"] = df["Full Baths"].astype("Int8")
    return df


def clean_cc_residential_prop_chars_design_plan_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    design_plan_map = {0: "Unknown", 1: "Architect", 2: "Stock Plan"}
    if "Stock Plan" not in df["Design Plan"].unique():
        df["Design Plan"] = df["Design Plan"].map(design_plan_map)
    df["Design Plan"] = df["Design Plan"].astype("category")
    return df


def clean_cc_residential_prop_chars_cathedral_ceiling_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    cathedral_ceiling_map = {0: "Unknown", 1: "Yes", 2: "No"}
    if "Yes" not in df["Cathedral Ceiling"].unique():
        df["Cathedral Ceiling"] = df["Cathedral Ceiling"].map(
            cathedral_ceiling_map
        )
    df["Cathedral Ceiling"] = df["Cathedral Ceiling"].astype("category")
    return df


def clean_cc_residential_prop_chars_construction_quality_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    construction_quality_map = {
        1: "Deluxe",
        2: "Average",
        3: "Unknown",
        4: "Unknown",
    }
    if "Deluxe" not in df["Construction Quality"].unique():
        df["Construction Quality"] = df["Construction Quality"].map(
            construction_quality_map
        )
    df["Construction Quality"] = df["Construction Quality"].astype("category")
    return df


def clean_cc_residential_prop_chars_renovation_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "yes" not in df["Renovation"].unique():
        renovation_map = {1: "yes", 2: "no"}
        df["Renovation"] = df["Renovation"].map(renovation_map)
    df["Renovation"] = df["Renovation"].astype("category")
    return df


def clean_cc_residential_prop_chars_site_desireability_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    site_desireability_map = {
        1: "Beneficial to Value",
        2: "Not relevant to Value",
        3: "Detracts from Value",
    }
    if "Not relevant to Value" not in df["Site Desireability"].unique():
        df["Site Desireability"] = df["Site Desireability"].map(
            site_desireability_map
        )
    df["Site Desireability"] = df["Site Desireability"].astype("category")
    return df


def get_garage_size_map() -> Dict:
    garage_size_map = {
        0: "Car Port or Driveway Only",
        1: "1 car",
        2: "1.5 car",
        3: "2 car",
        4: "2.5 cars",
        5: "3 cars",
        6: "3.5 cars",
        7: "None",
        8: "4 cars",
        9: "Unknown",
    }
    return garage_size_map


def clean_cc_residential_prop_chars_garage_size_col(
    df: pd.DataFrame, garage_num: str = "1"
) -> pd.DataFrame:
    garage_size_map = get_garage_size_map()
    bad_garage_size_value_mask = df[f"Garage {garage_num} Size"] >= 9
    df.loc[bad_garage_size_value_mask, f"Garage {garage_num} Size"] = 9
    if "1 car" not in df[f"Garage {garage_num} Size"].unique():
        df[f"Garage {garage_num} Size"] = df[f"Garage {garage_num} Size"].map(
            garage_size_map
        )
    df[f"Garage {garage_num} Size"] = df[f"Garage {garage_num} Size"].astype(
        "category"
    )
    return df


def get_garage_material_map() -> Dict:
    garage_material_map = {
        0: "Car Port or Driveway Only",
        1: "Frame",
        2: "Masonry",
        3: "Frame/Masonry",
        4: "Stucco",
    }
    return garage_material_map


def clean_cc_residential_prop_chars_garage_material_col(
    df: pd.DataFrame, garage_num: str = "1"
) -> pd.DataFrame:
    garage_material_map = get_garage_material_map()
    if "Frame" not in df[f"Garage {garage_num} Material"].unique():
        df[f"Garage {garage_num} Material"] = df[
            f"Garage {garage_num} Material"
        ].map(garage_material_map)
    df[f"Garage {garage_num} Material"] = df[
        f"Garage {garage_num} Material"
    ].astype("category")
    return df


def get_garage_attachment_map() -> Dict:
    garage_attached_map = {0: "Car Port or Driveway Only", 1: "Yes", 2: "No"}
    return garage_attached_map


def clean_cc_residential_prop_chars_garage_attachment_col(
    df: pd.DataFrame, garage_num: str = "1"
) -> pd.DataFrame:
    garage_attached_map = get_garage_attachment_map()
    if "Frame" not in df[f"Garage {garage_num} Attachment"].unique():
        df[f"Garage {garage_num} Attachment"] = df[
            f"Garage {garage_num} Attachment"
        ].map(garage_attached_map)
    df[f"Garage {garage_num} Attachment"] = df[
        f"Garage {garage_num} Attachment"
    ].astype("category")
    return df


def get_garage_area_map() -> Dict:
    garage_area_map = {
        0: "Car Port or Driveway Only",
        1: "Yes, garage area included in building area",
        2: "No, garage area not included in building area",
    }
    return garage_area_map


def clean_cc_residential_prop_chars_garage_area_col(
    df: pd.DataFrame, garage_num: str = "1"
) -> pd.DataFrame:
    garage_area_map = get_garage_area_map()
    if "Frame" not in df[f"Garage {garage_num} Area"].unique():
        df[f"Garage {garage_num} Area"] = df[f"Garage {garage_num} Area"].map(
            garage_area_map
        )
    df[f"Garage {garage_num} Area"] = df[f"Garage {garage_num} Area"].astype(
        "category"
    )
    return df


def clean_cc_residential_prop_chars_porch_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    porch_map = {1: "Frame", 2: "Masonry", 3: "None"}
    if "None" not in df["Porch"].unique():
        df["Porch"] = df["Porch"].map(porch_map)
    df["Porch"] = df["Porch"].astype("category")
    return df


def clean_cc_residential_prop_chars_repair_condition_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    state_of_repair_map = {
        1: "Above average",
        2: "Average",
        3: "Below average",
    }
    if "Average" not in df["Repair Condition"].unique():
        df["Repair Condition"] = df["Repair Condition"].map(
            state_of_repair_map
        )
    df["Repair Condition"] = df["Repair Condition"].astype("category")
    return df


def clean_cc_residential_prop_chars_multi_code_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    number_of_buildings_map = {
        2: "One building",
        3: "Two buildings",
        4: "Three buildings",
        5: "Four buildings",
        6: "Five buildings",
        7: "Six buildings",
        8: "Bad Value Entered",
    }
    bad_multi_code_value_mask = df["Multi Code"] >= 8
    df.loc[bad_multi_code_value_mask, "Multi Code"] = 9
    if "One building" not in df["Multi Code"].unique():
        df["Multi Code"] = df["Multi Code"].map(number_of_buildings_map)
    df["Multi Code"] = df["Multi Code"].astype("category")
    return df


def clean_cc_residential_prop_chars_number_of_commercial_units_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["Number of Commercial Units"] = df["Number of Commercial Units"].astype(
        "Int8"
    )
    return df


def clean_cc_residential_prop_chars_date_of_most_recent_sale_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """The format for the dates appears to be "%m/%d/%Y", but this
    necessittated, setting `errors="coerce"`, which indicates some incorrectly
    formatted dates, but I couldn't find them via inspection. The time to run
    this step without a format was negligable, so I'll leave this with the
    robust default mode.
    """
    df["Date of Most Recent Sale"] = pd.to_datetime(
        df["Date of Most Recent Sale"]
    )
    return df


def clean_cc_residential_prop_chars_census_tract_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["Census Tract"] = df["Census Tract"].astype("string")
    df["Census Tract"] = df["Census Tract"].str.zfill(6)
    return df


def clean_cc_residential_prop_chars_modeling_group_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Modeling group, as defined by the property class. Properties with
    class 200, 201, 241, 299 is defined as "NCHARS", short for "no
    characteristics", which are condos and vacant land classes. Properties
    with class 202, 203, 204, 205, 206, 207, 208, 209, 210, 235, 278, and 295
    are "SF", short for "single-family." Properties with class 211 and 212 are
    "MF", short for ""multi-family."
    """
    df["Modeling Group"] = df["Modeling Group"].astype("category")
    return df


def clean_cc_residential_prop_chars_multi_property_indicator_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["Multi Property Indicator"] = df["Multi Property Indicator"].astype(
        "boolean"
    )
    return df


def clean_cc_residential_prop_chars_age_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Age of the property. If missing, this defaults to 10. This field is a
    combination of original age and effective age where original age refers to
    the oldest component of the building and effective age is a relative
    judgement due to renovations or other improvements. For instance, if a
    property is completely demolished and built up again, the age resets to 1.
    But if portions of the original structure are kept, it may be more
    complicated to determine the age."""
    df["Age"] = df["Age"].astype("Int32")
    return df


def clean_cc_residential_prop_chars_number_of_units_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """For condos, the number of units in a building"""
    df["Number of Units"] = df["Number of Units"].astype("Int16")
    return df


def clean_cc_residential_prop_chars_use_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Use of property - 1 = single family, 2 = multi-family. If absent,
    default value is 1."""
    use_of_property_map = {1: "Single Family", 2: "Multi Family"}
    if "Single Family" not in df["Use"].unique():
        df["Use"] = df["Use"].map(use_of_property_map)
    df["Use"] = df["Use"].astype("category")
    return df


def clean_cc_residential_prop_chars_condo_class_factor_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Factor variable for NCHARS modeling group. Divides properties into two:
    Properties with class 200, 201, and 241 are given condo class factor as
    200. Properties with class 299 are given condo class factor as 299"""
    condo_class_factor_map = {
        200: "residential_land",
        299: "residential_condominium",
    }
    if "residential_condominium" not in df["Condo Class Factor"].unique():
        df["Condo Class Factor"] = df["Condo Class Factor"].map(
            condo_class_factor_map
        )
    df["Condo Class Factor"] = df["Condo Class Factor"].astype("category")
    return df


def clean_cc_residential_prop_chars_multi_family_indicator_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Factor variable for MF modeling group. Properties with class 211 are
    given multi-family indicator as 211. Properties with class 212 are given
    multi-family indicator as 212."""
    multi_family_map = {
        211: "Yes",
        212: "Yes",
    }
    if "Yes" not in df["Multi-Family Indicator"].unique():
        df["Multi-Family Indicator"] = df["Multi-Family Indicator"].map(
            multi_family_map
        )
    df["Multi-Family Indicator"] = df["Multi-Family Indicator"].astype(
        "category"
    )
    return df


def clean_cc_residential_prop_chars_large_lot_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Large lot factor variable, where 1 acre of land (land square feet >
    43559) is defined as a large lot. 1 = large lot, 0 = not a large lot."""
    df["Large Lot"] = df["Large Lot"].astype("boolean")
    return df


def clean_cc_residential_prop_chars_cdu_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """About 20 different codes attached to the face sheet that denote a
    number of seemingly unrelated characteristics associated with a PIN,
    ranging from condition to types of subsidies. This field does not match
    across the SQL server/AS-400 for 2018."""
    df["Condition, Desirability and Utility"] = df[
        "Condition, Desirability and Utility"
    ].astype("category")
    return df


def clean_cc_residential_prop_chars_deed_type_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    deed_type_map = {"W": "Warranty", "O": "Other", "T": "Trustee"}
    if "Warranty" not in df["Deed Type"].unique():
        df["Deed Type"] = df["Deed Type"].map(deed_type_map)
    df["Deed Type"] = df["Deed Type"].astype("category")
    return df


def clean_cc_residential_prop_chars_condo_strata_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """For condominiums, shows the decile of mean unit market value a
    condominimum building is."""
    decile_cat_type = CategoricalDtype(
        categories=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ordered=True
    )
    df["Condo Strata"] = df["Condo Strata"].astype(decile_cat_type)
    return df


def clean_cc_residential_prop_chars_garage_indicator_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Indicates presence of a garage of any size."""
    df["Garage indicator"] = df["Garage indicator"].astype("boolean")
    return df


def clean_cc_residential_prop_chars_residential_share_of_building_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """A sum of "percent of ownership" that only includes residential units
    within a condo building."""
    rounding_error_mask = (df["Residential share of building"] > 1) & (
        df["Residential share of building"] < 1.01
    )
    bigger_error_mask = df["Residential share of building"] >= 1.01
    df.loc[rounding_error_mask, "Residential share of building"] = 1
    df.loc[bigger_error_mask, "Residential share of building"] = 1.01
    return df


def clean_cc_residential_prop_chars_pure_market_sale_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Indicator for pure market sale."""
    df["Pure Market Sale"] = df["Pure Market Sale"].astype("boolean")
    return df


def clean_cc_residential_prop_chars_town_and_neighborhood_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["Town and Neighborhood"] = df["Town Code"].astype("string") + df[
        "Neighborhood Code"
    ].astype("string")
    df["Town and Neighborhood"] = df["Town and Neighborhood"].astype(
        "category"
    )
    return df


def clean_cc_residential_prop_chars_ohare_noise_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Indicator for the property under O'Hare approach flight path,
    within 1/4 mile."""
    df["O'Hare Noise"] = df["O'Hare Noise"].astype("boolean")
    return df


def clean_cc_residential_prop_chars_floodplain_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Indicator for properties on a floodplain, defined as a FEMA Special
    Flood Hazard Area."""
    df["Floodplain"] = df["Floodplain"].astype("boolean")
    return df


def clean_cc_residential_prop_chars_near_major_road_col(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Indicates whether the property is within 300 ft of a major road."""
    df["Near Major Road"] = df["Near Major Road"].astype("boolean")
    return df


def clean_cc_residential_prop_chars_drop_cols(
    df: pd.DataFrame,
) -> pd.DataFrame:
    drop_cols = [
        "Other Improvements",
        "Age Squared",
        "Age Decade",
        "Age Decade Squared",
        "Lot Size Squared",
        "Improvement Size Squared",
        "Pure Market Filter",
        "Neigborhood Code (mapping)",
        "Square root of lot size",
        "Square root of age",
        "Square root of improvement size",
    ]
    df = df.drop(columns=drop_cols)
    return df


def conditionally_fill_col_vals(
    df: pd.DataFrame, mask: pd.Series, null_col: str, fill_col: str
) -> pd.DataFrame:
    df = df.copy()
    df.loc[mask, null_col] = df.loc[mask, fill_col].copy()
    return df


def clean_cc_residential_property_characteristics_data(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    df = get_raw_cc_residential_property_characteristics_data(
        raw_file_path=raw_file_path, force_repull=force_repull
    )
    df = df.convert_dtypes()
    df = clean_cc_residential_prop_chars_drop_cols(df)
    df = clean_cc_residential_prop_chars_property_class_col(df)
    df = clean_cc_residential_prop_chars_neighborhood_code_col(df)
    df = clean_cc_residential_prop_chars_town_code_col(df)
    df = clean_cc_residential_prop_chars_type_of_residence_col(df)
    df = clean_cc_residential_prop_chars_apartments_col(df)
    df = clean_cc_residential_prop_chars_wall_material_col(df)
    df = clean_cc_residential_prop_chars_roof_material_col(df)
    df = clean_cc_residential_prop_chars_rooms_col(df)
    df = clean_cc_residential_prop_chars_bedrooms_col(df)
    df = clean_cc_residential_prop_chars_basement_col(df)
    df = clean_cc_residential_prop_chars_basement_finish_col(df)
    df = clean_cc_residential_prop_chars_central_heating_col(df)
    df = clean_cc_residential_prop_chars_other_heating_col(df)
    df = clean_cc_residential_prop_chars_central_air_col(df)
    df = clean_cc_residential_prop_chars_fireplaces_col(df)
    df = clean_cc_residential_prop_chars_attic_type_col(df)
    df = clean_cc_residential_prop_chars_attic_finish_col(df)
    df = clean_cc_residential_prop_chars_half_baths_col(df)
    df = clean_cc_residential_prop_chars_full_baths_col(df)
    df = clean_cc_residential_prop_chars_design_plan_col(df)
    df = clean_cc_residential_prop_chars_cathedral_ceiling_col(df)
    df = clean_cc_residential_prop_chars_construction_quality_col(df)
    df = clean_cc_residential_prop_chars_renovation_col(df)
    df = clean_cc_residential_prop_chars_site_desireability_col(df)
    df = clean_cc_residential_prop_chars_garage_size_col(df, "1")
    df = clean_cc_residential_prop_chars_garage_size_col(df, "2")
    df = clean_cc_residential_prop_chars_garage_material_col(df, "1")
    df = clean_cc_residential_prop_chars_garage_material_col(df, "2")
    df = clean_cc_residential_prop_chars_garage_attachment_col(df, "1")
    df = clean_cc_residential_prop_chars_garage_attachment_col(df, "2")
    df = clean_cc_residential_prop_chars_garage_area_col(df, "1")
    df = clean_cc_residential_prop_chars_garage_area_col(df, "2")
    df = clean_cc_residential_prop_chars_repair_condition_col(df)
    df = clean_cc_residential_prop_chars_multi_code_col(df)
    df = clean_cc_residential_prop_chars_number_of_commercial_units_col(df)
    df = clean_cc_residential_prop_chars_date_of_most_recent_sale_col(df)
    df = clean_cc_residential_prop_chars_census_tract_col(df)
    df = clean_cc_residential_prop_chars_multi_property_indicator_col(df)
    df = clean_cc_residential_prop_chars_modeling_group_col(df)
    df = clean_cc_residential_prop_chars_age_col(df)
    df = clean_cc_residential_prop_chars_number_of_units_col(df)
    df = clean_cc_residential_prop_chars_condo_class_factor_col(df)
    df = clean_cc_residential_prop_chars_multi_family_indicator_col(df)
    df = clean_cc_residential_prop_chars_large_lot_col(df)
    df = clean_cc_residential_prop_chars_cdu_col(df)
    df = clean_cc_residential_prop_chars_deed_type_col(df)
    df = clean_cc_residential_prop_chars_condo_strata_col(df)
    df = clean_cc_residential_prop_chars_garage_indicator_col(df)
    df = clean_cc_residential_prop_chars_residential_share_of_building_col(df)
    df = clean_cc_residential_prop_chars_pure_market_sale_col(df)
    df = clean_cc_residential_prop_chars_town_and_neighborhood_col(df)
    df = clean_cc_residential_prop_chars_ohare_noise_col(df)
    df = clean_cc_residential_prop_chars_floodplain_col(df)
    df = clean_cc_residential_prop_chars_near_major_road_col(df)
    return df


def get_2010_cook_county_census_tract_gdf(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    gdf = get_gdf_of_data_portal_data(
        file_name="census_tracts__cook_county_2010.parquet.gzip",
        url="https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/TRACT/2010/tl_2020_17031_tract10.zip",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return gdf


def get_2020_cook_county_census_tract_gdf(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    gdf = get_gdf_of_data_portal_data(
        file_name="census_tracts__cook_county_2020.parquet.gzip",
        url="https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/TRACT/2020/tl_2020_17031_tract20.zip",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return gdf


def get_cook_county_boundary_gdf(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    # Not sure how durable the url
    # 'https://hub-cookcountyil.opendata.arcgis.com/datasets/ea127f9e96b74677892722069c984198_1/explore'
    # is, but I can deal with that if/when I have to.
    gdf = get_gdf_of_data_portal_data(
        file_name="cook_county_boundary_from_opendata.parquet.gzip",
        url="https://opendata.arcgis.com/api/v3/datasets/ea127f9e96b74677892722069c984198_1/downloads/data?format=shp&spatialRefId=3435",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return gdf


def split_cc_census_tracts_by_neighborhood() -> gpd.GeoDataFrame:
    cc_neighborhood_gdf = get_clean_cc_residential_neighborhood_geodata()
    cc_neighborhood_gdf = cc_neighborhood_gdf.drop_duplicates(
        ignore_index=True
    )
    cc_neighborhood_gdf["township_n"] = cc_neighborhood_gdf[
        "township_n"
    ].str.upper()
    cc_nbhd_gdf = cc_neighborhood_gdf[
        ["nbhd", "town_nbhd", "township_n", "geometry"]
    ].copy()

    census_tract10_gdf = get_2010_cook_county_census_tract_gdf()
    census_tract10_gdf = census_tract10_gdf.to_crs(cc_neighborhood_gdf.crs)
    census_tract10_gdf = census_tract10_gdf.loc[
        (
            census_tract10_gdf["AWATER10"]
            != census_tract10_gdf["AWATER10"].max()
        )
    ].copy()
    census_tract10_gdf = census_tract10_gdf.reset_index(drop=True)

    all_split_tracts = []
    for tract_i in range(len(census_tract10_gdf)):
        tract_geom = census_tract10_gdf.loc[tract_i:tract_i].copy()
        cc_nbhd_gdf_geoms = []
        intersecting_tracts = cc_nbhd_gdf.sjoin(
            tract_geom, how="inner", predicate="intersects"
        ).copy()
        intersecting_tracts = intersecting_tracts.reset_index(drop=True)
        for i in range(len(intersecting_tracts)):
            nbhd_geom = intersecting_tracts.loc[i:i].copy()
            tract_nbhd = tract_geom.overlay(nbhd_geom, how="intersection")
            cc_nbhd_gdf_geoms.append(tract_nbhd)
        cc_nbhd_gdf_geoms_gdf = pd.concat(cc_nbhd_gdf_geoms)
        all_split_tracts.append(cc_nbhd_gdf_geoms_gdf)

    new_gdf = pd.concat(all_split_tracts)
    new_gdf = new_gdf.reset_index(drop=True)
    assert (
        new_gdf.loc[
            (new_gdf["TRACTCE10_1"] == new_gdf["TRACTCE10_2"])
            & (new_gdf["GEOID10_1"] == new_gdf["GEOID10_2"])
            & (new_gdf["NAME10_1"] == new_gdf["NAME10_2"])
            & (new_gdf["NAMELSAD10_1"] == new_gdf["NAMELSAD10_2"])
            & (new_gdf["MTFCC10_1"] == new_gdf["MTFCC10_2"])
            & (new_gdf["FUNCSTAT10_1"] == new_gdf["FUNCSTAT10_2"])
            & (new_gdf["ALAND10_1"] == new_gdf["ALAND10_2"])
            & (new_gdf["INTPTLAT10_1"] == new_gdf["INTPTLAT10_2"])
            & (new_gdf["INTPTLON10_1"] == new_gdf["INTPTLON10_2"])
            & (new_gdf["STATEFP10_1"] == new_gdf["STATEFP10_2"])
            & (new_gdf["COUNTYFP10_1"] == new_gdf["COUNTYFP10_2"])
        ].shape[0]
        == new_gdf.shape[0]
    )
    drop_cols = [col for col in new_gdf.columns if col.endswith("_2")]
    new_gdf = new_gdf.drop(columns=drop_cols)
    new_gdf.columns = [col.replace("_1", "") for col in new_gdf.columns]
    new_gdf = new_gdf.drop(
        columns=[
            "index_right",
            "ALAND10",
            "AWATER10",
            "NAMELSAD10",
            "FUNCSTAT10",
            "MTFCC10",
        ]
    )

    cat_cols = [
        "STATEFP10",
        "COUNTYFP10",
        "TRACTCE10",
        "GEOID10",
        "NAME10",
        "nbhd",
        "town_nbhd",
        "township_n",
    ]
    for cat_col in cat_cols:
        new_gdf[cat_col] = new_gdf[cat_col].astype("category")

    new_gdf["INTPTLAT10"] = new_gdf["INTPTLAT10"].astype(float)
    new_gdf["INTPTLON10"] = new_gdf["INTPTLON10"].astype(float)
    new_gdf["rep_point"] = new_gdf.geometry.representative_point()
    return new_gdf


def get_cc_census_tracts_split_by_neighborhood_geodata(
    clean_file_path: Union[str, bool] = None,
    force_remake: bool = False,
) -> gpd.GeoDataFrame:
    if clean_file_path is None:
        file_dir = os.path.join(
            os.path.expanduser("~"),
            "projects",
            "cook_county_real_estate",
            "data_clean",
        )
        clean_file_path = os.path.join(
            file_dir,
            "cc_census_tracts_split_by_neighborhood_boundaries.parquet.gzip",
        )
    if os.path.isfile(clean_file_path) and not force_remake:
        gdf = gpd.read_parquet(clean_file_path)
        return gdf
    else:
        gdf = split_cc_census_tracts_by_neighborhood()
    gdf.to_parquet(clean_file_path, compression="gzip")
    return gdf
