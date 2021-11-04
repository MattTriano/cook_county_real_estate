from typing import Dict, List, Union, Optional
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
from pandas.api.types import CategoricalDtype

from myccao.utils import (
    get_df_of_data_portal_data,
    conditionally_fill_col_vals,
    make_gdf_from_latlongs,
    download_zip_archive,
    prepare_raw_file_path,
    make_point_in_polygon_feature,
)
from myccao.locations import clean_cc_property_locations_data


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


def fill_latlong_missing_cc_residential_prop_char_cols(
    cc_res_prop_char_df: pd.DataFrame,
) -> pd.DataFrame:
    cc_prop_locs_df = clean_cc_property_locations_data()
    cc_res_prop_char_w_locs_df = pd.merge(
        left=cc_res_prop_char_df,
        right=cc_prop_locs_df,
        how="left",
        left_on="PIN",
        right_on="pin",
        suffixes=("_char", "_loc"),
    )
    lon_null_mask = cc_res_prop_char_w_locs_df["longitude"].isnull()
    Lon_null_mask = cc_res_prop_char_w_locs_df["Longitude"].isnull()
    lat_null_mask = cc_res_prop_char_w_locs_df["latitude"].isnull()
    Lat_null_mask = cc_res_prop_char_w_locs_df["Latitude"].isnull()
    cc_res_prop_char_w_locs_df = conditionally_fill_col_vals(
        df=cc_res_prop_char_w_locs_df,
        mask=(~lon_null_mask & Lon_null_mask),
        null_col="Longitude",
        fill_col="longitude",
    )
    cc_res_prop_char_w_locs_df = conditionally_fill_col_vals(
        df=cc_res_prop_char_w_locs_df,
        mask=(~lat_null_mask & Lat_null_mask),
        null_col="Latitude",
        fill_col="latitude",
    )
    cc_res_prop_char_w_locs_df = conditionally_fill_col_vals(
        df=cc_res_prop_char_w_locs_df,
        mask=(~lon_null_mask & Lon_null_mask),
        null_col="O'Hare Noise",
        fill_col="ohare_noise",
    )
    cc_res_prop_char_w_locs_df = conditionally_fill_col_vals(
        df=cc_res_prop_char_w_locs_df,
        mask=(~lon_null_mask & Lon_null_mask),
        null_col="Floodplain",
        fill_col="floodplain",
    )
    cc_res_prop_char_w_locs_df["withinmr300"] = (
        cc_res_prop_char_w_locs_df["withinmr100"]
    ) | (cc_res_prop_char_w_locs_df["withinmr101300"])
    cc_res_prop_char_w_locs_df = conditionally_fill_col_vals(
        df=cc_res_prop_char_w_locs_df,
        mask=(~lon_null_mask & Lon_null_mask),
        null_col="Near Major Road",
        fill_col="withinmr300",
    )
    cc_res_prop_char_w_locs_df = cc_res_prop_char_w_locs_df.drop(
        columns=["withinmr300"]
    )
    cc_res_prop_char_w_locs_df = cc_res_prop_char_w_locs_df.loc[
        (cc_res_prop_char_w_locs_df["Latitude"].notnull())
        & (cc_res_prop_char_w_locs_df["Longitude"].notnull())
    ].copy()
    cc_res_prop_char_w_locs_df = cc_res_prop_char_w_locs_df.reset_index(
        drop=True
    )
    return cc_res_prop_char_w_locs_df


def extend_cc_residential_prop_chars_ohare_noise_zone(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    ohare_noise_gdf = gdf.loc[(gdf["O'Hare Noise"])].copy()
    ohare_noise_zone = ohare_noise_gdf.iloc[0:1].copy()
    ohare_noise_zone = ohare_noise_zone.reset_index(drop=True)
    ohare_noise_zone["geometry"] = ohare_noise_gdf[
        "geometry"
    ].unary_union.convex_hull

    ohare_noise_zone_mask = gdf.within(ohare_noise_zone.loc[0, "geometry"])
    gdf.loc[ohare_noise_zone_mask, "O'Hare Noise"] = True
    return gdf


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
    df = fill_latlong_missing_cc_residential_prop_char_cols(df)
    gdf = make_gdf_from_latlongs(df)
    gdf = extend_cc_residential_prop_chars_ohare_noise_zone(gdf)
    gdf = make_fema_flood_features(gdf)
    return gdf


def download_fema_firm_db_zip_archive_for_cook_county(
    force_repull: bool = False,
) -> None:
    """To adapt to other counties, go to the below link, zoom in until panels
    appear and click a panel in the county of interest. A tooltip will appear,
    and while the panel is not county, there will be a download link for GIS
    data for that panel's county. Copy that link and enter it as the `url`
    parameter, with a `file_name` corresponding to the county.

    https://hazards-fema.maps.arcgis.com/apps/webappviewer/index.html
    """
    download_zip_archive(
        file_name="cook_county_fema_flood_insurance_rate_map_db_files.zip",
        url="https://msc.fema.gov/portal/downloadProduct?productID=NFHL_17031C",
        raw_file_path=None,
        force_repull=force_repull,
    )


def get_fema_firm_db_table_file_name_mapper(
    location_label: str,
) -> Dict[str, str]:
    """For making FEMA FIRM db table names more interpretable. 
    More details available at:
    https://www.fema.gov/sites/default/files/documents/ \
     fema_firm-database-technical-reference.pdf
    """
    prefix = f"fema_firm_{location_label}"
    table_name_map = {
        "S_BASE_INDEX": f"{prefix}__raster_base_map_index.parquet.gzip",
        "S_BFE": f"{prefix}__base_flood_elev_lines.parquet.gzip",
        "S_CST_GAGE": f"{prefix}__coastal_gauge_details.parquet.gzip",
        "S_CST_TSCT_LN": f"{prefix}__coastal_transect_lines.parquet.gzip",
        "S_FIRM_PAN": f"{prefix}__map_panel_data.parquet.gzip",
        "S_FLD_HAZ_AR": f"{prefix}__flood_hazard_areas.parquet.gzip",
        "S_FLD_HAZ_LN": f"{prefix}__flood_hazard_area_boundaries.parquet.gzip",
        "S_GEN_STRUCT": f"{prefix}__flood_control_structures.parquet.gzip",
        "S_HYDRO_REACH": f"{prefix}__hydrologic_connections.parquet.gzip",
        "S_LABEL_LD": f"{prefix}__label_to_feature_lines.parquet.gzip",
        "S_LABEL_PT": f"{prefix}__label_points_and_details.parquet.gzip",
        "S_LEVEE": f"{prefix}__levee_centerlines.parquet.gzip",
        "S_LIMWA": f"{prefix}__limit_of_moderate_wave_action.parquet.gzip",
        "S_LOMR": f"{prefix}__lomrs_not_yet_in_firm.parquet.gzip",
        "S_NODES": f"{prefix}__hydrologic_nodes.parquet.gzip",
        "S_PFD_LN": f"{prefix}__primary_frontal_dune_lines.parquet.gzip",
        "S_PLSS_AR": f"{prefix}__public_land_survey_system_areas.parquet.gzip",
        "S_POL_AR": f"{prefix}__political_areas.parquet.gzip",
        "S_PROFIL_BASLN": f"{prefix}__stream_centerlines_and_profiles.parquet.gzip",
        "S_STN_START": f"{prefix}__stream_starting_points.parquet.gzip",
        "S_SUBBASINS": f"{prefix}__subbasins.parquet.gzip",
        "S_SUBMITTAL_INFO": f"{prefix}__survey_scope_info.parquet.gzip",
        "S_TRNSPORT_LN": f"{prefix}__road_rails_transport_lines.parquet.gzip",
        "S_TSCT_BASLN": f"{prefix}__coastal_transect_baselines.parquet.gzip",
        "S_WTR_AR": f"{prefix}__hydrography_feature_areas.parquet.gzip",
        "S_WTR_LN": f"{prefix}__hydrography_feature_lines.parquet.gzip",
        "S_XS": f"{prefix}__cross_section_lines.parquet.gzip",
    }
    return table_name_map


def unpack_fema_firm_db_tables_from_zip_archive(
    location_label: str = "cc",
    zip_file_name: str = "cook_county_fema_flood_insurance_rate_map_db_files.zip",
    raw_file_path: Union[str, None] = None,
) -> None:
    raw_file_path = prepare_raw_file_path(
        file_name=zip_file_name, raw_file_path=raw_file_path
    )
    raw_file_dir = os.path.dirname(raw_file_path)

    with ZipFile(raw_file_path) as zf:
        zipped_files = zf.namelist()
    shapefile_names = [
        fn.split(".")[0] for fn in zipped_files if fn.lower().endswith(".shp")
    ]

    fema_table_name_mapper = get_fema_firm_db_table_file_name_mapper(
        location_label=location_label
    )
    for shapefile_name in shapefile_names:
        output_name = fema_table_name_mapper[shapefile_name]
        output_path = os.path.join(raw_file_dir, output_name)
        file_path = f"zip://{raw_file_path}!{shapefile_name}.shp"
        fema_gdf = gpd.read_file(file_path)
        fema_gdf = fema_gdf.convert_dtypes()
        fema_gdf.to_parquet(output_path, compression="gzip")


def make_fema_flood_features(
    gdf: gpd.GeoDataFrame,
    fema_file_name: str = "fema_firm_cc__flood_hazard_areas.parquet.gzip",
) -> gpd.GeoDataFrame:
    gdf_ = gdf.copy()
    gdf_.sindex
    raw_file_path = prepare_raw_file_path(file_name=fema_file_name)
    if not os.path.isfile(raw_file_path):
        download_fema_firm_db_zip_archive_for_cook_county()
        unpack_fema_firm_db_tables_from_zip_archive()
    fema_gdf = gpd.read_parquet(raw_file_path)
    if gdf_.crs != fema_gdf.crs:
        fema_gdf = fema_gdf.to_crs(gdf_.crs)
    gdf_ = make_point_in_polygon_feature(
        gdf=gdf_,
        zone_gdf=fema_gdf.loc[
            (fema_gdf["ZONE_SUBTY"] == "0.2 PCT ANNUAL CHANCE FLOOD HAZARD")
        ].copy(),
        zone_descr="FEMA_500yr_flood_risk",
    )
    gdf_ = make_point_in_polygon_feature(
        gdf=gdf_,
        zone_gdf=fema_gdf.loc[(fema_gdf["ZONE_SUBTY"] == "FLOODWAY")].copy(),
        zone_descr="FEMA_floodway",
    )
    return gdf_


def make_within_quarter_mile_of_interstate_feature(
    gdf_: gpd.GeoDataFrame, cc_streets_gdf: Optional[gpd.GeoDataFrame] = None
) -> gpd.GeoDataFrame:
    """Per the paper at the link below, air pollution tapers off to background
    levels around 400 meters from interstates, which is about a quarter mile.
    https://web.archive.org/web/20210416065345/https://pubs.acs.org/doi/pdf/10.1021/acs.est.7b00891
    """
    if cc_streets_gdf is None:
        cc_streets_gdf = loaders.get_raw_cook_county_gis_streets()
    zone_gdf = cc_streets_gdf.loc[
        (cc_streets_gdf["HWYTYPE"] == "INTERSTATE")
    ].copy()
    zone_gdf["geometry"] = zone_gdf["geometry"].buffer(1320)
    gdf_ = make_point_in_polygon_feature(
        gdf=gdf_, zone_gdf=zone_gdf, zone_descr="within_qtr_mile_of_interstate"
    )
    return gdf_
