import os
from typing import Dict, List, Union, Optional

import missingno as msno
import pandas as pd
from pandas.api.types import CategoricalDtype
import geopandas as gpd
from shapely.geometry import Point


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


def prepare_raw_file_path(
    file_name: str, raw_file_path: Union[str, None] = None
) -> os.path:
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
    return raw_file_path


def download_zip_archive(
    file_name: str,
    url: str,
    raw_file_path: Union[str, None] = None,
    force_repull: bool = False,
) -> None:
    raw_file_path = prepare_raw_file_path(
        file_name=file_name, raw_file_path=raw_file_path
    )
    if not os.path.isfile(raw_file_path) or force_repull:
        urlretrieve(url, raw_file_path)


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


def make_gdf_from_latlongs(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    long_col: str = "Longitude",
    geom_col: str = "geometry",
    latlong_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    df = df.copy()
    df[geom_col] = df.apply(lambda x: Point(x[long_col], x[lat_col]), axis=1)
    gdf = gpd.GeoDataFrame(df, crs=latlong_crs)
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


def conditionally_fill_col_vals(
    df: pd.DataFrame, mask: pd.Series, null_col: str, fill_col: str
) -> pd.DataFrame:
    df = df.copy()
    df.loc[mask, null_col] = df.loc[mask, fill_col].copy()
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


def make_point_in_polygon_feature(
    gdf: gpd.GeoDataFrame,
    zone_gdf: gpd.GeoDataFrame,
    zone_descr: str,
    geom_col: str = "geometry",
) -> gpd.GeoDataFrame:
    zone_union = zone_gdf[geom_col].unary_union
    sindex_query_results = gdf.sindex.query(zone_union, predicate="intersects")
    gdf[zone_descr] = False
    gdf.loc[sindex_query_results, zone_descr] = True
    gdf[zone_descr] = gdf[zone_descr].astype("boolean")
    return gdf
