import os
from typing import Dict, List, Union, Optional

import geopandas as gpd

from myccao.locations import clean_chicago_building_footprint_geodata
from myccao.utils import get_gdf_of_data_portal_data


def get_clean_building_footprint_geodata(
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
            file_dir, "cc_chicago_building_footprints.parquet.gzip"
        )
    if (
        os.path.isfile(clean_file_path)
        and not force_reclean
        and not force_repull
    ):
        gdf = gpd.read_parquet(clean_file_path)
        return gdf
    elif force_reclean and not force_repull:
        gdf = clean_chicago_building_footprint_geodata(
            raw_file_path=raw_file_path
        )
    else:
        gdf = clean_chicago_building_footprint_geodata(
            raw_file_path=raw_file_path, force_repull=force_repull
        )
    gdf.to_parquet(clean_file_path, compression="gzip")
    return gdf


def get_raw_chicago_city_boundary(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    gdf = get_gdf_of_data_portal_data(
        file_name="chicago_city_boundary.parquet.gzip",
        url="https://data.cityofchicago.org/api/geospatial/ewy2-6yfk?method=export&format=Shapefile",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return gdf


def get_raw_cook_county_street_midlines_2015(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    gdf = get_gdf_of_data_portal_data(
        file_name="cook_county_2015_street_midlines.parquet.gzip",
        url="https://datacatalog.cookcountyil.gov/api/geospatial/73aw-3v3w?method=export&format=Shapefile",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return gdf


def get_raw_cook_county_gis_streets(
    raw_file_path: Union[str, None] = None, force_repull: bool = False
) -> gpd.GeoDataFrame:
    """Data Documentation page:
    https://hub-cookcountyil.opendata.arcgis.com/datasets/\
    4569d77e6d004c0ea5fada54640189cf_5/about
    """
    gdf = get_gdf_of_data_portal_data(
        file_name="cook_county_gis_streets.parquet.gzip",
        url="https://opendata.arcgis.com/api/v3/datasets/4569d77e6d004c0ea5fada54640189cf_5/downloads/data?format=shp&spatialRefId=3435",
        raw_file_path=raw_file_path,
        force_repull=force_repull,
    )
    return gdf
