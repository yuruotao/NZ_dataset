# Coding: utf-8
# Main analysis
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import requests
from os import listdir
from os.path import isfile, join
import missingno as msno
import seaborn as sns
import matplotlib as mpl
import geopandas as gpd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from utils.visualization import *
from utils.flow_process import *


if __name__ == "__main__":
    # Plot the distribution
    """
    raw_db_address = 'sqlite:///./data/NZDB/NZDB.db'
    raw_db_engine = create_engine(raw_db_address)
    raw_flow_meta_query = 'SELECT * FROM flow_meta'
    raw_flow_meta_df = pd.read_sql(raw_flow_meta_query, raw_db_engine)
    raw_weather_meta_query = 'SELECT * FROM weather_meta'
    raw_weather_meta_df = pd.read_sql(raw_weather_meta_query, raw_db_engine)
    
    # Distribution visualization
    distribution_visualization(raw_flow_meta_df, raw_weather_meta_df, 
                               highway_shp_path="./data/state_highway/state_highway.shp",
                               boundary_shp_path="./data/boundary/city_districts/city_districts.shp",
                               output_path="./result/distribution/")
    """
    ####################################################################################################
    # Import data from missing filtered database
    # Connect to database
    flow_process_db_address = 'sqlite:///./data/NZDB_flow_process.db'
    flow_process_engine = create_engine(flow_process_db_address)
    flow_meta_query = 'SELECT * FROM filtered_flow_meta'
    flow_meta_df = pd.read_sql(flow_meta_query, flow_process_engine)

    weather_process_db_address = 'sqlite:///./data/NZDB_weather_process.db'
    weather_process_engine = create_engine(weather_process_db_address)
    weather_meta_query = 'SELECT * FROM filtered_weather_meta'
    weather_meta_df = pd.read_sql(weather_meta_query, weather_process_engine)
    
    flow_meta_gdf = df_to_gdf(flow_meta_df, "LON", "LAT")
    weather_meta_gdf = df_to_gdf(weather_meta_df, "LON", "LAT")
    
    # Obtain the shape of 3 cities for analysis
    # 047-Wellington 060-Christchurch 076-Auckland
    boundary_shp = gpd.read_file("./data/boundary/city_districts/city_districts.shp")
    Wellington_shp = boundary_shp[boundary_shp["TA2023_V1_"] == "047"]
    Christchurch_shp = boundary_shp[boundary_shp["TA2023_V1_"] == "060"]
    Auckland_shp = boundary_shp[boundary_shp["TA2023_V1_"] == "076"]
    
    ####################################################################################################
    # Weather correlation
    
    flow_df_query = 'SELECT * FROM filtered_flow'
    flow_df = pd.read_sql(flow_df_query, flow_process_engine)
    
    weather_df_query = 'SELECT * FROM filtered_weather'
    weather_df = pd.read_sql(weather_df_query, weather_process_engine)

    

    