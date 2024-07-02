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



if __name__ == "__main__":
    # Connect to database
    flow_process_db_address = 'sqlite:///./data/NZDB_flow_process.db'
    flow_process_engine = create_engine(flow_process_db_address)
    
    weather_process_db_address = 'sqlite:///./data/NZDB_weather_process.db'
    weather_process_engine = create_engine(weather_process_db_address)
    
    # Import data
    flow_meta_query = 'SELECT * FROM filtered_flow_meta'
    flow_meta_df = pd.read_sql(flow_meta_query, flow_process_engine)
    
    weather_meta_query = 'SELECT * FROM filtered_weather_meta'
    weather_meta_df = pd.read_sql(weather_meta_query, weather_process_engine)
    
    
    
    
    
    
    # Calculate the nearest flow point to weather point
    
    
    
    
    ###########################################################
    # Extreme weather
    
    ###########################################################
    # Holiday
    # 