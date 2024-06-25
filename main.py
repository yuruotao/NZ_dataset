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
    process_db_address = 'sqlite:///./NZDB_process.db'
    process_engine = create_engine(process_db_address)
    
    # Flow

    
    # Light and Heavy
    
    
    # Region
    
    
    # Weekday
    
    ###########################################################
    
    # Weather
    # Calculate the basic statistics for each weather station
    filtered_weather_path = "./result/weather/stations_imputed/"
    filtered_weather_list = [filtered_weather_path + f for f in listdir(filtered_weather_path) if isfile(join(filtered_weather_path, f))]
    for path in filtered_weather_list:
        print(path)
        temp_df = pd.read_excel(path)
        basic_statistics(temp_df, "./result/weather/filtered_basic_statistics/" + path.split("/")[-1].strip(".xlsx") + "/")
    
    
    ###########################################################