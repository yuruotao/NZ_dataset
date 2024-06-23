# Coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from os import listdir
from os.path import isfile, join
import time

from utils.basic_statistics import basic_statistics

if __name__ == "__main__":
    # Flow
    # Calculate the basic statistics for each flow station
    #basic_statistics()
    
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
        basic_statistics(temp_df, "./result/weather/basic_statistics/" + path.split("/")[-1].strip(".xlsx") + "/")
    
    
    ###########################################################