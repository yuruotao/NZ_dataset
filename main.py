# Coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

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
    weather_meta = pd.read_excel("./data/weather/weather_meta.xlsx")
    weather_station_list = weather_meta["Station_ID"].to_list()
    print(weather_meta)
    
    #basic_statistics(, "./result/weather/basic_statistics/" +  + "/")
    
    
    
    ###########################################################