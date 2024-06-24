# Coding: utf-8
# Script for weather data obtain
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import gdelt
import requests
from os import listdir
from os.path import isfile, join
import missingno as msno
import seaborn as sns
import matplotlib as mpl
import geopandas as gpd

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'

def NCDC_weather_data_obtain(meta_path, output_path, start_year, stop_year):
    """Obtain the weather data from NCDC

    Args:
        meta_path (string): xlsx containing the NCDC station data
        output_path (string): folder to contain the weather data
        start_year (int): the start year of data
        stop_year (int): the stop year of data

    Returns:
        None
    """
    # meta_df is obtained from
    # https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/
    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df.loc[meta_df["CTRY"]== "NZ"]
    meta_df = meta_df.astype({"BEGIN":int, "END":int, "USAF":str, "WBAN":str})
    meta_df = meta_df.loc[meta_df["END"]>stop_year*10000]
    meta_df["WBAN"] = meta_df["WBAN"].str.zfill(5)
    station_str = meta_df["USAF"] + meta_df["WBAN"]
    meta_df["Station_ID"] = station_str
    meta_df = meta_df.reset_index(drop=True)
    print(meta_df)
    meta_df.to_excel(output_path + "weather_meta.xlsx", index=False)

    # download the data with respect to "USAF"
    base = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"
    for year in range(start_year, stop_year + 1):
        base_year = base + str(year) + "/"
        output_year = output_path + str(year) + "/"
        if not os.path.exists(output_year):
            os.makedirs(output_year)
        print("Year", year)
        
        for index, row in meta_df.iterrows():
            print(row["Station_ID"])
            url = base_year + row["Station_ID"] + ".csv"
            response = requests.get(url)
            print(response)
            csv_data = response.text
            
            filename = output_year + row["Station_ID"] + ".csv"
            with open(filename, 'w') as f:
                f.write(csv_data)
    return None

def NCDC_weather_data_station_merge(meta_path, 
                                    input_path, output_path, 
                                    start_year, stop_year):
    """Merge the collected data into single station files

    Args:
        meta_path (string): xlsx containing the NCDC station data
        input_path (string): folder to contain the weather data
        output_path (string): folder to save the station weather data
        start_year (int): the start year of data
        stop_year (int): the stop year of data

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    time_index = pd.date_range(start="2013-01-01", end="2022-12-31", freq="D")
    meta_df = pd.read_excel(meta_path)
    
    for index, row in meta_df.iterrows():
        Station_ID = str(row["Station_ID"])
        print(Station_ID)
        
        # For each station
        year_df = pd.DataFrame()
        temp_time_index = pd.DataFrame()
        temp_time_index["Datetime"] = time_index
        for year in range(start_year, stop_year):
            
            temp_path = input_path + str(year) + "/" + Station_ID + ".csv"
            temp_df = pd.read_csv(temp_path)
            
            if temp_df.shape[0] <= 10:
                pass
            else:
                temp_df = temp_df.astype({"DATE":"datetime64[ns]"})
                if year == start_year:
                    year_df = temp_df
                else:
                    year_df = pd.concat([year_df, temp_df], ignore_index=True)
        try:
            output_df = pd.merge(temp_time_index, year_df, left_on="Datetime", right_on="DATE", how="left")
        except:
            output_df = temp_time_index
            
        output_df.to_excel(output_path + Station_ID + ".xlsx", index=False)

    return None

if __name__ == "__main__":
    # Obtain the NCDC data
    NCDC_weather_data_obtain("./data/isd-history.csv", "./data/weather/", 2013, 2022)

    # Merge the data among years and save for each station
    NCDC_weather_data_station_merge("./data/weather/weather_meta.xlsx",
                                    "./data/weather/", 
                                    "./result/weather/stations/",
                                    2013, 2022)                      
