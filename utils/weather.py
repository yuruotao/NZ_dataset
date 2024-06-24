# Coding: utf-8
# Script for weather data processing
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

def weather_df_to_gdf(input_path, output_path, epsg):
    """Project the traffic data to geodataframe

    Args:
        input_path (string): path to processed data
        output_path (string): path to save the geodataframe
        epsg (string): the coordinate system

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    df = pd.read_excel(input_path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LON, df.LAT), crs='EPSG:' + epsg)
    gdf.to_file(output_path + "weather_stations.shp")
    
    return None


def weather_missing_data_visualization(input_path, output_path):
    """Visualize the missing data of flow data

    Args:
        input_path (string): path to the raw data
        output_path (string): path to save the figure

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    missing_matrix_path = output_path + "/matrix/"
    missing_bar_path = output_path + "/bar/"
    
    if not os.path.exists(missing_matrix_path):
        os.makedirs(missing_matrix_path)
    
    if not os.path.exists(missing_bar_path):
        os.makedirs(missing_bar_path)
    
    time_index = pd.date_range(start="2013-01-01", end="2021-12-31", freq="D")
    weather_df = pd.DataFrame()
    weather_df["Datetime"] = time_index
    
    dir_list = os.listdir(input_path)
    dir_list = [(input_path + dir) for dir in dir_list]
    
    for dir in dir_list:
        print(dir)
        Station_ID = dir.split("/")[-1].split(".")[0]
        temp_df = pd.read_excel(dir)
        try:
            weather_df[Station_ID] = temp_df["STATION"]
        except:
            weather_df[Station_ID] = np.nan
            
    temp_weather_df = weather_df.set_index("Datetime")
    
    # Divide into chunks
    chunks = [temp_weather_df.iloc[:, i:i+20] for i in range(0, len(temp_weather_df.columns), 20)]
    
    # Missing data visualization
    index = 0
    for chunk in chunks:
        print(index)
        # Matrix plot
        ax = msno.matrix(chunk, fontsize=20, figsize=(20, 16), label_rotation=45, freq="6M")
        plt.xlabel("Weather Monitoring Sites", fontsize=20)
        plt.ylabel("Sample Points", fontsize=20)
        plt.savefig(missing_matrix_path + 'matrix_' + str(index) + '.png', dpi=600)
        plt.close()
        
        # Bar plot
        ax = msno.bar(chunk, fontsize=20, figsize=(20, 16), label_rotation=45)
        plt.xlabel("Weather Monitoring Sites", fontsize=20)
        plt.ylabel("Sample Points", fontsize=20)
        plt.savefig(missing_bar_path + 'bar_' + str(index) + '.png', dpi=600)
        plt.close()
        
        index = index + 1
        
    return None

def traffic_missing_filter(meta_path, input_path, threashold, gpd_output_path, output_path):
    """Delete the stations whose missing data percentage reach the threashold

    Args:
        meta_path (string): xlsx containing the NCDC station data
        input_path (string): path to the raw data
        threashold (float): threashold for deletion
        gpd_output_path (string): path to save the geopandas dataframe
        output_path (string): path to save the filtered raw data

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if not os.path.exists(gpd_output_path):
        os.makedirs(gpd_output_path)
    
    time_index = pd.date_range(start="2013-01-01", end="2021-12-31", freq="D")
    weather_df = pd.DataFrame()
    weather_df["Datetime"] = time_index
    
    dir_list = os.listdir(input_path)
    dir_list = [(input_path + dir) for dir in dir_list]
    
    for dir in dir_list:
        Station_ID = dir.split("/")[-1].split(".")[0]
        print(Station_ID)
        temp_df = pd.read_excel(dir)
        try:
            weather_df[Station_ID] = temp_df["STATION"]
        except:
            weather_df[Station_ID] = np.nan
    
    # Calculate percentage of missing values in each column
    missing_percentages = weather_df.isna().mean() * 100
    
    # Drop columns where the percentage of missing values exceeds the threshold
    columns_to_drop = missing_percentages[missing_percentages > threashold].index
    processed_df = weather_df.drop(columns=columns_to_drop)
    stations_higher_than_threshold = processed_df.columns.to_list()
    stations_higher_than_threshold.remove("Datetime")
    
    meta_df = pd.read_excel(meta_path)
    meta_df = meta_df.astype({"Station_ID": "str"})
    filtered_meta_df = meta_df[meta_df['Station_ID'].isin(stations_higher_than_threshold)].reset_index(drop=True)
    print(filtered_meta_df)
    filtered_meta_df.to_excel(output_path + "missing_value_filtered_stations.xlsx", index=False)
    gdf = gpd.GeoDataFrame(filtered_meta_df, geometry=gpd.points_from_xy(filtered_meta_df.LON, filtered_meta_df.LAT), crs='EPSG:' + "4167")
    gdf.to_file(gpd_output_path + "weather_stations_filtered.shp")
    
    return None

def NCDC_weather_data_imputation(filtered_meta_df_path, data_path, output_path):
    """Reformat and impute the missing data of weather data
    Add relative humidity "RH" to the dataframe

    Args:
        data_path (string): path to station data
        output_path (string): path to store the imputed data

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    filtered_meta_df = pd.read_excel(filtered_meta_df_path)
    station_list = filtered_meta_df["Station_ID"].to_list()
    
    station_files = [str(station) + ".xlsx" for station in station_list]
    for station_file in station_files:
        print(station_file)
        temp_df = pd.read_excel(data_path + station_file)
        temp_df = temp_df[["Datetime",
                           "TEMP", "DEWP", "SLP", "STP", "VISIB", 
                           "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", 
                           "SNDP"]]
        
        start_date = '2013-01-01'
        end_date = '2021-12-31'
        temp_df = temp_df[(temp_df['Datetime'] >= start_date) & (temp_df['Datetime'] <= end_date)]
        
        # Missing data
        temp_df.replace(99.99, np.nan, inplace=True)
        temp_df.replace(999.9, np.nan, inplace=True)
        temp_df.replace(9999.9, np.nan, inplace=True)
        
        # Degree to Celsius
        temp_df['TEMP'] = temp_df.apply(lambda x: (x['TEMP']-32)*(5/9), axis=1)
        temp_df['MAX'] = temp_df.apply(lambda x: (x['MAX']-32)*(5/9), axis=1)
        temp_df['MIN'] = temp_df.apply(lambda x: (x['MIN']-32)*(5/9), axis=1)
        temp_df['DEWP'] = temp_df.apply(lambda x: (x['DEWP']-32)*(5/9), axis=1)
        
        # Dew point to relative humidity
        def calculate_relative_humidity(dew_point_celsius, air_temperature_celsius):
            # Calculate saturation vapor pressure at dew point and air temperature
            es_td = 6.112 * np.exp(17.67 * dew_point_celsius / (dew_point_celsius + 243.5))
            es_t = 6.112 * np.exp(17.67 * air_temperature_celsius / (air_temperature_celsius + 243.5))

            # Calculate relative humidity
            relative_humidity = 100 * (es_td / es_t)

            return relative_humidity
        
        # RH for relative humidity
        temp_df['RH'] = temp_df.apply(lambda x: calculate_relative_humidity(x['DEWP'], x['TEMP']), axis=1)
        
        # Millibar to kPa
        temp_df['SLP'] = temp_df.apply(lambda x: x['SLP']/10, axis=1)
        temp_df['STP'] = temp_df.apply(lambda x: x['STP']/10, axis=1)
        
        # Miles to km
        temp_df['VISIB'] = temp_df.apply(lambda x: x['VISIB']*1.609, axis=1)
        
        # Knots to m/s
        temp_df['WDSP'] = temp_df.apply(lambda x: x['WDSP']*0.51444, axis=1)
        temp_df['MXSPD'] = temp_df.apply(lambda x: x['MXSPD']*0.51444, axis=1)
        temp_df['GUST'] = temp_df.apply(lambda x: x['GUST']*0.51444, axis=1)
        
        # Inches to meter
        temp_df['PRCP'] = temp_df.apply(lambda x: x['PRCP']*0.0254, axis=1)
        temp_df['SNDP'] = temp_df.apply(lambda x: x['SNDP']*0.0254, axis=1)
        
        datetime_column = temp_df["Datetime"]
        temp_df = temp_df.drop(columns=["Datetime"])

        forward_df = temp_df.shift(-7*24)
        backward_df = temp_df.shift(7*24)
        
        average_values = (forward_df + backward_df) / 2
        temp_df = temp_df.copy()
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.notna()] = average_values[temp_df.isna() & forward_df.notna() & backward_df.notna()]
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.isna()] = forward_df[temp_df.isna() & forward_df.notna() & backward_df.isna()]
        temp_df[temp_df.isna() & backward_df.notna() & forward_df.isna()] = backward_df[temp_df.isna() & backward_df.notna() & forward_df.isna()]

        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('Datetime', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            mean_value = temp_df[column].mean()
            # Fill NaN values with the mean
            temp_df[column].fillna(mean_value, inplace=True)

        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime"])
        imputed_df = pd.concat([datetime_column, imputed_df], axis=1)

        imputed_df.to_excel(output_path + station_file, index=False)

    return None

if __name__ == "__main__":
    weather_missing_data_visualization("./result/weather/stations/", "./result/weather/missing")
    
    traffic_missing_filter("./data/weather/weather_meta.xlsx", "./result/weather/stations/", 
                           30, "./result/weather/filtered_shp/", "./result/weather/")
    
    NCDC_weather_data_imputation("./result/weather/missing_value_filtered_stations.xlsx",
        "./result/weather/stations/", "./result/weather/stations_imputed/")
