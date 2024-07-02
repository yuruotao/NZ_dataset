# Coding: utf-8
# Script for weather data processing
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

from basic_statistics import basic_statistics

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'

Base = declarative_base()

# CTRY STATE ICAO LAT LON ELEV(M) BEGIN END
class filtered_weather_meta(Base):
    __tablename__ = 'filtered_weather_meta'
    STATION_ID = Column(String, primary_key=True, unique=True, nullable=False)
    USAF = Column(String)
    WBAN = Column(String)
    STATION_NAME = Column(String)
    CTRY = Column(String)
    STATE = Column(String)
    ICAO = Column(String)
    LAT = Column(Float)
    LON = Column(Float)
    ELEV = Column(Float)
    BEGIN = Column(Integer)
    END = Column(Integer)

# Datetime TEMP DEWP SLP STP VISIB WDSP MXSPD GUST MAX MIN PRCP SNDP RH
class filtered_weather(Base):
    __tablename__ = 'filtered_weather'
    ID = Column(Integer, primary_key=True, unique=True, nullable=False)
    STATION_ID = Column(String)
    DATETIME = Column(DateTime)
    TEMP = Column(Float)
    DEWP = Column(Float)
    SLP = Column(Float)
    STP = Column(Float)
    VISIB = Column(Float)
    WDSP = Column(Float)
    MXSPD = Column(Float)
    GUST = Column(Float)
    MAX = Column(Float)
    MIN = Column(Float)
    PRCP = Column(Float)
    SNDP = Column(Float)
    RH = Column(Float)

class basic_statistics_sql_class(Base):
    __tablename__ = 'basic_statistics_weather'
    ID = Column(Integer, primary_key=True, unique=True, nullable=False)
    INDEX = Column(String)
    INDICATOR = Column(String)
    MEAN = Column(Float)
    STD = Column(Float)
    SKEW = Column(Float)
    KURTOSIS = Column(Float)
    PERCENTILE_0 = Column(Float)
    PERCENTILE_2_5 = Column(Float)
    PERCENTILE_50 = Column(Float)
    PERCENTILE_97_5 = Column(Float)
    PERCENTILE_100 = Column(Float)

def weather_missing_data_visualization(input_df, output_path):
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
    
    temp_weather_df = input_df.set_index("DATETIME")
    
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

def weather_missing_filter(meta_df, merged_df, threshold, engine):
    """Delete the stations whose missing data percentage reach the threshold

    Args:
        meta_df (dataframe): dataframe containing the NCDC station meta data
        merged_df (merged_df): raw data merged_df
        threshold (float): threshold for deletion
        engine (sqlalchemy_engine): engine to save the filtered meta dataframe

    Returns:
        None
    """

    
    # Calculate percentage of missing values in each column
    missing_percentages = merged_df.isna().mean() * 100
    
    # Drop columns where the percentage of missing values exceeds the threshold
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    processed_df = merged_df.drop(columns=columns_to_drop)
    stations_higher_than_threshold = processed_df.columns.to_list()
    stations_higher_than_threshold.remove("DATETIME")

    filtered_meta_df = meta_df[meta_df['STATION_ID'].isin(stations_higher_than_threshold)].reset_index(drop=True)
    filtered_meta_df.to_sql('filtered_weather_meta', con=engine, if_exists='replace', index=False)

    return filtered_meta_df

def NCDC_weather_data_imputation(filtered_meta_df, merged_df, engine):
    """Reformat and impute the missing data of weather data
    Add relative humidity "RH" to the dataframe

    Args:
        data_path (string): path to station data
        output_path (string): path to store the imputed data

    Returns:
        None
    """

    station_id_list = filtered_meta_df["STATION_ID"].to_list()
    
    for station_id in station_id_list:
        print(station_id)
        temp_df = merged_df[merged_df["STATION_ID"] == station_id]
        datetime_column = temp_df["DATETIME"]
        station_id_column = temp_df["STATION_ID"]
        temp_df = temp_df.drop(columns=["DATETIME", "ID", "STATION_ID"])
        basic_statistics_df = basic_statistics(temp_df)
        basic_statistics_df["INDEX"] = str(station_id)

        forward_df = temp_df.shift(-1)
        backward_df = temp_df.shift(1)
        average_values = (forward_df + backward_df) / 2
        
        temp_df = temp_df.copy()
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.notna()] = average_values[temp_df.isna() & forward_df.notna() & backward_df.notna()]
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.isna()] = forward_df[temp_df.isna() & forward_df.notna() & backward_df.isna()]
        temp_df[temp_df.isna() & backward_df.notna() & forward_df.isna()] = backward_df[temp_df.isna() & backward_df.notna() & forward_df.isna()]

        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('DATETIME', inplace=True)
        # Set Datetime column as index
        for column in temp_df.columns:
            mean_value = temp_df[column].mean()
            # Fill NaN values with the mean
            temp_df[column].fillna(mean_value, inplace=True)

        temp_df = temp_df.reset_index()
        temp_df["STATION_ID"] = station_id_column
        
        temp_df.to_sql('filtered_weather', con=engine, if_exists='append', index=False)
        basic_statistics_df.to_sql('basic_statistics_weather', con=engine, if_exists='append', index=False)

    return None

if __name__ == "__main__":
    process_db_address = 'sqlite:///./data/NZDB_weather_process.db'
    process_engine = create_engine(process_db_address)
    Base.metadata.create_all(process_engine)
    
    db_address = 'sqlite:///./data/NZDB/NZDB.db'
    engine = create_engine(db_address)
    
    weather_meta_query = 'SELECT * FROM weather_meta'
    weather_meta_df = pd.read_sql(weather_meta_query, engine)

    weather_query = 'SELECT * FROM weather'
    weather_df = pd.read_sql(weather_query, engine)
    weather_df = weather_df.astype({"DATETIME":"datetime64[ns]"})
    temp_weather_df = weather_df[["DATETIME", "STATION_ID", "TEMP"]]
    
    time_index = pd.date_range(start="2013-01-01", end="2021-12-31", freq="D")
    datetime_df = pd.DataFrame()
    datetime_df["DATETIME"] = time_index
    
    pivot_df = temp_weather_df.pivot(index='DATETIME', columns='STATION_ID', values='TEMP')
    merged_df = datetime_df.merge(pivot_df, on='DATETIME', how='left')
    
    # Visualize the missing data
    #weather_missing_data_visualization(merged_df, "./result/weather/missing")
    
    # Filter based on missing value percentage
    Session = sessionmaker(bind=process_engine)
    session = Session()
    
    #filtered_meta_df = weather_missing_filter(weather_meta_df, merged_df, 30, process_engine)
    #NCDC_weather_data_imputation(filtered_meta_df, weather_df, process_engine)
    
    # Output a dataframe of basic statistics
    basic_statistics_weather_query = 'SELECT * FROM basic_statistics_weather'
    basic_statistics_weather_df = pd.read_sql(basic_statistics_weather_query, process_engine)
    basic_statistics_weather_df = basic_statistics_weather_df[basic_statistics_weather_df["INDEX"] == "93004099999"]
    basic_statistics_weather_df = basic_statistics_weather_df[~basic_statistics_weather_df["INDICATOR"].isin(["VISIB", "GUST", "SNDP"])]
    basic_statistics_weather_df = basic_statistics_weather_df.round(2)
    basic_statistics_weather_df.to_excel("./result/weather/basic_statistics.xlsx", index=False)
    
    #session.commit()
    #session.close()
    
    
    