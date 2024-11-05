# Coding: utf-8
# Script for flow data processing
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib
import requests
from os import listdir
from os.path import isfile, join
import missingno as msno
import seaborn as sns
import matplotlib as mpl
import matplotlib.dates as mdates
import geopandas as gpd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from matplotlib.ticker import MaxNLocator, FuncFormatter
from shapely.geometry import Point
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree
from basic_statistics import basic_statistics

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'

Base = declarative_base()

def traffic_flow_import_19(input_path, siteRef_list, engine, commit_flag):
    """Import traffic flow data of 2019
    https://opendata-nzta.opendata.arcgis.com/datasets/tms-traffic-quarter-hourly-oct-2020-to-jan-2022/about
    https://opendata-nzta.opendata.arcgis.com/datasets/b90f8908910f44a493c6501c3565ed2d_0

    Args:
        input_path (string): path of traffic flow between 2019
        siteRef_list (list): list contain strings of siteRef
        engine (sqlalchemy_engine): engine used for database creation
        commit_flag (bool): determine whether to upload the data to database

    Returns:
        dataframe
    """

    # Read all files within the folder
    traffic_flow_list = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    df = pd.DataFrame()
    for file in traffic_flow_list:
        print(file)
        if "2019" in file:
            temp_df = pd.read_csv(input_path + file)
            # class siteRef startDatetime endDatetime direction count
            temp_df["siteRef"] = temp_df["siteRef"].apply(lambda x: str(x).zfill(8))
            temp_df = temp_df[temp_df["siteRef"].isin(siteRef_list)]
            temp_df = temp_df.rename(columns={"startDatetime":"Datetime", "count":"Flow", 
                                            "class":"Weight", "direction":"Direction"})
            
            temp_df = temp_df[["Datetime", "siteRef", "Flow", "Weight", "Direction"]]
            temp_df['Datetime'] = pd.to_datetime(temp_df['Datetime'], format='%d-%b-%Y %H:%M')
            temp_df['Weight'] = temp_df['Weight'].replace('H', 'Heavy')
            temp_df['Weight'] = temp_df['Weight'].replace('L', 'Light')
            temp_df = temp_df.groupby(["Datetime", "siteRef", "Weight", "Direction"], as_index=False)[["Flow",]].sum().reset_index(drop=True)
            temp_df.columns = temp_df.columns.str.upper()
            
            if commit_flag:
                temp_df.to_sql("filtered_flow", con=engine, if_exists='append', index=False)
            
            df = pd.concat([df, temp_df], axis=0)
            
    return df

    
if __name__ == "__main__":
    process_db_address = 'sqlite:///./data/NZDB_flow_process.db'
    process_engine = create_engine(process_db_address)
    Base.metadata.create_all(process_engine)
    
    db_address = 'sqlite:///./data/NZDB/NZDB.db'
    engine = create_engine(db_address)
    
    flow_meta_query = 'SELECT * FROM flow_meta'
    flow_meta_df = pd.read_sql(flow_meta_query, engine)
    
    time_index = pd.date_range(start="2019-01-01 00:00:00", end="2019-12-31 23:45:00", freq="15min")
    datetime_df = pd.DataFrame()
    datetime_df["DATETIME"] = time_index
    """
    Auckland_list = flow_meta_df["SITEREF"].to_list()

    flow_df = traffic_flow_import_19("./data/traffic/flow_data_13_20/", Auckland_list, process_engine, False)
    flow_df = flow_df.astype({"DATETIME":"datetime64[ms]"})

    total_flow = flow_df.groupby(['SITEREF', 'DATETIME', 'WEIGHT'])['FLOW'].sum().reset_index()
    total_flow = total_flow.rename(columns={'FLOW': 'TOTAL_FLOW'})
    
    proportion_flow_df = pd.merge(flow_df, total_flow, on=['SITEREF', 'DATETIME', 'WEIGHT'])
    proportion_flow_df['PROPORTION'] = proportion_flow_df['FLOW'] / proportion_flow_df['TOTAL_FLOW']
    
    # Filter out the rows with proportion higher than 50%, row PROPORTION is the imbalance
    proportion_flow_df = proportion_flow_df[proportion_flow_df['PROPORTION'] > 0.5]
    proportion_flow_df = proportion_flow_df.drop(columns=['DIRECTION'])
    
    # Select by weight
    light_df = proportion_flow_df[proportion_flow_df['WEIGHT'] == "Light"]
    temp_light_flow_df = light_df[["DATETIME", "SITEREF", "FLOW"]]
    
    light_pivot_df = temp_light_flow_df.pivot(index='DATETIME', columns='SITEREF', values='FLOW')
    light_merged_df = datetime_df.merge(light_pivot_df, on='DATETIME', how='left')

    Wellington_summary = basic_statistics(light_merged_df)
    Wellington_summary = Wellington_summary[Wellington_summary["PERCENTAGE_MISSING"] <= 30]
    
    Wellington_summary.to_excel("./summary.xlsx", index=False)
    """
    summary_df = pd.read_excel("./summary.xlsx")
    print(summary_df)
    print(flow_meta_df)
    
    merged_df = summary_df.merge(flow_meta_df, left_on='INDICATOR', right_on='SITEREF', how='left')

    # Add the 'REGION' column to summary_df and rename it to 'city'
    summary_df['city'] = merged_df['REGION']
    summary_df["city"] = summary_df["city"].str.split(' - ').str.get(-1)
    summary_df.to_excel("./summary.xlsx", index=False)