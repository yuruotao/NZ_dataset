# Coding: utf-8
# Script for weather data processing
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
import geopandas as gpd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from basic_statistics import basic_statistics

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'

Base = declarative_base()

# SH RS RP siteRef lane type HEAVY_RATIO DESCRIPTION region siteType LAT LON
class filtered_flow_meta(Base):
    __tablename__ = 'filtered_flow_meta'
    STATION_ID = Column(String, primary_key=True, unique=True, nullable=False)
    SH = Column(Integer)
    RS = Column(Integer)
    RP = Column(String)
    SITEREF = Column(String)
    LANE = Column(Integer)
    TYPE = Column(String)
    HEAVY_RATIO = Column(Integer)
    DESCRIPTION = Column(String)
    REGION = Column(String)
    SITETYPE = Column(String)
    LAT = Column(Float)
    LON = Column(Float)

class basic_statistics_sql_class(Base):
    __tablename__ = 'basic_statistics_flow'
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

class filtered_siteref_base(Base):
    __tablename__ = 'filtered_flow'
    ID = Column(Integer, primary_key=True, unique=True, nullable=False)
    SITEREF = Column(String)
    DATETIME = Column(DateTime)
    FLOW = Column(Float)
    WEIGHT = Column(String)
    DIRECTION = Column(Integer)

def traffic_flow_import_19(input_path, siteRef_list, engine, commit_flag):
    """Import traffic flow data of 2019
    https://opendata-nzta.opendata.arcgis.com/datasets/tms-traffic-quarter-hourly-oct-2020-to-jan-2022/about
    https://opendata-nzta.opendata.arcgis.com/datasets/b90f8908910f44a493c6501c3565ed2d_0

    Args:
        input_path (string): path of traffic flow between 2019
        siteRef_list (list): list contain strings of siteRef
        engine (sqlalchemy_engine): engine used for database creation

    Returns:
        None
    """

    # Read all files within the folder
    traffic_flow_list = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    df = pd.DataFrame()
    for file in traffic_flow_list:
        print(file)
        if "2019" in file:
            temp_df = pd.read_csv(input_path + file, encoding='unicode_escape')
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

def lineplot_breaknans(data, break_at_nan=True, break_at_inf=True, **kwargs):
    '''sns.lineplot by default doesn't break the line at nans or infs, 
        which can lead to misleading plots.
    See https://github.com/mwaskom/seaborn/issues/1552 
        and https://stackoverflow.com/questions/52098537/avoid-plotting-missing-values-on-a-line-plot
    
    This function rectifies this, and allows the user to specify 
        if it should break the line at nans, infs, or at both (default).
    
    Note: using this function means you can't use the `units` argument of sns.lineplot.'''

    cum_num_nans_infs = np.zeros(len(data))
    if break_at_nan: cum_num_nans_infs += np.cumsum(np.isnan(data[kwargs['y']]))
    if break_at_inf: cum_num_nans_infs += np.cumsum(np.isinf(data[kwargs['y']]))
    sns.lineplot(data, **kwargs, units=cum_num_nans_infs, 
            estimator=None)  #estimator must be None when specifying units

def direction_plot(siteRef_list, flow_df, output_path):
    
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    flow_df = flow_df.astype({"DIRECTION":int})
        
    time_index = pd.date_range(start="2021-01-01 00:00:00", end="2021-12-31 23:45:00", freq="15min")
    datetime_df = pd.DataFrame()
    datetime_df["DATETIME"] = time_index
        
    for siteRef in siteRef_list:
        temp_flow_df = flow_df[flow_df["SITEREF"] == siteRef]
        dir_set = set(temp_flow_df["DIRECTION"].to_list())
        if len(dir_set) >= 2:
            print("Direction number:", len(dir_set))
        
            temp_merged_df = datetime_df
            for direction in dir_set:
                temp_dir_df = temp_flow_df[temp_flow_df["DIRECTION"] == direction]
                temp_dir_df = temp_dir_df[["DATETIME", "FLOW"]]
                temp_merged_df = temp_merged_df.merge(temp_dir_df, on='DATETIME', how='left')
                temp_merged_df = temp_merged_df.fillna(value=np.nan)
                temp_merged_df.replace(to_replace=[None], value=np.nan, inplace=True)
                temp_merged_df = temp_merged_df.rename({"FLOW":str(direction)}, axis=1)

            temp_merged_df = temp_merged_df.reset_index(drop=True)
            print(temp_merged_df)
            
            color_list = ["#669bbc", "#003049", "#780000", "#0466c8", "#d90429",  "#0353a4" ]

            column_names = list(temp_merged_df.columns)
            column_names.remove('DATETIME')
            subplot_num = len(dir_set)
            
            matplotlib.rc('xtick', labelsize=22)
            matplotlib.rc('ytick', labelsize=22)
            plt.rc('legend', fontsize=22)
            
            done = False
            while not done:
                try:
                    fig, axes = plt.subplots(subplot_num, 1, figsize=(16, 12))
                    done = True
                except ValueError:
                    continue

            for iter in range(subplot_num):
                lineplot_breaknans(data=temp_merged_df, x='DATETIME', y=column_names[iter], ax=axes[iter], color=color_list[iter], break_at_nan=True, break_at_inf=True)
                axes[iter].set_title("Direction " + column_names[iter], fontsize=22)

                n = 1500  # Set the desired frequency of ticks
                ticks_list = list(temp_merged_df["DATETIME"])
                ticks = ticks_list[::n]
                axes[iter].set_xticks(ticks)
                axes[iter].set(xlabel="", ylabel="")
                axes[iter].set_xlim(temp_merged_df.DATETIME.min(), temp_merged_df.DATETIME.max())
                axes[iter].set_ylim(bottom=0)
                    
            # Hide xticks for all subplots except the bottom one
            for ax in axes[:-1]:
                ax.xaxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=False)

            # Display xticks only at the bottom subplot
            axes[-1].xaxis.set_ticks_position('both')  # Display ticks on both sides
            axes[-1].xaxis.set_tick_params(which='both', bottom=True, top=False)  # Only bottom ticks are visible
            
            # Rotate the x-tick labels for better readability
            plt.xticks(rotation=45)

            # Adjust layout
            plt.tight_layout()
            # Show the plot
            plt.savefig(output_path + siteRef + ".png", dpi=600)
            plt.close()
        
    
    return None

def flow_missing_data_visualization(input_df, output_path):
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
    
    temp_flow_df = input_df.set_index("DATETIME")
    
    # Divide into chunks
    chunks = [temp_flow_df.iloc[:, i:i+20] for i in range(0, len(temp_flow_df.columns), 20)]
    
    # Missing data visualization
    index = 0
    for chunk in chunks:
        print(index)
        # Matrix plot
        ax = msno.matrix(chunk, fontsize=20, figsize=(20, 16), label_rotation=45, freq="6M")
        plt.xlabel("Flow Monitoring Sites", fontsize=20)
        plt.ylabel("Sample Points", fontsize=20)
        plt.savefig(missing_matrix_path + 'matrix_' + str(index) + '.png', dpi=600)
        plt.close()
        
        # Bar plot
        ax = msno.bar(chunk, fontsize=20, figsize=(20, 16), label_rotation=45)
        plt.xlabel("Flow Monitoring Sites", fontsize=20)
        plt.ylabel("Sample Points", fontsize=20)
        plt.savefig(missing_bar_path + 'bar_' + str(index) + '.png', dpi=600)
        plt.close()
        
        index = index + 1
        
    return None

def flow_missing_filter(meta_df, merged_df, threshold, engine):
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

    filtered_meta_df = meta_df[meta_df['SITEREF'].isin(stations_higher_than_threshold)].reset_index(drop=True)
    filtered_meta_df.to_sql('filtered_flow_meta', con=engine, if_exists='replace', index=False)

    return filtered_meta_df

def flow_data_imputation(filtered_meta_df, merged_df, engine):
    """Reformat and impute the missing data of weather data
    Add relative humidity "RH" to the dataframe

    Args:
        data_path (string): path to station data
        output_path (string): path to store the imputed data

    Returns:
        None
    """

    siteref_list = filtered_meta_df["SITEREF"].to_list()
    
    for siteref in siteref_list:
        print(siteref)
        temp_df = merged_df[merged_df["SITEREF"] == siteref]
        datetime_column = temp_df["DATETIME"]
        siteref_column = temp_df["SITEREF"]
        temp_df = temp_df.drop(columns=["DATETIME", "ID", "SITEREF"])
        basic_statistics_df = basic_statistics(temp_df)
        basic_statistics_df["INDEX"] = str(siteref)

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
        temp_df["SITEREF"] = siteref_column
        
        temp_df.to_sql('filtered_flow', con=engine, if_exists='append', index=False)
        basic_statistics_df.to_sql('basic_statistics_flow', con=engine, if_exists='append', index=False)

    return None

if __name__ == "__main__":
    process_db_address = 'sqlite:///./data/NZDB_flow_process.db'
    process_engine = create_engine(process_db_address)
    Base.metadata.create_all(process_engine)
    
    db_address = 'sqlite:///./data/NZDB/NZDB.db'
    engine = create_engine(db_address)
    
    flow_meta_query = 'SELECT * FROM flow_meta'
    flow_meta_df = pd.read_sql(flow_meta_query, engine)
    flow_meta_df.to_sql('filtered_flow_meta', con=process_engine, if_exists='replace', index=False)

    # If RAM is huge, select from the established database
    #flow_query = "SELECT * FROM flow WHERE strftime('%Y', DATETIME) IN ('2021')"
    #flow_df = pd.read_sql(flow_query, engine)
    #flow_df = flow_df.astype({"DATETIME":"datetime64[ns]"})
    #print(flow_df)
    
    # If RAM is limited, create a new database with only desired time range
    siteRef_list = flow_meta_df["SITEREF"].to_list()
    flow_df = traffic_flow_import_19("./data/traffic/flow_data_13_20/", siteRef_list, process_engine, True)
    
    
    
    # Plot the comparison between directions for each station
    #direction_plot(siteRef_list, heavy_flow_df, "./result/traffic/direction/heavy/")
    #direction_plot(siteRef_list, light_flow_df, "./result/traffic/direction/light/")
    
    
        

    """
    temp_flow_df = flow_df[["DATETIME", "SITEREF", "FLOW"]]
    
    time_index = pd.date_range(start="2021-01-01", end="2021-12-31", freq="D")
    datetime_df = pd.DataFrame()
    datetime_df["DATETIME"] = time_index
    
    pivot_df = temp_flow_df.pivot(index='DATETIME', columns='SITEREF', values='FLOW')
    merged_df = datetime_df.merge(pivot_df, on='DATETIME', how='left')
    
    # Visualize the missing data
    flow_missing_data_visualization(merged_df, "./result/flow/missing")
    
    # Filter based on missing value percentage
    Session = sessionmaker(bind=process_engine)
    session = Session()
    
    filtered_meta_df = flow_missing_filter(flow_meta_df, merged_df, 30, process_engine)
    flow_data_imputation(filtered_meta_df, flow_df, process_engine)
    """
    
    # Light and Heavy
    
    
    # Region
    
    
    # Weekday