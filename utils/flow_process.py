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
    TOTAL_FLOW = Column(Float)
    PROPORTION = Column(Float)

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

def lineplot_breaknans(data, break_at_nan=True, break_at_inf=True, **kwargs):
    """Make lineplot break at nans

    Args:
        data (dataframe): data to be plotted
        break_at_nan (bool, optional): whether to break at nan. Defaults to True.
        break_at_inf (bool, optional): whether to break at inf. Defaults to True.

    Raises:
        ValueError: dataframe do not contain any column

    Returns:
        ax
    """
    
    # Automatically detect the y column and use index as x
    if 'y' not in kwargs:
        columns = data.columns
        if len(columns) >= 1:
            kwargs['y'] = columns[0]
        else:
            raise ValueError("DataFrame must contain at least one column for y detection.")
    
    # Reset index to have a column for the x-axis
    data_reset = data.reset_index()
    kwargs['x'] = data_reset.columns[0]

    # Create a cumulative sum of NaNs and infs to use as units
    cum_num_nans_infs = np.zeros(len(data_reset))
    if break_at_nan: cum_num_nans_infs += np.cumsum(np.isnan(data_reset[kwargs['y']]))
    if break_at_inf: cum_num_nans_infs += np.cumsum(np.isinf(data_reset[kwargs['y']]))

    # Plot using seaborn's lineplot
    ax = sns.lineplot(data=data_reset, **kwargs, units=cum_num_nans_infs, estimator=None)  # estimator must be None when specifying units
    return ax

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
    chunks = [temp_flow_df.iloc[:, i:i+15] for i in range(0, len(temp_flow_df.columns), 15)]
    
    # Missing data visualization
    index = 0
    for chunk in chunks:
        print(index)
        # Matrix plot
        ax = msno.matrix(chunk, fontsize=10.5, figsize=(8, 6), label_rotation=45, freq="1M")
        ax.tick_params(labelsize=10.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.xlabel("Traffic flow monitoring sites", fontsize=10.5)
        plt.savefig(missing_matrix_path + 'matrix_' + str(index) + '.svg', format="svg", dpi=1200)
        plt.close()
        
        # Bar plot
        ax = msno.bar(chunk, fontsize=10.5, figsize=(8, 6), label_rotation=45)
        ax.tick_params(labelsize=10.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.xlabel("Traffic flow monitoring sites", fontsize=10.5)
        plt.savefig(missing_bar_path + 'bar_' + str(index) + '.svg', format="svg", dpi=1200)
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

def flow_data_imputation(filtered_meta_df, merged_df, engine, commit_flag):
    """impute the missing data

    Args:
        filtered_meta_df (dataframe): meta data of flow stations
        merged_df (dataframe): flow data
        engine (sqlalchemy_engine): engine to save the imputed flow data
        commit_flag (bool): whether to commit data to the database

    Returns:
        imputed dataframe
    """
    time_index = pd.date_range(start="2019-01-01 00:00:00", end="2019-12-31 23:45:00", freq="15min")
    datetime_df = pd.DataFrame()
    datetime_df["DATETIME"] = time_index
    
    siteref_list = filtered_meta_df["SITEREF"].to_list()
    imputed_df = pd.DataFrame()
    for siteref in siteref_list:
        print(siteref)
        temp_df = merged_df[merged_df["SITEREF"] == siteref]
        temp_df = pd.merge(datetime_df, temp_df, on="DATETIME", how="left")
        
        datetime_column = temp_df["DATETIME"]
        try:
            temp_df = temp_df.drop(columns=["DATETIME", "SITEREF", "WEIGHT", "FLOW"])
        except:
            temp_df = temp_df.drop(columns=["DATETIME", "SITEREF"])
        basic_statistics_df = basic_statistics(temp_df)
        basic_statistics_df["INDEX"] = str(siteref)

        forward_df = temp_df.shift(-24*4*7)
        backward_df = temp_df.shift(24*4*7)
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
        temp_df["SITEREF"] = siteref

        if commit_flag == True:
            temp_df.to_sql('filtered_flow', con=engine, if_exists='append', index=False)
            basic_statistics_df.to_sql('basic_statistics_flow', con=engine, if_exists='append', index=False)
        
        imputed_df = pd.concat([imputed_df, temp_df], axis=0)

    return imputed_df

def imputation(input_df, imputation_method, save_path):
    """carry out the imputation for raw data with missing values

    Args:
        input_df (dataframe): the dataframe containing raw data
        imputation_method (string): specify the method of imputation
        save_path (string): specify the folder to save the imputed data

    Returns:
        dataframe: dataframe containing the imputed data
    """
    
    print("Imputation begin")
    datetime_column = input_df["DATETIME"]
    input_df = input_df.drop(columns=["DATETIME"])
    
    imputation_dir = save_path + "/"
    
    if not os.path.exists(imputation_dir):
        os.makedirs(imputation_dir)
        
    if imputation_method == "Linear":
        imputed_df = input_df.interpolate(method='linear')

    elif imputation_method == "Forward-Backward":
        forward_df = input_df.shift(-4*24*7)
        backward_df = input_df.shift(4*24*7)
        
        average_values = (forward_df + backward_df) / 2
        temp_df = input_df.copy()
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.notna()] = average_values[temp_df.isna() & forward_df.notna() & backward_df.notna()]
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.isna()] = forward_df[temp_df.isna() & forward_df.notna() & backward_df.isna()]
        temp_df[temp_df.isna() & backward_df.notna() & forward_df.isna()] = backward_df[temp_df.isna() & backward_df.notna() & forward_df.isna()]

        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('DATETIME', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['DATETIME'].dt.dayofweek
            df_grouped['time'] = df_grouped['DATETIME'].dt.time
            # Group by day of the week and time of day, then calculate the mean
            mean_values = df_grouped.groupby(['dayofweek', 'time'])[column].mean().reset_index()
            
            # Function to fill missing values in a column based on datetime index
            def fill_missing_values(index_value, column, mean_values):
                if pd.isnull(temp_df.loc[index_value, column]):
                    # Find the corresponding mean value based on datetime index
                    mean_value = mean_values.loc[(mean_values['dayofweek'] == index_value.dayofweek) & (mean_values['time'] == index_value.time()), column].values[0]
                    return mean_value
                else:
                    return temp_df.loc[index_value, column]
            temp_df[column] = temp_df.apply(lambda row: fill_missing_values(row.name, column, mean_values), axis=1)

        imputed_df = temp_df.reset_index(drop=True)

    elif imputation_method == "Forward":
        imputed_df = input_df.fillna(input_df.shift(-4*24*7))
        temp_df = imputed_df
        
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('DATETIME', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['DATETIME'].dt.dayofweek
            df_grouped['time'] = df_grouped['DATETIME'].dt.time
            # Group by day of the week and time of day, then calculate the mean
            mean_values = df_grouped.groupby(['dayofweek', 'time'])[column].mean().reset_index()
            
            # Function to fill missing values in a column based on datetime index
            def fill_missing_values(index_value, column, mean_values):
                if pd.isnull(temp_df.loc[index_value, column]):
                    # Find the corresponding mean value based on datetime index
                    mean_value = mean_values.loc[(mean_values['dayofweek'] == index_value.dayofweek) & (mean_values['time'] == index_value.time()), column].values[0]
                    return mean_value
                else:
                    return temp_df.loc[index_value, column]
            temp_df[column] = temp_df.apply(lambda row: fill_missing_values(row.name, column, mean_values), axis=1)

        imputed_df = temp_df.reset_index(drop=True)
        
    elif imputation_method == "Backward":
        imputed_df = input_df.fillna(input_df.shift(4*24*7))
        temp_df = imputed_df
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('DATETIME', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['DATETIME'].dt.dayofweek
            df_grouped['time'] = df_grouped['DATETIME'].dt.time
            # Group by day of the week and time of day, then calculate the mean
            mean_values = df_grouped.groupby(['dayofweek', 'time'])[column].mean().reset_index()
            
            # Function to fill missing values in a column based on datetime index
            def fill_missing_values(index_value, column, mean_values):
                if pd.isnull(temp_df.loc[index_value, column]):
                    # Find the corresponding mean value based on datetime index
                    mean_value = mean_values.loc[(mean_values['dayofweek'] == index_value.dayofweek) & (mean_values['time'] == index_value.time()), column].values[0]
                    return mean_value
                else:
                    return temp_df.loc[index_value, column]
            temp_df[column] = temp_df.apply(lambda row: fill_missing_values(row.name, column, mean_values), axis=1)

        imputed_df = temp_df.reset_index(drop=True)
    

    imputed_df = pd.concat([datetime_column, imputed_df], axis=1)
    imputed_df.to_excel(imputation_dir + "/" + "imputed_data_" + imputation_method + ".xlsx", index=False)
    
    return imputed_df

def df_to_gdf(df, lon_name, lat_name):
    """convert dataframe to geodataframe

    Args:
        df (dataframe): input dataframe for conversion
        lon_name (string): column name for longitude
        lat_name (string): column name for latitude

    Returns:
        geodataframe
    """
    geometry = [Point(xy) for xy in zip(df[lon_name], df[lat_name])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs(epsg=4167, inplace=True)

    return gdf

# Visualization functions
def imputation_visualization(raw_data_df, start_time, end_time, method_list, column, output_path):
    """Visualize the imputation methods

    Args:
        raw_data_df (dataframe): dataframe containing raw flow data
        start_time (datetime): start time for plot
        end_time (datetime): end time for plot
        method_list (list): methods to be plotted
        column (string): station for visualization
        output_path (string): path to save the output figure

    Returns:
        None
    """
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    sns.set_theme(style="white")
    mpl.rcParams['font.family'] = 'Times New Roman'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    raw_data_df = raw_data_df[["DATETIME", column]]
    raw_data_df = raw_data_df.loc[(raw_data_df['DATETIME'] >= start_time) & (raw_data_df['DATETIME'] <= end_time)]
    raw_data_df = raw_data_df.rename(columns={column:"Raw"})
    raw_data_df = raw_data_df.fillna(value=np.nan)
    raw_data_df.replace(to_replace=[None], value=np.nan, inplace=True)
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'DATETIME': time_index})
    for method in method_list:
        temp_df = pd.read_excel("./result/flow/imputation/imputed_data_" + method +".xlsx")
        temp_df = temp_df[["DATETIME", column]]
        temp_df = temp_df.loc[(temp_df['DATETIME'] >= start_time) & (temp_df['DATETIME'] <= end_time)]
        temp_df = temp_df.rename(columns={column:method})
        time_series_df = pd.merge(time_series_df, temp_df, on='DATETIME', how="left")
        
    time_series_df = pd.merge(time_series_df, raw_data_df, on='DATETIME', how="left")
    time_series_df = time_series_df.set_index("DATETIME")
    
    plt.figure(figsize=(8, 5))
    ax = lineplot_breaknans(data=time_series_df, y="Raw", markers=True, linewidth=1.5, break_at_nan=True)
    
    columns_to_plot = [col for col in time_series_df.columns if col != "Raw" and col != "Forward-Backward"]
    temp_time_series_df = time_series_df[columns_to_plot]
    sns.lineplot(data=temp_time_series_df, ax=ax, markers=True, linewidth=1.5)
    
    missing_mask = time_series_df['Raw'].isna().values.astype(int)
    ax.set_xlim(time_series_df.index[0], time_series_df.index[-1])
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  missing_mask[np.newaxis], cmap='Blues', alpha=0.2)
    
    if "Forward-Backward" in time_series_df.columns:
        sns.lineplot(data=time_series_df["Forward-Backward"], ax=ax, color='#000000', linewidth=1.5, label="Forward-Backward")
    
    plt.rc('legend', fontsize=10.5)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    
    # Set the date format on the x-axis to show minutes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    # Put a legend below current axis
    ax.legend(loc='lower left', mode="expand", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=3, frameon=False)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.tick_params(labelsize=10.5)
    plt.xlabel("Time (hour)", fontsize=10.5)
    plt.ylabel("Traffic flow", fontsize=10.5)
        
    #plt.tight_layout()
    plt.savefig(output_path + "imputation_methods.png", dpi=600)
    plt.close()
    
    return None

def direction_visualization(place_list, flow_meta_list, flow_df, output_path):
    # deprecated

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    hue_order = ['Light', 'Heavy']
    palette = {'Light': '#0466c8', 'Heavy': '#d90429'}
    for iter in range(len(place_list)):
        place = place_list[iter]
        temp_flow_meta_gdf = flow_meta_list[iter]
        temp_siteRef_list = temp_flow_meta_gdf["SITEREF"].to_list()
        temp_flow_df = flow_df[flow_df['SITEREF'] == temp_siteRef_list[0]]
        temp_flow_df = temp_flow_df.sort_values(by=['WEIGHT'], ascending=False)
        temp_flow_df = temp_flow_df.fillna(value=np.nan)
        
        if iter == 0:
            scatter = sns.scatterplot(ax=axes[iter], data=temp_flow_df, x='PROPORTION', y=temp_flow_df['DATETIME'].dt.hour,
                hue='WEIGHT', size='TOTAL_FLOW', sizes=(10, 500), hue_order=hue_order, palette=palette, linewidth=0)
            handles, labels = scatter.get_legend_handles_labels()
            handles.pop()
            labels.pop()
            handles.pop()
            labels.pop()
            # Create a single legend for the entire figure
            fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1), frameon=False, fontsize=16)
            scatter.legend_.remove()
        else:
            scatter = sns.scatterplot(ax=axes[iter], data=temp_flow_df, x='PROPORTION', y=temp_flow_df['DATETIME'].dt.hour,
                hue='WEIGHT', size='TOTAL_FLOW', sizes=(10, 500), hue_order=hue_order, palette=palette, linewidth=0, legend=False)
        
        axes[iter].tick_params(axis='both', which='major', labelsize=16)
        axes[iter].set_xlabel(place, fontsize=16)
        axes[iter].set_yticks(np.arange(0, 24, 2))
    
    # Set shared y-axis label
    axes[0].set_ylabel('Time', fontsize=16)
    fig.text(0.5, 0.04, 'Proportion (0.5 to 1)', ha='center', va='center', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.85])

    plt.savefig(output_path + "proportion.png", dpi=600)
    plt.close()
    
    return

def direction_cat_visualization(place_list, flow_meta_list, flow_df, output_path):
    
    sns.set_theme(style="white")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    sns.set_context("notebook", rc={"axes.titlesize":10.5, "axes.labelsize":10.5, "xtick.labelsize":10.5, "ytick.labelsize":10.5})
    mpl.rcParams['font.family'] = 'Times New Roman'
    region_palette = {"Wellington":'#fca311', "Christchurch":'#0466c8', "Auckland":'#c1121f'}
    
    flow_df = flow_df.groupby(['SITEREF', 'DATETIME', 'WEIGHT'])['FLOW'].sum().reset_index()
    
    date_range1 = pd.date_range(start="2019-01-01 00:00:00", end="2019-12-31 23:45:00", freq="15min")
    weight1 = ['Light'] * len(date_range1)

    # Create the second date range with 'Heavy' weight
    date_range2 = pd.date_range(start="2019-01-01 00:00:00", end="2019-12-31 23:45:00", freq="15min")
    weight2 = ['Heavy'] * len(date_range2)

    # Create DataFrames for each range
    df1 = pd.DataFrame({'DATETIME': date_range1, 'WEIGHT': weight1})
    df2 = pd.DataFrame({'DATETIME': date_range2, 'WEIGHT': weight2})

    # Concatenate the two DataFrames
    cat_df = pd.concat([df1, df2]).reset_index(drop=True)

    for iter in range(len(place_list)):
        place = place_list[iter]
        temp_flow_meta_gdf = flow_meta_list[iter]
        temp_siteRef_list = temp_flow_meta_gdf["SITEREF"].to_list()
        temp_flow_df = flow_df[flow_df['SITEREF'] == temp_siteRef_list[0]]
        temp_flow_meta_gdf = temp_flow_meta_gdf[temp_flow_meta_gdf['SITEREF'] == temp_siteRef_list[0]]

        result_dfs = []
        for direction, group_df in temp_flow_df.groupby('WEIGHT'):
            # Drop the 'DIRECTION' column
            sub_df = group_df.drop(columns=['WEIGHT'])
            # Apply the imputation function
            imputed_df = flow_data_imputation(temp_flow_meta_gdf, sub_df, process_engine, False)
            # Add the 'DIRECTION' column back
            imputed_df['WEIGHT'] = direction
            # Store the result
            result_dfs.append(imputed_df)
        # Concatenate all the results
        temp_flow_df = pd.concat(result_dfs).reset_index(drop=True)
        
        temp_flow_df = temp_flow_df.rename(columns={'FLOW':place})
        temp_flow_df = temp_flow_df.groupby(['DATETIME', 'WEIGHT'])[place].mean()
        print(temp_flow_df)
        cat_df = pd.merge(cat_df, temp_flow_df, on=['DATETIME', "WEIGHT"], how="left")
    
    cat_df['Hour'] = cat_df['DATETIME'].dt.strftime('%H:%M')

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    
    print(cat_df)
    for col_name in place_list:
        sns.lineplot(x='Hour', y=col_name, errorbar=("ci", 95), data=cat_df[cat_df['WEIGHT'] == 'Light'], ax=axes[0], linewidth=1.5, color=region_palette.get(col_name))
        sns.lineplot(x='Hour', y=col_name, errorbar=("ci", 95), data=cat_df[cat_df['WEIGHT'] == 'Heavy'], ax=axes[1], linewidth=1.5, color=region_palette.get(col_name))

    for i in range(2):
        axes[i].tick_params(axis='both', which='major', labelsize=10.5)
        axes[i].set_xlim(cat_df.Hour.min(), cat_df.Hour.max())
        axes[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
        if i == 0:
            axes[i].set_xlabel("Time (hour)" + "\n" + "(a) Traffic flow of light duty vehicles", fontsize=10.5)
        else:
            axes[i].set_xlabel("Time (hour)" + "\n" + "(b) Traffic flow of heavy duty vehicles", fontsize=10.5)
    
    axes[0].set_ylabel('Traffic flow', fontsize=10.5)
    axes[1].set_ylabel('', fontsize=10.5)
    #fig.text(0.5, 0.04, 'Time', ha='center', va='center', fontsize=10.5)
    
    handles = [plt.Line2D([0], [0], color=color, lw=1.5) for color in region_palette.values()]
    labels = list(region_palette.keys())
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=10.5)
    plt.savefig(output_path + "flow_cat.png", dpi=600)
    plt.close()
    
    return

def city_traffic_visualization(place_list, city_week_traffic_df, output_path):
    
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    sns.set_theme(style="white")
    mpl.rcParams['font.family'] = 'Times New Roman'

    city_week_traffic_df['Hour'] = city_week_traffic_df['DATETIME'].dt.strftime('%H:%M')
    city_week_traffic_df['DayOfWeek'] = city_week_traffic_df['DATETIME'].dt.dayofweek
    city_week_traffic_df['DayType'] = city_week_traffic_df['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    city_week_traffic_df = city_week_traffic_df[["Wellington", "Christchurch", "Auckland", "DayType", "Hour"]]
    #week_df = city_week_traffic_df.groupby(['DayType', 'Hour']).mean().reset_index()
    week_df = city_week_traffic_df
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    region_palette = {"Wellington":'#fca311', "Christchurch":'#0466c8', "Auckland":'#c1121f'}
    
    for col_name in place_list:
        sns.lineplot(x='Hour', y=col_name, errorbar=("pi", 50), data=week_df[week_df['DayType'] == 'Weekday'], ax=axes[0], linewidth=1.5, color=region_palette.get(col_name))
        sns.lineplot(x='Hour', y=col_name, errorbar=("pi", 50), data=week_df[week_df['DayType'] == 'Weekend'], ax=axes[1], linewidth=1.5, color=region_palette.get(col_name))

    for i in range(2):
        axes[i].tick_params(axis='both', which='major', labelsize=10.5)
        axes[i].set_xlim(week_df.Hour.min(), week_df.Hour.max())
        axes[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
        if i == 0:
            axes[i].set_xlabel("Time (hour)" + "\n" + "(a) Weekday", fontsize=10.5)
        else:
            axes[i].set_xlabel("Time (hour)" + "\n" + "(b) Weekend", fontsize=10.5)
    
    axes[0].set_ylabel('Traffic flow', fontsize=10.5)
    #fig.text(0.5, 0.04, 'Time', ha='center', va='center', fontsize=10.5)
    
    handles = [plt.Line2D([0], [0], color=color, lw=1.5) for color in region_palette.values()]
    labels = list(region_palette.keys())
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=10.5)
    plt.savefig(output_path + "weekday_weekend.png", dpi=600)
    plt.close()
    
    return
    
def morning_afternoon_peak_visualization(city_traffic_df, output_path):
    
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    sns.set_theme(style="white")
    mpl.rcParams['font.family'] = 'Times New Roman'
    
    auckland_df = city_traffic_df[["DATETIME", "Auckland"]]
    auckland_df['DATETIME'] = pd.to_datetime(auckland_df['DATETIME'])
    auckland_df['Month'] = auckland_df['DATETIME'].dt.month
    
    morning_df = auckland_df[(auckland_df['DATETIME'].dt.hour >= 8) & (auckland_df['DATETIME'].dt.hour < 9)]
    afternoon_df = auckland_df[(auckland_df['DATETIME'].dt.hour >= 17) & (auckland_df['DATETIME'].dt.hour < 18)]
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for i in range(2):
        axes[i].tick_params(axis='both', which='major', labelsize=10.5)
        #axes[i].set_xticks(np.arange(1, 13, 2))
    
    sns.boxplot(x='Month', y='Auckland', data=morning_df, ax=axes[0], palette="Spectral", whis=(0, 100))
    sns.boxplot(x='Month', y='Auckland', data=afternoon_df, ax=axes[1], palette="mako", whis=(0, 100))
    sns.stripplot(x='Month', y='Auckland', data=morning_df, ax=axes[0], size=2, color="#495057")
    sns.stripplot(x='Month', y='Auckland', data=afternoon_df, ax=axes[1], size=2, color="#495057")
    
    axes[0].set_xlabel('(a) Morning peak', fontsize=10.5)
    axes[1].set_xlabel('(b) Afternoon peak', fontsize=10.5)
    axes[0].set_ylabel('Traffic flow', fontsize=10.5)
    fig.text(0.5, 0.04, 'Month', ha='center', va='center', fontsize=10.5)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path +"peak_time.png", dpi=600)
    plt.close()

    return

def weather_correlation_visualization(daily_flow_df, weather_id_list, weather_df, auckland_weather_meta_gdf, auckland_flow_meta_gdf, light_df, process_engine, output_path):
    
    for weather_id in weather_id_list:
        auckland_weather_df = weather_df[weather_df["STATION_ID"] == weather_id]
        auckland_weather_df["DATETIME"] = pd.to_datetime(auckland_weather_df["DATETIME"])
        auckland_weather_df = auckland_weather_df[auckland_weather_df["DATETIME"].dt.year == 2019]
        auckland_weather_df = auckland_weather_df[["DATETIME", "TEMP", "RH", ]]
        auckland_weather_df = auckland_weather_df.rename(columns = {"TEMP":"Temperature(C)",
                                                          #"DEWP":"Dew Point(C)", 
                                                          "RH":"Humidity(%)", 
                                                          #"PRCP":"Precipitation(m)"
                                                          })
        auckland_weather_df = auckland_weather_df.groupby('DATETIME').mean().reset_index()
        auckland_weather_df = auckland_weather_df.astype(str)
        
        temp_auckland_weather_meta_gdf = auckland_weather_meta_gdf[auckland_weather_meta_gdf["STATION_ID"] == weather_id]
        temp_siteref = temp_auckland_weather_meta_gdf["NEAREST"].to_list()[0]
        temp_auckland_flow_meta_gdf = auckland_flow_meta_gdf[auckland_flow_meta_gdf["SITEREF"] == temp_siteref]
    
        imputed_light_df = flow_data_imputation(temp_auckland_flow_meta_gdf, light_df, process_engine, False)
        auckland_df = imputed_light_df[imputed_light_df["SITEREF"] == temp_siteref]
        auckland_df = auckland_df[["DATETIME", "TOTAL_FLOW"]]
        auckland_df.set_index('DATETIME', inplace=True)
        #daily_flow_df["MAX_FLOW"] = auckland_df['TOTAL_FLOW'].resample('D').max().to_list()
        daily_flow_df["Mean Traffic Flow"] = auckland_df['TOTAL_FLOW'].resample('D').mean().to_list()
        daily_flow_df = daily_flow_df.astype(str)
        
        correlation_df = pd.merge(auckland_weather_df, daily_flow_df, on='DATETIME', how='left')
        correlation_df = correlation_df.drop(["DATETIME"], axis=1)
        for col in correlation_df.columns:
            correlation_df[col] = correlation_df[col].astype(float)
        
        # Plot the correlation
        matplotlib.rc('xtick', labelsize=10.5)
        matplotlib.rc('ytick', labelsize=10.5)
        mpl.rcParams["axes.labelsize"] = 10.5
        plt.rc('legend', fontsize=10.5)
        
        def corrfunc(x, y, **kwds):
            cmap = kwds['cmap']
            norm = kwds['norm']
            ax = plt.gca()
            ax.tick_params(bottom=False, top=False, left=False, right=False, axis="both", which="major", labelsize=10.5)
            sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
            r, _ = pearsonr(x, y)
            facecolor = cmap(norm(r))
            ax.set_facecolor(facecolor)
            lightness = (max(facecolor[:3]) + min(facecolor[:3]) ) / 2
            # Correlation number on the plot
            ax.annotate(f"{r:.2f}", xy=(.5, .5), xycoords=ax.transAxes,
                    color='white' if lightness < 0.7 else 'black', size=26, ha='center', va='center')
        
        for i in range(2):
            plt.figure(figsize=(8, 8))
            g = sns.PairGrid(correlation_df)
            """
            if i == 0: # With precipitation
                temp_correlation_df = correlation_df[correlation_df["Precipitation(m)"] == 0]
                temp_correlation_df = temp_correlation_df.drop(["Precipitation(m)"], axis=1)
                g = sns.PairGrid(temp_correlation_df)

            else:
                temp_correlation_df = correlation_df[correlation_df["Precipitation(m)"] > 0]
                g = sns.PairGrid(temp_correlation_df)
            """
            
            g.map_lower(plt.scatter, s=22)
            g.map_diag(sns.histplot, kde=False)
            g.map_upper(corrfunc, cmap=plt.get_cmap('crest'), norm=plt.Normalize(vmin=0, vmax=1))
            
            # Adjust label size for all axes
            for ax in g.axes.flatten():
                ax.tick_params(axis='both', which='major', labelsize=10.5)
                ax.get_yaxis().set_label_coords(-0.25, 0.5)
            
            plt.tight_layout(rect=[0.02, 0, 1, 1])
            plt.savefig(output_path + "correlation_" + weather_id + "_" + str(i) + ".png", dpi=600)
            plt.close()

    return

def event_all_visualization(event_colors, event_df, output_path):

    fig, ax = plt.subplots(figsize=(26, 14), layout='constrained')
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.plot(event_df.index, event_df['Auckland'], color='#274c77')

    matplotlib.rc('xtick', labelsize=22)
    matplotlib.rc('ytick', labelsize=22)
    plt.rc('legend', fontsize=22)
    
    for event, color in event_colors.items():
        subset = event_df[event_df["EVENT"] == event]
        print(event)
        
        if not subset.empty:
            dfs = []
            start_idx = subset.index[0]
            end_idx = None
        
            for idx in subset.index[1:]:
                if ((idx - start_idx).seconds // 3600) > 1:
                    # End of current part found
                    dfs.append(subset.loc[start_idx:end_idx])
                    start_idx = idx
                    end_idx = None
                else:
                    # Continuation of current part
                    end_idx = idx

            df_num = 0
            for group_df in dfs:
                if event == "None":
                    ax.axvspan(group_df.index[0], group_df.index[-1], alpha=0, edgecolor='none')
                else:
                    if df_num == 0:
                        ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.5, edgecolor='none', label=str(event))
                        df_num = df_num + 1
                    else:
                        ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.5, edgecolor='none')
                        df_num = df_num + 1
    
    ax.set_xlim(event_df.index.min(), event_df.index.max())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set(xlabel="", ylabel="")
    plt.xlabel("Time", fontsize=22)
    plt.ylabel("Traffic Flow", fontsize=22)
    
    fig.legend(loc='outside center right')        
    
    plt.savefig(output_path + "events.png", dpi=600)
    plt.close()

    return

def event_subplot_visualization(event_colors, event_df, output_path):
    alphabet_list = [chr(chNum) for chNum in list(range(ord('a'),ord('z')+1))]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    # Flatten the axes array for easy iteration
    axs = axs.flatten()
    
    event_num = 0
    for event, color in event_colors.items():
        subset = event_df[event_df["EVENT"] == event]
        print(event)
        
        if event == "None":
            break
        
        if not subset.empty:
            dfs = []
            start_idx = subset.index[0]
            end_idx = None
        
            found = 0
            for idx in subset.index[1:]:
                if found == 1:
                    break
                
                if ((idx - start_idx).seconds // 3600) > 1:
                    # End of current part found
                    start_time = start_idx
                    end_time = start_idx + pd.Timedelta(days=1)
                    
                    dfs.append(subset.loc[start_idx:end_idx])
                    
                    start_idx = idx
                    end_idx = None
                    found = 1
                else:
                    # Continuation of current part
                    end_idx = idx
            if dfs:
                ax = axs[event_num]
                
                extreme_df = event_df.loc[(event_df.index >= start_time) & (event_df.index < end_time)]
                
                extreme_df_before = event_df.loc[(event_df.index >= start_time - pd.Timedelta(days=1)) & 
                (event_df.index <= end_time - pd.Timedelta(days=1))]
                extreme_df_before = extreme_df_before.shift(freq="24H")
                
                extreme_df_week_before = event_df.loc[(event_df.index >= start_time - pd.Timedelta(days=7)) & 
                (event_df.index <= end_time - pd.Timedelta(days=7))]
                extreme_df_week_before = extreme_df_week_before.shift(freq="168H")

                extreme_df_after = event_df.loc[(event_df.index >= start_time + pd.Timedelta(days=1)) & 
                (event_df.index <= end_time + pd.Timedelta(days=1))]
                extreme_df_after = extreme_df_after.shift(freq="-24H")
                
                if event_num == 0:
                    #ax.plot(extreme_df.index, week_df[week_df['DayType'] == 'Weekday']["Auckland"], color='blue', linestyle='--', label='Weekday Average')
                    #ax.plot(extreme_df.index, week_df[week_df['DayType'] == 'Weekend']["Auckland"], color='green', linestyle='--', label='Weekend Average')
                    ax.plot(extreme_df_before.index, extreme_df_before['Auckland'], color='#274c77', label="Previous Day", linewidth=1.5)
                    ax.plot(extreme_df_after.index, extreme_df_after['Auckland'], color='#fca311', label="Next Day", linewidth=1.5)
                    ax.plot(extreme_df_week_before.index, extreme_df_week_before['Auckland'], color='#000000', label="Same Day Last Week", linewidth=1.5)
                    ax.plot(extreme_df.index, extreme_df['Auckland'], color="#ba181b", label="Event", linewidth=1.5)
                else:
                    #ax.plot(extreme_df.index, week_df[week_df['DayType'] == 'Weekday']["Auckland"], color='blue', linestyle='--')
                    #ax.plot(extreme_df.index, week_df[week_df['DayType'] == 'Weekend']["Auckland"], color='green', linestyle='--')
                    ax.plot(extreme_df_before.index, extreme_df_before['Auckland'], color='#274c77', linewidth=1.5)
                    ax.plot(extreme_df_after.index, extreme_df_after['Auckland'], color='#fca311', linewidth=1.5)
                    ax.plot(extreme_df_week_before.index, extreme_df_week_before['Auckland'], color='#000000', linewidth=1.5)
                    ax.plot(extreme_df.index, extreme_df['Auckland'], color="#ba181b", linewidth=1.5)
                
                ax.set_title("(" + alphabet_list[event_num] + ") " + event, fontsize=10.5)
                ax.set_xlim(extreme_df.index.min(), extreme_df.index.max())
                ax.tick_params(axis='both', which='major', labelsize=10.5)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45) 
                ax.set(xlabel="", ylabel="")
        
                event_num = event_num + 1
    
    # Hide the empty subplots
    for ax in axs[6:]:
        ax.axis('off')

    # Adjust layout
    plt.tight_layout(rect=[0.02, 0.05, 1, 1])
    # Add ylabel to the entire figure
    fig.text(0.004, 0.5, 'Traffic flow', va='center', rotation='vertical', fontsize=10.5)
    
    fig.legend(loc='lower center', ncol=4, frameon=False, fontsize=10.5)
    # Show the plot
    plt.savefig(output_path + "event_sub_all.png", dpi=600)
    plt.close()

def weight_proportion_visualization(place_list, flow_meta_list, flow_df, output_path):

    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    sns.set_theme(style="white")
    mpl.rcParams['font.family'] = 'Times New Roman'

    plot_df_list = []
    for iter in range(len(place_list)):
        temp_flow_meta_gdf = flow_meta_list[iter]
        temp_siteRef_list = temp_flow_meta_gdf["SITEREF"].to_list()
        temp_flow_df = flow_df[flow_df['SITEREF'] == temp_siteRef_list[0]]
        temp_flow_meta_gdf = temp_flow_meta_gdf[temp_flow_meta_gdf['SITEREF'] == temp_siteRef_list[0]]

        result_dfs = []
        for weight, group_df in temp_flow_df.groupby('WEIGHT'):
            # Drop the 'WEIGHT' column
            sub_df = group_df.drop(columns=['WEIGHT'])
            # Apply the imputation function
            imputed_df = flow_data_imputation(temp_flow_meta_gdf, sub_df, process_engine, False)
            # Add the 'WEIGHT' column back
            imputed_df['WEIGHT'] = weight
            # Store the result
            result_dfs.append(imputed_df)
        # Concatenate all the results
        temp_flow_df = pd.concat(result_dfs).reset_index(drop=True)
        
        
        temp_flow_df["HOUR"] = temp_flow_df['DATETIME'].dt.strftime('%H:%M')
        temp_flow_df = temp_flow_df[["HOUR", "WEIGHT", "FLOW"]]
        temp_flow_df = temp_flow_df.groupby(['HOUR', 'WEIGHT']).mean().reset_index()
        plot_df_list.append(temp_flow_df)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    region_palette = {"Wellington":['#fca311', "#ee6c4d"], "Christchurch":['#03045e', "#99d98c"], "Auckland":["#370617", '#dc2f02']}
    
    for i in range(len(place_list)):
        palette = region_palette.get(place_list[i])
        plot_df = plot_df_list[i]
        sns.histplot(data=plot_df, x="HOUR", weights="FLOW", hue="WEIGHT", multiple="stack", ax=axes[i], palette=palette, hue_order=["Light", "Heavy"])
        axes[i].tick_params(axis='both', which='major', labelsize=10.5)
        axes[i].set_xlim(plot_df.HOUR.min(), plot_df.HOUR.max())
        axes[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
        if i == 1:
            axes[i].set_xlabel("(b) " + place_list[i] + "\n" + "Time (hour)", fontsize=10.5)
        elif i == 0:
            axes[i].set_xlabel("(a) " + place_list[i], fontsize=10.5)
        else:
            axes[i].set_xlabel("(c) " + place_list[i], fontsize=10.5)
        
        legend = axes[i].get_legend()
        legend.set_title(None)
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(10.5)  # Set legend text fontsize
    
    axes[0].set_ylabel('Traffic flow', fontsize=10.5)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Show the plot
    plt.savefig(output_path + "light_heavy_proportion.png", dpi=600)
    plt.close()
    
    return

def direction_line_visualization(place_list, flow_meta_list, flow_df, output_path):
    
    flow_df = flow_df.groupby(['SITEREF', 'DATETIME', 'DIRECTION'])['FLOW'].sum().reset_index()
    
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    sns.set_theme(style="white")
    mpl.rcParams['font.family'] = 'Times New Roman'
    
    plot_df_list = []
    for iter in range(len(place_list)):
        temp_flow_meta_gdf = flow_meta_list[iter]
        temp_siteRef_list = temp_flow_meta_gdf["SITEREF"].to_list()
        
        # Imputation
        temp_flow_df = flow_df[flow_df['SITEREF'] == temp_siteRef_list[0]]
        temp_flow_meta_gdf = temp_flow_meta_gdf[temp_flow_meta_gdf['SITEREF'] == temp_siteRef_list[0]]

        result_dfs = []
        for direction, group_df in temp_flow_df.groupby('DIRECTION'):
            # Drop the 'DIRECTION' column
            sub_df = group_df.drop(columns=['DIRECTION'])
            # Apply the imputation function
            imputed_df = flow_data_imputation(temp_flow_meta_gdf, sub_df, process_engine, False)
            # Add the 'DIRECTION' column back
            imputed_df['DIRECTION'] = direction
            # Store the result
            result_dfs.append(imputed_df)
        # Concatenate all the results
        temp_flow_df = pd.concat(result_dfs).reset_index(drop=True)
        
        temp_flow_df["HOUR"] = temp_flow_df['DATETIME'].dt.strftime('%H:%M')
        temp_flow_df = temp_flow_df[["HOUR", "FLOW", "DIRECTION"]]
        temp_flow_df = temp_flow_df.groupby(['HOUR', 'DIRECTION']).mean().reset_index()
        plot_df_list.append(temp_flow_df)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    region_palette = {"Wellington":['#fca311', "#ee6c4d"], "Christchurch":['#03045e', "#99d98c"], "Auckland":["#370617", '#dc2f02']}
    
    for i in range(len(place_list)):
        palette = region_palette.get(place_list[i])
        plot_df = plot_df_list[i]
        sns.histplot(data=plot_df, x="HOUR", weights="FLOW", hue="DIRECTION", multiple="stack", ax=axes[i], palette=palette)
        axes[i].tick_params(axis='both', which='major', labelsize=10.5)
        axes[i].set_xlim(plot_df.HOUR.min(), plot_df.HOUR.max())
        axes[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
        if i == 1:
            axes[i].set_xlabel("(b) " + place_list[i] + "\n" + "Time (hour)", fontsize=10.5)
        elif i == 0:
            axes[i].set_xlabel("(a) " + place_list[i], fontsize=10.5)
        else:
            axes[i].set_xlabel("(c) " + place_list[i], fontsize=10.5)
            
        legend = axes[i].get_legend()
        legend.set_title(None)
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(10.5)  # Set legend text fontsize
    
    axes[0].set_ylabel('Traffic flow', fontsize=10.5)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Show the plot
    plt.savefig(output_path + "direction_stack.png", dpi=600)
    plt.close()
    
    return

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

    # If RAM is huge, select from the established database
    #flow_query = "SELECT * FROM flow WHERE strftime('%Y', DATETIME) IN ('2019')"
    #flow_df = pd.read_sql(flow_query, engine)
    #flow_df = flow_df.astype({"DATETIME":"datetime64[ns]"})

    # If RAM is limited, create a new database with only desired time range
    siteRef_list = flow_meta_df["SITEREF"].to_list()

    flow_df = traffic_flow_import_19("./data/traffic/flow_data_13_20/", siteRef_list, process_engine, False)
    flow_df = flow_df.astype({"DATETIME":"datetime64[ms]"})
    ####################################################################################################
    # Process the data to obtain proportion instead of analyzing the direction

    total_flow = flow_df.groupby(['SITEREF', 'DATETIME', 'WEIGHT'])['FLOW'].sum().reset_index()
    total_flow = total_flow.rename(columns={'FLOW': 'TOTAL_FLOW'})
    
    proportion_flow_df = pd.merge(flow_df, total_flow, on=['SITEREF', 'DATETIME', 'WEIGHT'])
    proportion_flow_df['PROPORTION'] = proportion_flow_df['FLOW'] / proportion_flow_df['TOTAL_FLOW']
    
    # Filter out the rows with proportion higher than 50%, row PROPORTION is the imbalance
    proportion_flow_df = proportion_flow_df[proportion_flow_df['PROPORTION'] > 0.5]
    proportion_flow_df = proportion_flow_df.drop(columns=['DIRECTION'])
    
    # Select by weight
    light_df = proportion_flow_df[proportion_flow_df['WEIGHT'] == "Light"]
    #heavy_df = flow_df[flow_df['WEIGHT'] == "Heavy"]

    # Analyze the light vehicles
    temp_light_flow_df = light_df[["DATETIME", "SITEREF", "FLOW"]]

    # Select light vehicles for analysis
    light_pivot_df = temp_light_flow_df.pivot(index='DATETIME', columns='SITEREF', values='FLOW')
    light_merged_df = datetime_df.merge(light_pivot_df, on='DATETIME', how='left')
    
    # Visualize the missing data
    flow_missing_data_visualization(light_merged_df, "./result/flow/missing")
    print("Done")
    time.sleep(100000)
    """
    # Filter based on missing value percentage
    Session = sessionmaker(bind=process_engine)
    session = Session()
    """
    #filtered_meta_df = flow_missing_filter(flow_meta_df, light_merged_df, 30, process_engine)
    #filtered_meta_query = 'SELECT * FROM filtered_flow_meta'
    #filtered_meta_df = pd.read_sql(filtered_meta_query, process_engine)
    #imputed_light_df = flow_data_imputation(filtered_meta_df, light_df, process_engine, False)
    
    #session.commit()
    #session.close()
    ####################################################################################################
    # Imputation method visualization, take one station for visualization
    """
    station_df = light_merged_df[["DATETIME", "00200091"]]
    station_df.to_excel("./result/flow/imputation/raw.xlsx", index=False)
    
    imputation_methods = ["Linear", "Forward", "Backward", "Forward-Backward"]
    for method in imputation_methods:
        imputed_df = imputation(station_df, save_path="./result/flow/imputation", imputation_method=method)
        imputed_df = pd.read_excel("./result/flow/imputation/imputed_data_" + method + ".xlsx")
    """
    """
    print("Imputation Visualization")
    station_df = pd.read_excel("./result/flow/imputation/raw.xlsx")
    imputation_visualization(station_df, '2019-12-10 00:00:00', '2019-12-13 00:00:00', 
                                        ["Forward", "Backward", "Forward-Backward"],
                                        "00200091",
                                        "./result/flow/imputation/")
    """
    ####################################################################################################
    # Weight-percentage analysis
    # 047-Wellington 060-Christchurch 076-Auckland
    boundary_shp = gpd.read_file("./data/boundary/city_districts/city_districts.shp")
    Wellington_shp = boundary_shp[boundary_shp["TA2023_V1_"] == "047"]
    Christchurch_shp = boundary_shp[boundary_shp["TA2023_V1_"] == "060"]
    Auckland_shp = boundary_shp[boundary_shp["TA2023_V1_"] == "076"]
    
    filtered_meta_query = 'SELECT * FROM filtered_flow_meta'
    filtered_meta_df = pd.read_sql(filtered_meta_query, process_engine)
    flow_meta_gdf = df_to_gdf(filtered_meta_df, "LON", "LAT")
    place_list = ["Wellington", "Christchurch", "Auckland"]
    shp_list = [Wellington_shp, Christchurch_shp, Auckland_shp]
    flow_meta_list = [flow_meta_gdf[flow_meta_gdf.geometry.within(shp.unary_union)] for shp in shp_list]
    
    # Direction visualization distribution, deprecated
    #direction_visualization(place_list, flow_meta_list, flow_df, output_path="./result/flow/")
    
    # Direction visualization cat plot
    direction_cat_visualization(place_list, flow_meta_list, flow_df, output_path="./result/flow/")
    
    # Weight proportion visualization
    print("Weight proportion stack")
    weight_proportion_visualization(place_list, flow_meta_list, flow_df, output_path="./result/flow/")
    
    # Direction visualization lineplot
    print("Direction lineplot")
    direction_line_visualization(place_list, flow_meta_list, flow_df, output_path="./result/flow/")
    ####################################################################################################
    # Weekday and weekend
    """
    light_df = imputed_light_df[["DATETIME", "SITEREF", "TOTAL_FLOW"]]
    city_traffic_df = pd.DataFrame()
    city_traffic_df["DATETIME"] = time_index
    for iter in range(len(place_list)):
        place = place_list[iter]
        temp_flow_meta_gdf = flow_meta_list[iter]
        
        temp_siteRef_list = temp_flow_meta_gdf["SITEREF"].to_list()
        temp_light_df = light_df[light_df['SITEREF'].isin(temp_siteRef_list)]
        temp_light_df = temp_light_df.rename(columns={'TOTAL_FLOW':place})
        average_total_flow = temp_light_df.groupby('DATETIME')[place].mean()
        city_traffic_df = pd.merge(city_traffic_df, average_total_flow, on='DATETIME', how="left")
    
    city_traffic_df.to_excel("./result/flow/city_mean.xlsx", index=False)
    """
    print("Weekdays and weekends")
    city_week_traffic_df = pd.read_excel("./result/flow/city_mean.xlsx")
    #city_traffic_visualization(place_list, city_week_traffic_df, "./result/flow/")
    ####################################################################################################
    # Morning peak and afternoon peak of Auckland
    print("Morning afternoon peak")
    city_week_traffic_df = pd.read_excel("./result/flow/city_mean.xlsx")
    #morning_afternoon_peak_visualization(city_week_traffic_df, "./result/flow/")
    ####################################################################################################
    # Weather correlation
    print("Weather correlation")
    # Auckland
    # Weather data preparation
    weather_process_db_address = 'sqlite:///./data/NZDB_weather_process.db'
    weather_process_engine = create_engine(weather_process_db_address)
    weather_df_query = 'SELECT * FROM filtered_weather'
    weather_df = pd.read_sql(weather_df_query, weather_process_engine)
    weather_meta_query = 'SELECT * FROM filtered_weather_meta'
    weather_meta_df = pd.read_sql(weather_meta_query, weather_process_engine)
    weather_meta_gdf = df_to_gdf(weather_meta_df, "LON", "LAT")
    auckland_weather_meta_gdf = weather_meta_gdf[weather_meta_gdf.geometry.within(Auckland_shp.unary_union)]
    
    # Flow data preparation
    # Resample data to daily frequency and calculate max and mean
    daily_flow_df = pd.DataFrame()
    daily_flow_df["DATETIME"] = pd.date_range(start="2019-01-01 00:00:00", end="2019-12-31 00:00:00", freq="D")
    auckland_flow_meta_gdf = flow_meta_gdf[flow_meta_gdf.geometry.within(Auckland_shp.unary_union)]
    
    # Calculate the nearest flow station to weather station
    weather_coords = np.array(list(auckland_weather_meta_gdf.geometry.apply(lambda geom: (geom.x, geom.y))))
    flow_coords = np.array(list(auckland_flow_meta_gdf.geometry.apply(lambda geom: (geom.x, geom.y))))
    # Create a cKDTree for fast spatial queries
    tree = cKDTree(flow_coords)
    distances, indices = tree.query(weather_coords, k=1)
    nearest_siteref = auckland_flow_meta_gdf.iloc[indices]['SITEREF'].values
    auckland_weather_meta_gdf['NEAREST'] = nearest_siteref

    light_df = light_df[["DATETIME", "SITEREF", "TOTAL_FLOW"]]
    weather_id_list = auckland_weather_meta_gdf["STATION_ID"].to_list()
    weather_correlation_visualization(daily_flow_df, weather_id_list, weather_df, auckland_weather_meta_gdf, 
                                      auckland_flow_meta_gdf, light_df, process_engine, "./result/weather/")

    ####################################################################################################
    # Extreme weather
    print("Events")
    event_df = pd.DataFrame()
    event_df["DATETIME"] = time_index

    extreme_weather_query = 'SELECT * FROM extreme_weather'
    extreme_weather_df = pd.read_sql(extreme_weather_query, engine)
    extreme_weather_df['START_DATE'] = pd.to_datetime(extreme_weather_df['START_DATE'])
    extreme_weather_df = extreme_weather_df[extreme_weather_df["START_DATE"].dt.year == 2019]
    extreme_weather_df = extreme_weather_df[extreme_weather_df["REGION"].isin(["New Zealand"])]
    extreme_weather_df = extreme_weather_df[["START_DATE", "IDENTIFIER"]]
    extreme_weather_df = extreme_weather_df.rename(columns={"IDENTIFIER":"EVENT"})
    
    # Extend the date
    extended_weather_df = pd.DataFrame({'DATETIME': time_index})
    extended_weather_df['START_DATE'] = extended_weather_df['DATETIME'].dt.date
    extended_weather_df['START_DATE'] = extended_weather_df['START_DATE'].astype(str)
    extreme_weather_df['START_DATE'] = extreme_weather_df['START_DATE'].astype(str)
    extreme_weather_df = pd.merge(extended_weather_df, extreme_weather_df, on='START_DATE', how='left')
    extreme_weather_df = extreme_weather_df.drop(columns='START_DATE')
    
    # Merge
    event_df = pd.merge(event_df, extreme_weather_df, on='DATETIME', how="left")

    # Holiday
    holiday_query = 'SELECT * FROM holiday'
    holiday_df = pd.read_sql(holiday_query, engine)
    holiday_df['START_DATE'] = pd.to_datetime(holiday_df['START_DATE'])
    holiday_df = holiday_df[holiday_df["START_DATE"].dt.year == 2019]
    holiday_df = holiday_df[holiday_df["REGION"].isin(["all", "Auckland"])]
    holiday_df = holiday_df[["START_DATE", "HOLIDAY"]]
    holiday_df = holiday_df.rename(columns={"HOLIDAY":"EVENT"})
    
    # Extend the date
    extended_holiday_df = pd.DataFrame({'DATETIME': time_index})
    extended_holiday_df['START_DATE'] = extended_holiday_df['DATETIME'].dt.date
    extended_holiday_df['START_DATE'] = extended_holiday_df['START_DATE'].astype(str)
    holiday_df['START_DATE'] = holiday_df['START_DATE'].astype(str)
    holiday_df = pd.merge(extended_holiday_df, holiday_df, on='START_DATE', how='left')
    holiday_df = holiday_df.drop(columns='START_DATE')
    
    event_df = pd.merge(event_df, holiday_df, on=['DATETIME'], how="left")
    event_df['EVENT'] = event_df['EVENT_x'].combine_first(event_df['EVENT_y'])
    event_df = event_df.drop(columns=['EVENT_x', 'EVENT_y'])
    
    auckland_df = pd.read_excel("./result/flow/city_mean.xlsx")
    auckland_df = auckland_df[["DATETIME", "Auckland"]]
    auckland_df['DATETIME'] = pd.to_datetime(auckland_df['DATETIME'])
    event_df = pd.merge(event_df, auckland_df, on='DATETIME', how='left')
    
    event_df['Hour'] = event_df['DATETIME'].dt.strftime('%H:%M')
    event_df['DayOfWeek'] = event_df['DATETIME'].dt.dayofweek
    event_df['DayType'] = event_df['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    week_df = event_df[["DayType", "Hour", "Auckland"]]
    week_df = week_df.groupby(['DayType', 'Hour']).mean().reset_index()

    event_df = event_df.fillna("None")
    event_df = event_df.set_index("DATETIME")
    
    event_df['EVENT'] = event_df['EVENT'].replace('August_2019_New_Zealand_Storm', 'Winter Storm')
    event_df['EVENT'] = event_df['EVENT'].replace('December_2019_New_Zealand_Storm', 'Summer Storm')
    
    # Plot, one annual
    event_colors = {
                    #"New Year's Day":                   '#5a189a', 
                    "Day after New Year's Day":         '#7b2cbf', 
                    'Regional anniversary':             '#00a6fb',
                    #'Waitangi Day':                     '#003049',
                    'Good Friday':                      '#e09f3e',
                    #'Easter Monday':                    '#31572c',
                    #'ANZAC Day':                        '#3c6e71',
                    #"Queen's Birthday":                 '#ff7d00',
                    'Winter Storm':                     '#bd1f36',
                    #'Labour Day':                       '#b08968',
                    #'Summer Storm':                     '#85182a',
                    "Christmas Day":                    '#240046',
                    "Boxing Day":                       '#3c096c',
                    'None':                             "#FFFFFF"}
    
    #event_all_visualization(event_colors, event_df, "./result/event/")

    # Subplot
    #event_subplot_visualization(event_colors, event_df, "./result/event/")
