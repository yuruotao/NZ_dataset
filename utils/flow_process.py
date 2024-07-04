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

    Returns:
        None
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
    '''sns.lineplot by default doesn't break the line at nans or infs, 
        which can lead to misleading plots.
    See https://github.com/mwaskom/seaborn/issues/1552 
        and https://stackoverflow.com/questions/52098537/avoid-plotting-missing-values-on-a-line-plot
    
    This function rectifies this, and allows the user to specify 
        if it should break the line at nans, infs, or at both (default).
    
    Note: using this function means you can't use the `units` argument of sns.lineplot.'''
    
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
    chunks = [temp_flow_df.iloc[:, i:i+20] for i in range(0, len(temp_flow_df.columns), 20)]
    
    # Missing data visualization
    index = 0
    for chunk in chunks:
        print(index)
        # Matrix plot
        ax = msno.matrix(chunk, fontsize=20, figsize=(20, 16), label_rotation=45, freq="1M")
        plt.xlabel("Flow Count Sites", fontsize=20)
        plt.ylabel("Sample Points", fontsize=20)
        plt.savefig(missing_matrix_path + 'matrix_' + str(index) + '.png', dpi=600)
        plt.close()
        
        # Bar plot
        ax = msno.bar(chunk, fontsize=20, figsize=(20, 16), label_rotation=45)
        plt.xlabel("Flow Count Sites", fontsize=20)
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


    siteref_list = filtered_meta_df["SITEREF"].to_list()
    
    for siteref in siteref_list:
        print(siteref)
        temp_df = merged_df[merged_df["SITEREF"] == siteref]
        datetime_column = temp_df["DATETIME"]
        temp_df = temp_df.drop(columns=["DATETIME", "SITEREF", "WEIGHT", "FLOW"])
        basic_statistics_df = basic_statistics(temp_df)
        basic_statistics_df["INDEX"] = str(siteref)

        forward_df = temp_df.shift(-24*4)
        backward_df = temp_df.shift(24*4)
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

        temp_df.to_sql('filtered_flow', con=engine, if_exists='append', index=False)
        basic_statistics_df.to_sql('basic_statistics_flow', con=engine, if_exists='append', index=False)

    return None

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
        forward_df = input_df.shift(-4*24)
        backward_df = input_df.shift(4*24)
        
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
        imputed_df = input_df.fillna(input_df.shift(-4*24))
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
        imputed_df = input_df.fillna(input_df.shift(4*24))
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

def imputation_visualization(raw_data_df, start_time, end_time, method_list, column, output_path):

    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
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
    
    plt.figure(figsize=(20, 12))
    ax = lineplot_breaknans(data=time_series_df, y="Raw", markers=True, linewidth=3, break_at_nan=True)
    temp_time_series_df = time_series_df.drop(["Raw"], axis=1, inplace=False)
    sns.lineplot(data=temp_time_series_df, ax=ax, markers=True, linewidth=3)
    missing_mask = time_series_df['Raw'].isna().values.astype(int)
    ax.set_xlim(time_series_df.index[0], time_series_df.index[-1])
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  missing_mask[np.newaxis], cmap='Blues', alpha=0.2)
    
    plt.rc('legend', fontsize=22)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='lower left', mode="expand", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=5)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.tick_params(labelsize=22)
    plt.xlabel("Time", fontsize=22)
    plt.ylabel("Traffic Flow", fontsize=22)
        
    #plt.tight_layout()
    plt.savefig(output_path + "imputation_methods.png", dpi=600)
    plt.close()
    
    return None


if __name__ == "__main__":
    process_db_address = 'sqlite:///./data/NZDB_flow_process.db'
    process_engine = create_engine(process_db_address)
    Base.metadata.create_all(process_engine)
    
    db_address = 'sqlite:///./data/NZDB/NZDB.db'
    engine = create_engine(db_address)
    
    flow_meta_query = 'SELECT * FROM flow_meta'
    flow_meta_df = pd.read_sql(flow_meta_query, engine)

    # If RAM is huge, select from the established database
    #flow_query = "SELECT * FROM flow WHERE strftime('%Y', DATETIME) IN ('2019')"
    #flow_df = pd.read_sql(flow_query, engine)
    #flow_df = flow_df.astype({"DATETIME":"datetime64[ns]"})
    
    # If RAM is limited, create a new database with only desired time range
    siteRef_list = flow_meta_df["SITEREF"].to_list()
    flow_df = traffic_flow_import_19("./data/traffic/flow_data_13_20/", siteRef_list, process_engine, False)
    flow_df = flow_df.astype({"DATETIME":"datetime64[ms]"})
    ####################################################################################################
    # Process the data
    total_flow = flow_df.groupby(['SITEREF', 'DATETIME', 'WEIGHT'])['FLOW'].sum().reset_index()
    total_flow = total_flow.rename(columns={'FLOW': 'TOTAL_FLOW'})
    
    flow_df = flow_df.merge(total_flow, on=['SITEREF', 'DATETIME', 'WEIGHT'])
    flow_df['PROPORTION'] = flow_df['FLOW'] / flow_df['TOTAL_FLOW']
    
    # Filter out the rows with proportion higher than 50%, row PROPORTION is the imbalance
    flow_df = flow_df[flow_df['PROPORTION'] > 0.5]
    flow_df = flow_df.drop(columns=['DIRECTION'])
    
    # Select by weight
    light_df = flow_df[flow_df['WEIGHT'] == "Light"]
    heavy_df = flow_df[flow_df['WEIGHT'] == "Heavy"]

    # Analyze the light vehicles
    temp_light_flow_df = light_df[["DATETIME", "SITEREF", "FLOW"]]
    time_index = pd.date_range(start="2019-01-01 00:00:00", end="2019-12-31 23:45:00", freq="15min")
    datetime_df = pd.DataFrame()
    datetime_df["DATETIME"] = time_index
    
    # Select light vehicles for analysis
    light_pivot_df = temp_light_flow_df.pivot(index='DATETIME', columns='SITEREF', values='FLOW')
    light_merged_df = datetime_df.merge(light_pivot_df, on='DATETIME', how='left')

    # Visualize the missing data
    #flow_missing_data_visualization(light_merged_df, "./result/flow/missing")
    
    # Filter based on missing value percentage
    Session = sessionmaker(bind=process_engine)
    session = Session()
    
    filtered_meta_df = flow_missing_filter(flow_meta_df, light_merged_df, 30, process_engine)
    flow_data_imputation(filtered_meta_df, light_df, process_engine)
    
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
    
    station_df = pd.read_excel("./result/flow/imputation/raw.xlsx")
    imputation_visualization(station_df, '2019-12-10 00:00:00', '2019-12-17 00:00:00', 
                                        ["Linear", "Forward", "Backward", "Forward-Backward"],
                                        "00200091",
                                        "./result/flow/imputation/")
    """
    
