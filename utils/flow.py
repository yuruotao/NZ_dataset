import pandas as pd
import geopandas as gpd
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.ops import snap
import seaborn as sns
from sqlalchemy import create_engine
import time

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'


def traffic_flow_import_20_22(input_path, siteRef_list, engine):
    """Import traffic flow data of 2020 to 2022
    https://opendata-nzta.opendata.arcgis.com/datasets/tms-traffic-quarter-hourly-oct-2020-to-jan-2022/about
    https://opendata-nzta.opendata.arcgis.com/datasets/b90f8908910f44a493c6501c3565ed2d_0

    Args:
        input_path (string): path of traffic flow between 2020 and 2022
        output_path (string): path to save the output

    Returns:
        Dataframe: dataframe containing traffic flow during 2020 and 2022
    """

    # Read all files within the folder
    traffic_flow_list = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    traffic_df = pd.DataFrame()
    for file in traffic_flow_list:
        print(file)
        temp_df = pd.read_csv(input_path + file, encoding='unicode_escape')
        # START_DATE SITE_ALIAS REGION_NAME SITE_REFERENCE CLASS_WEIGHT SITE_DESCRIPTION LANE_NUMBER FLOW_DIRECTION TRAFFIC_COUNT
        temp_df["SITE_REFERENCE"] = temp_df["SITE_REFERENCE"].apply(lambda x: str(x).zfill(8))
        temp_df = temp_df[temp_df["SITE_REFERENCE"].isin(siteRef_list)]
        temp_df = temp_df.rename(columns={"START_DATE":"Datetime", "TRAFFIC_COUNT":"Flow", 
                                                "SITE_REFERENCE":"siteRef", "CLASS_WEIGHT":"Weight", 
                                                "FLOW_DIRECTION":"Direction"})
        temp_df = temp_df[["Datetime", "siteRef", "Flow", "Weight", "Direction"]]
        temp_df['Datetime'] = pd.to_datetime(temp_df['Datetime'])
        temp_df = temp_df.groupby(["Datetime", "siteRef", "Weight", "Direction"], as_index=False)[["Flow",]].sum().reset_index(drop=True)
        with engine.begin() as connection:
            temp_df.to_sql(name='users', con=connection, if_exists='append')
        
            
        traffic_df = pd.concat([traffic_df, temp_df], ignore_index=True)
    
    return traffic_df

def traffic_flow_import_13_20(input_path, siteRef_list):
    # 
    """Import traffic flow data of 2013 to 2020
    https://opendata-nzta.opendata.arcgis.com/datasets/b719083bbb09489087649f1fc03ba53a/about

    Args:
        input_path (string): path of traffic flow between 2013 and 2020
        output_path (string): path to save the output

    Returns:
        Dataframe: dataframe containing traffic flow during 2013 and 2020
    """

    # Read all files within the folder
    traffic_flow_list = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    traffic_df = pd.DataFrame()
    for file in traffic_flow_list:
        print(file)
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
    
    
    return 

def traffic_flow_siteRef(siteRef_list):
    engine = create_engine('sqlite:///./result/traffic/database/traffic_data.db')
    
    # For year 13 to 20
    #traffic_df_13_20 = traffic_flow_import_13_20("./data/traffic/flow_data_13_20/", siteRef_list)
    #print(traffic_df_13_20)
    
    # For year 20 to 21
    traffic_df_20_21 = traffic_flow_import_20_22("./data/traffic/flow_data_20_22/", siteRef_list, engine)
    #print(traffic_df_20_21)
    
    #traffic_df = pd.concat([traffic_df_13_20, traffic_df_20_21], ignore_index=True)
    #traffic_df.to_excel("./result/traffic/flow_data_13_21.xlsx", index=False)

    return



def traffic_missing_data_visualization(input_path, output_path):
    """Visualize the missing data of flow data

    Args:
        input_path (string): path to the raw data
        output_path (string): path to save the figure

    Returns:
        None
    """
    missing_matrix_path = output_path + "/matrix/"
    missing_bar_path = output_path + "/bar/"
    
    if not os.path.exists(missing_matrix_path):
        os.makedirs(missing_matrix_path)
    
    if not os.path.exists(missing_bar_path):
        os.makedirs(missing_bar_path)
    
    flow_df = pd.read_excel(input_path)
    temp_flow_df = flow_df.set_index("Datetime")
    
    # Divide into chunks
    chunks = [temp_flow_df.iloc[:, i:i+20] for i in range(0, len(temp_flow_df.columns), 20)]
    
    # Missing data visualization
    index = 0
    for chunk in chunks:
        print(index)
        # Matrix plot
        ax = msno.matrix(chunk, fontsize=20, figsize=(20, 14), label_rotation=45, freq="M")
        plt.xlabel("Traffic Monitoring Sites", fontsize=20)
        plt.ylabel("Sample Points", fontsize=20)
        plt.savefig(missing_matrix_path + 'matrix_' + str(index) + '.png', dpi=600)
        plt.close()
        
        # Bar plot
        ax = msno.bar(chunk, fontsize=20, figsize=(16, 12), label_rotation=45)
        plt.xlabel("Traffic Monitoring Sites", fontsize=20)
        plt.ylabel("Sample Points", fontsize=20)
        plt.savefig(missing_bar_path + 'bar_' + str(index) + '.png', dpi=600)
        plt.close()
        
        index = index + 1
    return None

def traffic_missing_filter(input_path, threashold, output_path):
    """Delete the stations whose missing data percentage reach the threashold

    Args:
        input_path (string): path to the raw data
        threashold (float): threashold for deletion
        output_path (string): path to save the deleted raw data

    Returns:
        dataframe: raw data deleted
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    raw_df = pd.read_excel(input_path)
    # Calculate percentage of missing values in each column
    missing_percentages = raw_df.isna().mean() * 100

    # Drop columns where the percentage of missing values exceeds 20%
    columns_to_drop = missing_percentages[missing_percentages > threashold].index
    processed_df = raw_df.drop(columns=columns_to_drop)
    processed_df.to_excel(output_path + "deleted.xlsx", index=False)
    
    return processed_df

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
    datetime_column = input_df["Datetime"]
    input_df = input_df.drop(columns=["Datetime"])
    
    imputation_dir = save_path
    
    if not os.path.exists(imputation_dir):
        os.makedirs(imputation_dir)
        
    if imputation_method == "Linear":
        imputed_df = input_df.interpolate(method='linear')
       
    elif imputation_method == "Forward-Backward":
        forward_df = input_df.shift(-7*24)
        backward_df = input_df.shift(7*24)
        
        average_values = (forward_df + backward_df) / 2
        temp_df = input_df.copy()
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.notna()] = average_values[temp_df.isna() & forward_df.notna() & backward_df.notna()]
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.isna()] = forward_df[temp_df.isna() & forward_df.notna() & backward_df.isna()]
        temp_df[temp_df.isna() & backward_df.notna() & forward_df.isna()] = backward_df[temp_df.isna() & backward_df.notna() & forward_df.isna()]

        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('Datetime', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['Datetime'].dt.dayofweek
            df_grouped['time'] = df_grouped['Datetime'].dt.time
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

        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime"])

    elif imputation_method == "Forward":
        imputed_df = input_df.fillna(input_df.shift(-7*24))
        temp_df = imputed_df
        
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('Datetime', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['Datetime'].dt.dayofweek
            df_grouped['time'] = df_grouped['Datetime'].dt.time
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

        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime"])
        
    elif imputation_method == "Backward":
        imputed_df = input_df.fillna(input_df.shift(7*24))
        temp_df = imputed_df
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('Datetime', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['Datetime'].dt.dayofweek
            df_grouped['time'] = df_grouped['Datetime'].dt.time
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

        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime",])
    
    elif imputation_method == "Average":
        temp_df = input_df
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('Datetime', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['Datetime'].dt.dayofweek
            df_grouped['time'] = df_grouped['Datetime'].dt.time
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

        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime",])
    
    print(imputed_df)
    imputed_df = pd.concat([datetime_column, imputed_df], axis=1)
    imputed_df.to_excel(imputation_dir + "/" + "imputed_data_" + imputation_method + ".xlsx", index=False)
    
    return imputed_df

def imputation_visualization(raw_data_df, start_time, end_time, method_list, column, output_path):
    """Visualize the imputation result, comparing methods

    Args:
        raw_data_df (dataframe): contain the raw data
        start_time (string): plot start time
        end_time (string): plot stop time
        method_list (list): contain the imputation methods
        column (string): contain the transformer index
        output_path (string): path to save the figure

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    raw_data_df = raw_data_df.loc[(raw_data_df['Datetime'] >= start_time) & (raw_data_df['Datetime'] <= end_time)]
    raw_data_df = raw_data_df.rename(columns={"Flow":"Raw"})
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'Datetime': time_index})
    for method in method_list:
        temp_df = pd.read_excel(output_path + "imputed/imputed_data_" + method +".xlsx")
        temp_df = temp_df[["Datetime", column]]
        temp_df = temp_df.loc[(temp_df['Datetime'] >= start_time) & (temp_df['Datetime'] <= end_time)]
        temp_df = temp_df.rename(columns={column:method})
        time_series_df = pd.merge(time_series_df, temp_df, on='Datetime', how="left")
        
    time_series_df = pd.merge(time_series_df, raw_data_df, on='Datetime', how="left")
    time_series_df = time_series_df.set_index("Datetime")
    print(time_series_df)

    plt.figure(figsize=(20, 12))
    ax = sns.lineplot(data=time_series_df, markers=True, linewidth=4)
    missing_mask = time_series_df['Raw'].isna().values.astype(int)
    ax.set_xlim(time_series_df.index[0], time_series_df.index[-1])
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  missing_mask[np.newaxis], cmap='Blues', alpha=0.2)
    
    # Set x-axis limits
    plt.rc('legend', fontsize=22)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='lower left', mode="expand", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=5)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.tick_params(labelsize=22)
    plt.xlabel("Time", fontsize=22)
    plt.ylabel("Traffic Counts", fontsize=22)
        
    #plt.tight_layout()
    plt.savefig(output_path + "imputation_methods.png", dpi=600)
    plt.close()
    
    return None


if __name__ == "__main__":
    
    traffic_site_gdf = gpd.read_file("./data/traffic/traffic_monitor_sites/State_highway_traffic_monitoring_sites.shp")
    traffic_site_gdf["siteRef"] = traffic_site_gdf["siteRef"].apply(lambda x: str(x).zfill(8))
    siteRef_list = traffic_site_gdf["siteRef"].to_list()
    traffic_flow_siteRef(siteRef_list)
    
    
    
    """
    time1 = time.time()
    traffic_df_20_22 = traffic_flow_import_20_22("./data/traffic/flow_data_20_22/", 
                                           "./data/traffic/traffic_monitor_sites/State_highway_traffic_monitoring_sites.shp",
                                           "./result/flow/aggregated/")
    time2 = time.time-time1
    print(time2)
    print(traffic_df_20_22)
    """
    """
    heavy_traffic_df = traffic_flow_import_20_22("./data/traffic/flow_data/", 
                                           "./data/traffic/traffic_monitor_sites/traffic_monitor_sites.shp",
                                           "Heavy",
                                           "./result/flow/aggregated/")
    
    traffic_missing_data_visualization("./result/flow/aggregated/Light/Light.xlsx", "./result/flow/missing_data/Light")
    traffic_missing_data_visualization("./result/flow/aggregated/Heavy/Heavy.xlsx", "./result/flow/missing_data/Heavy")
    
    processed_traffic_df = traffic_missing_filter("./result/flow/aggregated/Light/Light.xlsx", 30, 
                                                  "./result/flow/deleted/Light/")
    
    processed_traffic_df = traffic_missing_filter("./result/flow/aggregated/Heavy/Heavy.xlsx", 30, 
                                                  "./result/flow/deleted/Heavy/")
    
    # Traffic data imputation
    input_df = pd.read_excel("./result/flow/deleted/Heavy/deleted.xlsx")
    #imputation(input_df, "Linear", "./result/flow/imputed/Heavy/")
    #imputation(input_df, "Forward-Backward", "./result/flow/imputed/Heavy/")
    #imputation(input_df, "Forward", "./result/flow/imputed/Heavy/")
    #imputation(input_df, "Backward", "./result/flow/imputed/Heavy/")
    #imputation(input_df, "Average", "./result/flow/imputed/Heavy/")
    
    input_df = pd.read_excel("./result/flow/deleted/Light/deleted.xlsx")
    #imputation(input_df, "Linear", "./result/flow/imputed/Light/")
    #imputation(input_df, "Forward-Backward", "./result/flow/imputed/Light/")
    #imputation(input_df, "Forward", "./result/flow/imputed/Light/")
    #imputation(input_df, "Backward", "./result/flow/imputed/Light/")
    #imputation(input_df, "Average", "./result/flow/imputed/Light/")
    """
    """
    raw_data_df = pd.read_excel("./result/example/traffic/sites/Light/traffic_flow_00600547.xlsx")
    imputation_visualization(raw_data_df, '2021-09-25 00:00:00', '2021-09-28 00:00:00', 
                                        ["Linear", "Forward", "Backward", "Forward-Backward"],
                                        "00600547",
                                        "./result/example/traffic/")
    """