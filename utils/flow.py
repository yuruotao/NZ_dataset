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
import time

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'

def traffic_flow_import_20_22(input_path, site_path, output_path):
    """Import traffic flow data of 2020 to 2022
    https://opendata-nzta.opendata.arcgis.com/datasets/tms-traffic-quarter-hourly-oct-2020-to-jan-2022/about
    https://opendata-nzta.opendata.arcgis.com/datasets/b90f8908910f44a493c6501c3565ed2d_0

    Args:
        input_path (string): path of traffic flow between 2020 and 2022
        output_path (string): path to save the output

    Returns:
        Dataframe: dataframe containing traffic flow during 2020 and 2022
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Create Datetime dataframe
    time_index = pd.date_range(start="2020-10-01 00:00:00", end="2022-01-31 23:45:00", freq="15min")
    time_series_df = pd.DataFrame({'Datetime': time_index})
    
    # Read all files within the folder
    traffic_flow_list = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    traffic_df = pd.DataFrame()
    for file in traffic_flow_list:
        print(file)
        temp_df = pd.read_csv(input_path + file, encoding='unicode_escape')
        traffic_df = pd.concat([traffic_df, temp_df], ignore_index=True)

    # For each site, save to separate xlsx file
    traffic_df["SITE_REFERENCE"] = traffic_df["SITE_REFERENCE"].apply(lambda x: str(x).zfill(8))
    traffic_df = traffic_df.rename(columns={"START_DATE":"Datetime", "TRAFFIC_COUNT":"Flow", 
                                            "SITE_REFERENCE":"siteRef", "CLASS_WEIGHT":"Weight"})
    traffic_df['Datetime'] = pd.to_datetime(traffic_df['Datetime'])
    traffic_df = traffic_df.groupby(["Weight", "siteRef", "Datetime"])[["Flow"]].sum().reset_index()
    
    """
    traffic_site_gdf = gpd.read_file(site_path)
    
    traffic_site_gdf["siteRef"] = traffic_site_gdf["siteRef"].apply(lambda x: str(x).zfill(8))
    site_set = set(traffic_site_gdf["siteRef"])
    time_site_df = time_series_df
    for site in site_set:
        print(site)
        temp_df = traffic_df[traffic_df["siteRef"] == site]
        if len(temp_df.index) != 0:        
            temp_df = temp_df.astype({"Flow":float})
            # For one set, sum traffic from all directions
            flow_dir_set = set(temp_df["FLOW_DIRECTION"])
            flow = 0
            for flow_site in flow_dir_set:
                merge_df = temp_df[temp_df["FLOW_DIRECTION"] == flow_site]
                merge_df = pd.merge(time_series_df, merge_df, on='Datetime', how="left")
                if flow == 0:
                    dir_temp_df = merge_df
                else:
                    dir_temp_df["Flow"] = dir_temp_df["Flow"] + merge_df["Flow"]
                print(flow)
                flow = flow + 1

            dir_temp_df = dir_temp_df.sort_values(by=["Datetime"])
            dir_temp_df = dir_temp_df.reindex()
            dir_temp_df = dir_temp_df[["Datetime", "Flow"]]
            
            time_site_df[site] = dir_temp_df["Flow"]
            
    time_site_df.to_excel(output_path + ".xlsx", index=False)
    """
    return traffic_df
# https://opendata-nzta.opendata.arcgis.com/datasets/b719083bbb09489087649f1fc03ba53a/about
def traffic_flow_import_13_20(input_path, site_path, output_path):
    
    pass

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
    time1 = time.time()
    traffic_df_20_22 = traffic_flow_import_20_22("./data/traffic/flow_data_20_22/", 
                                           "./data/traffic/traffic_monitor_sites/State_highway_traffic_monitoring_sites.shp",
                                           "./result/flow/aggregated/")
    time2 = time.time-time1
    print(time2)
    print(traffic_df_20_22)
    
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