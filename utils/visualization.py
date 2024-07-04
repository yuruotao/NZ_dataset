# Coding: utf-8
# Visualization functions
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'

def df_to_gdf(df, lon_name, lat_name):
    
    geometry = [Point(xy) for xy in zip(df[lon_name], df[lat_name])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs(epsg=4167, inplace=True)

    return gdf

def distribution_visualization(flow_meta_df, weather_meta_df, highway_shp_path, boundary_shp_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    flow_meta_gdf = df_to_gdf(flow_meta_df, "LON", "LAT")
    weather_meta_gdf = df_to_gdf(weather_meta_df, "LON", "LAT")
    highway_shp = gpd.read_file(highway_shp_path)
    boundary_shp = gpd.read_file(boundary_shp_path)
    # Separate Chatham Islands Territory with main island
    boundary_main_shp = boundary_shp[boundary_shp["TA2023_V1_"] != "067"]
    boundary_Chatham_shp = boundary_shp[boundary_shp["TA2023_V1_"] == "067"]

    # Plot main land
    plt.figure(figsize=(20, 20))
    boundary_main_shp.boundary.plot(linewidth=0.3, color="#adb5bd", zorder=1)
    bounds = boundary_main_shp.total_bounds
    ax = plt.gca()
    
    ax.set_aspect('equal')
    extent = 0.5
    ax.set_xlim(bounds[0]-extent, bounds[2]+extent)
    ax.set_ylim(bounds[1]-extent, bounds[3]+extent)
    ax.axis('off')
    
    highway_shp.plot(ax=ax, color='#000000', linewidth=0.5, aspect="equal", zorder=1)
    for index, row in flow_meta_gdf.iterrows():
        ax.scatter(row.geometry.x, row.geometry.y, s=5, color='#a4161a', linewidths=0.1, edgecolor='#000000', alpha=1)
    
    for index, row in weather_meta_gdf.iterrows():
        ax.scatter(row.geometry.x, row.geometry.y, s=5, color='#fca311', linewidths=0.1, edgecolor='#000000', alpha=1)
    
    circle_2 = plt.Circle((bounds[0]+1, bounds[3]-2.2), radius=0.2, color='#a4161a', alpha=1)
    ax.add_artist(circle_2)
    ax.text(bounds[0]+1+0.8, bounds[3]-2.2, 'Flow', fontsize=12, ha='left', va='center')
        
    circle_3 = plt.Circle((bounds[0]+1, bounds[3]-1.2), radius=0.2, color='#fca311', alpha=1)
    ax.add_artist(circle_3)
    ax.text(bounds[0]+1+0.8, bounds[3]-1.2, 'Weather', fontsize=12, ha='left', va='center')
    
    plt.savefig(output_path + "distribution_main.png", dpi=900)
    plt.close()
    
    # Plot Chatham
    plt.figure(figsize=(20, 20))
    boundary_Chatham_shp.boundary.plot(linewidth=0.5, color="#adb5bd", zorder=1)
    bounds = boundary_Chatham_shp.total_bounds
    ax = plt.gca()
    
    ax.set_aspect('equal')
    extent = 0
    ax.set_xlim(bounds[0]-extent, bounds[2]-extent)
    ax.set_ylim(bounds[1]+extent, bounds[3]+extent)
    ax.axis('off')
    
    highway_shp.plot(ax=ax, color='#000000', linewidth=1, aspect="equal", zorder=1)
    for index, row in flow_meta_gdf.iterrows():
        ax.scatter(row.geometry.x, row.geometry.y, s=10, color='#a4161a', linewidths=0.1, edgecolor='#000000', alpha=1)
    
    for index, row in weather_meta_gdf.iterrows():
        ax.scatter(row.geometry.x, row.geometry.y, s=10, color='#fca311', linewidths=0.1, edgecolor='#000000', alpha=1)
    
    plt.savefig(output_path + "distribution_Chatham.png", dpi=900)
    plt.close()
    

    return None

def direction_percentage_visualization():
    
    
    
    return None