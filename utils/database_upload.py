# Coding: utf-8
# Create the database NZDB.db from multiple data sources
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
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import xml.etree.ElementTree as ET
import re

Base = declarative_base()

# CTRY STATE ICAO LAT LON ELEV(M) BEGIN END
class weather_meta(Base):
    __tablename__ = 'weather_meta'
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
class weather(Base):
    __tablename__ = 'weather'
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

# SH RS RP siteRef lane type HEAVY_RATIO DESCRIPTION region siteType LAT LON
class flow_meta(Base):
    __tablename__ = 'flow_meta'
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

# Datetime siteRef Flow Weight Direction
class flow(Base):
    __tablename__ = 'flow'
    ID = Column(Integer, primary_key=True, unique=True, nullable=False)
    SITEREF = Column(String)
    DATETIME = Column(DateTime)
    FLOW = Column(Float)
    WEIGHT = Column(String)
    DIRECTION = Column(Integer)

class holiday(Base):
    __tablename__ = 'holiday'
    ID = Column(Integer, primary_key=True, unique=True, nullable=False)
    HOLIDAY = Column(String)
    START_DATE = Column(DateTime)
    STOP_DATE = Column(DateTime)
    TYPE = Column(String)
    REGION = Column(String)
    
class extreme_weather(Base):
    __tablename__ = 'extreme_weather'
    HAZARD = Column(String)
    IDENTIFIER = Column(String, primary_key=True, unique=True, nullable=False)
    ABSTRACT = Column(String)
    START_DATE = Column(DateTime)
    REGION = Column(String)
    LATITUDE = Column(Float)
    LONGITUDE = Column(Float)
    IMPACT = Column(String)
    
def traffic_flow_import_20_22(input_path, siteRef_list, engine):
    """Import traffic flow data of 2020 to 2022
    https://opendata-nzta.opendata.arcgis.com/datasets/tms-traffic-quarter-hourly-oct-2020-to-jan-2022/about
    https://opendata-nzta.opendata.arcgis.com/datasets/b90f8908910f44a493c6501c3565ed2d_0

    Args:
        input_path (string): path of traffic flow between 2020 and 2022
        siteRef_list (list): list contain strings of siteRef
        engine (sqlalchemy_engine): engine used for database creation

    Returns:
        None
    """

    # Read all files within the folder
    traffic_flow_list = [f for f in listdir(input_path) if isfile(join(input_path, f))]
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
        temp_df.columns = temp_df.columns.str.upper()
        temp_df.to_sql('flow', con=engine, if_exists='append', index=False)

    return None

def traffic_flow_import_13_20(input_path, siteRef_list, engine):
    """Import traffic flow data of 2013 to 2020
    https://opendata-nzta.opendata.arcgis.com/datasets/b719083bbb09489087649f1fc03ba53a/about

    Args:
        input_path (string): path of traffic flow between 2020 and 2022
        siteRef_list (list): list contain strings of siteRef
        engine (sqlalchemy_engine): engine used for database creation

    Returns:
        None
    """

    # Read all files within the folder
    traffic_flow_list = [f for f in listdir(input_path) if isfile(join(input_path, f))]
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
        temp_df.columns = temp_df.columns.str.upper()
        temp_df.to_sql('flow', con=engine, if_exists='append', index=False)
    
    return None

def traffic_flow_database_upload(siteRef_list, engine):
    """Upload the flow data to database

    Args:
        siteRef_list (list): list contain strings of siteRef
        engine (sqlalchemy_engine): engine used for database creation

    Returns:
        None
    """
    
    # For year 13 to 20
    traffic_df_13_20 = traffic_flow_import_13_20("./data/traffic/flow_data_13_20/", siteRef_list, engine)

    # For year 20 to 21
    traffic_df_20_21 = traffic_flow_import_20_22("./data/traffic/flow_data_20_22/", siteRef_list, engine)

    return None

def NCDC_weather_data_process(meta_df, data_path, engine):
    """Reformat and impute the missing data of weather data
    Add relative humidity "RH" to the dataframe

    Args:
        data_path (string): path to station data
        output_path (string): path to store the imputed data

    Returns:
        None
    """

    station_list = meta_df["STATION_ID"].to_list()
    station_files = [str(station) + ".xlsx" for station in station_list]
    
    for station_file in station_files:
        print(station_file)
        temp_df = pd.read_excel(data_path + station_file)
        
        station_id = station_file.rstrip(".xlsx")
        
        try: 
            temp_df = temp_df[["Datetime",
                           "TEMP", "DEWP", "SLP", "STP", "VISIB", 
                           "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", 
                           "SNDP"]]
        except:
            temp_list = ["TEMP", "DEWP", "SLP", "STP", "VISIB", 
                           "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", 
                           "SNDP"]
            for col in temp_list:
                temp_df[col] = np.nan
        
        temp_df["Station_ID"] = station_id
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

        temp_df.columns = temp_df.columns.str.upper()
        temp_df.to_sql('weather', con=engine, if_exists='append', index=False)

    return None


if __name__ == "__main__":
    # Create the SQLite database
    engine = create_engine('sqlite:///./data/NZDB/NZDB.db')
    Base.metadata.create_all(engine)
    
    # Create a new session
    Session = sessionmaker(bind=engine)
    session = Session()
    #############################################################################
    # Weather meta
    meta_df = pd.read_csv("./data/isd-history.csv")
    meta_df = meta_df.loc[meta_df["CTRY"]== "NZ"]
    meta_df = meta_df.astype({"BEGIN":int, "END":int, "USAF":str, "WBAN":str, "ELEV(M)":float})
    meta_df = meta_df.rename({"ELEV(M)":"ELEV"}, axis=1)
    meta_df = meta_df.loc[meta_df["END"]>2022*10000]
    meta_df["WBAN"] = meta_df["WBAN"].str.zfill(5)
    station_str = meta_df["USAF"] + meta_df["WBAN"]
    meta_df["Station_ID"] = station_str
    meta_df = meta_df.reset_index(drop=True)
    meta_df.columns = meta_df.columns.str.upper()
    #meta_df.to_sql('weather_meta', con=engine, if_exists='replace', index=False)
    
    # Weather
    #NCDC_weather_data_process(meta_df, "./result/weather/stations/", engine)
    
    #############################################################################
    # Flow meta
    flow_meta_gdf = gpd.read_file("./data/traffic/traffic_monitor_sites/State_highway_traffic_monitoring_sites.shp")
    flow_meta_gdf = flow_meta_gdf.to_crs("EPSG:4167")
    flow_meta_gdf["siteRef"] = flow_meta_gdf["siteRef"].apply(lambda x: str(x).zfill(8))
    flow_meta_gdf["LAT"] = flow_meta_gdf.geometry.y
    flow_meta_gdf["LON"] = flow_meta_gdf.geometry.x
    
    # OBJECTID SH RS RP siteRef lane type percentHea equipmentC descriptio region acceptedDa AADT5years AADT4years AADT3years AADT2years AADT1yearA siteType geometry LAT LON
    flow_meta_gdf = flow_meta_gdf[["SH", "RS", "RP", "siteRef", "lane", "type", "percentHea", "descriptio", "region", "siteType", "LAT", "LON"]]
    flow_meta_gdf = flow_meta_gdf.rename({"percentHea":"HEAVY_RATIO", "descriptio":"DESCRIPTION"}, axis=1)
    flow_meta_gdf.columns = flow_meta_gdf.columns.str.upper()
    #flow_meta_gdf.to_sql('flow_meta', con=engine, if_exists='replace', index=False)
    
    # Flow
    #siteRef_list = flow_meta_gdf["SITEREF"].to_list()
    #traffic_flow_database_upload(siteRef_list, engine)
    #############################################################################
    # Extreme weather
    # https://hwe.niwa.co.nz/search/summary/Startdate/01-01-2013/Enddate/31-01-2022/Regions/all/Hazards/all/Impacts/all/Keywords/none/numberOfEvents/20/page/1#/
    namespaces = {
    'exist': 'http://exist.sourceforge.net/NS/exist',
    'hwe': 'http://hwe.niwa.co.nz/schema/2011',
    'gml': 'http://www.opengis.net/gml',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }
    
    tree = ET.parse('./data/extreme_weather/extreme.xml')
    root = tree.getroot()
    # Use relative XPath to find all WeatherEvent elements
    temp_weather_events = root.findall('.//hwe:WeatherEvent', namespaces)
    
    tree_2 = ET.parse('./data/extreme_weather/extreme2.xml')
    root_2 = tree_2.getroot()
    weather_events = root_2.findall('.//hwe:WeatherEvent', namespaces)
    
    weather_events = weather_events + temp_weather_events
    
    extreme_weather_df = pd.DataFrame()
    HAZARD_list = []
    IDENTIFIER_list = []
    ABSTRACT_list = []
    START_DATE_list = []
    REGION_list = []
    LATITUDE_list = []
    LONGITUDE_list = []
    IMPACT_list = []
    for event in weather_events:
        identifier = event.find('hwe:Identifier', namespaces).text
        start_date = event.find('hwe:StartDate', namespaces).text
        abstract = event.find('hwe:Abstract', namespaces).text
        
        regions_list = event.findall('hwe:Regions/hwe:Region', namespaces)
        for region_element in regions_list:
            hazards_list = region_element.findall('hwe:Hazards/hwe:Hazard', namespaces)
            for hazard_element in hazards_list:
                hazard = hazard_element.attrib.get("type")
                small_region_element = hazard_element.find('hwe:Location', namespaces)
                region = small_region_element.attrib.get("name")
                lat_lon = small_region_element.find("gml:Point/gml:pos", namespaces)
                if lat_lon == None:
                    lat = None
                    lon = None
                else:
                    pattern = r'\d+\.\d+'
                    # Find all numbers in the string
                    numbers = re.findall(pattern, lat_lon.text)
                    # Convert the found numbers from strings to integers
                    lon, lat = map(float, numbers)
                
                impact = ""
                impact_element_list = hazard_element.findall('hwe:Impacts/hwe:Impact', namespaces)
                for impact_element in impact_element_list:
                    temp_impact = impact_element.text.lstrip(" ").strip("\n").strip("\t")
                    if temp_impact[-1] != ".":
                        temp_impact = temp_impact + "."
                    temp_impact = re.sub(r'\.(?!\s)', '. ', temp_impact)
                    temp_impact = re.sub(r'\s+', ' ', temp_impact)
                    temp_impact = temp_impact.strip()
                    impact = impact + temp_impact
                
                HAZARD_list.append(hazard)
                IDENTIFIER_list.append(identifier)
                ABSTRACT_list.append(abstract)
                START_DATE_list.append(start_date)
                REGION_list.append(region)
                LATITUDE_list.append(lat)
                LONGITUDE_list.append(lon)
                IMPACT_list.append(impact)

    extreme_weather_df["HAZARD"] = HAZARD_list
    extreme_weather_df["IDENTIFIER"] = IDENTIFIER_list
    extreme_weather_df["ABSTRACT"] = ABSTRACT_list
    extreme_weather_df["START_DATE"] = START_DATE_list
    extreme_weather_df["REGION"] = REGION_list
    extreme_weather_df["LATITUDE"] = LATITUDE_list
    extreme_weather_df["LONGITUDE"] = LONGITUDE_list
    extreme_weather_df["IMPACT"] = IMPACT_list
    extreme_weather_df.to_sql('extreme_weather', con=engine, if_exists='replace', index=False)

    #############################################################################
    # Holiday
    # https://www.employment.govt.nz/leave-and-holidays/public-holidays/previous-years-public-holidays-and-anniversary-dates#/
    holiday_df = pd.read_excel("./data/holiday/holiday.xlsx")
    holiday_df["Observed date"] = holiday_df["Observed date"].str.strip("*")
    holiday_df["Observed date"] = holiday_df["Observed date"].str.strip(";")
    holiday_df["Observed date"] = holiday_df["Observed date"].str.rstrip(" ")
    
    start_date_list = []
    stop_date_list = []
    for index, row in holiday_df.iterrows():
        year = row["Year"]
        
        date = row["Observed date"].split(" or ")
        if len(date) > 1:
            start_date = date[0]
            stop_date = date[1]
            
            start_date = start_date.split(" ")
            start_month = start_date[-1]
            start_day = start_date[1]

            stop_date = stop_date.split(" ")
            stop_month = stop_date[-1]
            stop_day = stop_date[1]
        else:
            start_date = date[0]
            start_date = start_date.split(" ")
            start_month = start_date[-1]
            start_day = start_date[1]
            
            stop_day = start_day
            stop_month = start_month
        
        start_date_string = f"{start_month} {start_day}, {year}"
        start_date_object = datetime.strptime(start_date_string, "%B %d, %Y")
        start_date_list.append(start_date_object)

        stop_date_string = f"{stop_month} {stop_day}, {year}"
        stop_date_object = datetime.strptime(stop_date_string, "%B %d, %Y")
        stop_date_list.append(stop_date_object)
        
    holiday_df["start_date"] = start_date_list
    holiday_df["stop_date"] = stop_date_list
    
    def update_region_and_holiday(row):
        if row['Type'] == 'Public holiday':
            row['region'] = 'all'
        else:
            row['region'] = row['Holiday']
            row['Holiday'] = 'Regional anniversary'
        return row
    
    holiday_df = holiday_df.apply(update_region_and_holiday, axis=1)
    holiday_df = holiday_df[["Holiday", "Type", "start_date", "stop_date", "region"]]
    holiday_df.columns = holiday_df.columns.str.upper()
    holiday_df.to_sql('holiday', con=engine, if_exists='replace', index=False)

    session.commit()
    session.close()