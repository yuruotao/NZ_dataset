# Coding: utf-8
# Script for calculating the basic statistics
import pandas as pd
import os
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


def basic_statistics(input_df):
    """Calculate the basic statistics of input_df

    Args:
        input_df (dataframe): dataframe containing data to be computed

    Returns:
        dataframe: the result dataframe
    """
    try:
        input_df = input_df.drop(columns=["DATETIME"])
    except:
        pass
    
    STA_df = pd.DataFrame({'MEAN':input_df.mean(), 
                           'STD':input_df.std(),
                           'SKEW':input_df.skew(),
                           'KURTOSIS':input_df.kurtosis(),
                           'PERCENTILE_0':input_df.quantile(q=0),
                           'PERCENTILE_2_5':input_df.quantile(q=0.025),
                           'PERCENTILE_50':input_df.quantile(q=0.5),
                           'PERCENTILE_97_5':input_df.quantile(q=0.975),
                           'PERCENTILE_100':input_df.quantile(q=1)
                           },
                   dtype = 'float')
    STA_df["INDICATOR"] = input_df.columns
    
    return STA_df

