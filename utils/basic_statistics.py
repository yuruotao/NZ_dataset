import pandas as pd
import os
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})


def basic_statistics(input_df, save_path):
    """Calculate the basic statistics of input_df

    Args:
        input_df (dataframe): dataframe containing data to be computed
        save_path (string): path to save the basic statistics

    Returns:
        None
    """
    datetime_column = input_df["Datetime"]
    input_df = input_df.drop(columns=["Datetime"])
    column_list = input_df.columns.values.tolist()

    stats_dir = save_path + "/"
    
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    STA_df = pd.DataFrame({'Mean':input_df.mean(), 
                           'Standard deviation':input_df.std(),
                           'Skew':input_df.skew(),
                           'Kurtosis':input_df.kurtosis(),
                           '0th percentile':input_df.quantile(q=0),
                           '2.5th percentile':input_df.quantile(q=0.025),
                           '50th percentile':input_df.quantile(q=0.5),
                           '97.5th percentile':input_df.quantile(q=0.975),
                           '100th percentile':input_df.quantile(q=1)
                           },
                   dtype = 'float')

    STA_df = STA_df.round(2)
    print(STA_df)
    STA_df.to_excel(stats_dir + '/basic_statistics.xlsx', float_format='%.2f')
    
    return None

