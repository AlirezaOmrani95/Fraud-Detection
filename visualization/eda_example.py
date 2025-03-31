'''
File: eda_plots.py

Author: Ali Reza (ARO) Omrani
Email: omrani.alireza95@gmail.com
Date: 4th March 2025

Description:
------
This file contains examples of loading functions from EDA_plots.py.

Requirements:
------
- pandas

'''
import pandas as pd
from eda_plots import *
from utils.data_prerpocessing import pca

if __name__ == '__main__':

    data = pd.read_csv("./dataset4.csv")
    print(data.describe())
    
    #histogram
    histogram(data,'category',30)
    
    #boxplot
    box_plot(data,'amt','category')

    #scatter plots
    scatter_plot(data, 'amt', 'is_fraud')

    #heatmaps
    target_columns = ['lat','long','merch_lat','merch_long','zip']
    data = pca(data,target_columns,3)
    data = data.drop(['trans_date_trans_time','merchant','category','gender','city','state','job','dob',
                      'unix_time','trans_num','lat','long','merch_lat','merch_long','zip'],axis=1)
    heatmaps(data)

    #pairplots
    data = data.drop(['trans_date_trans_time','merchant','category','gender','city','state','job','dob',
                      'unix_time','trans_num','lat','long','merch_lat','merch_long','zip'],axis=1)
    pair_plot(data)
    
    #barplots
    bar_plot(data,'category')
    
    #lineplots
    line_plot(data,'trans_date_trans_time', 'amt')
    
    #violinplots
    violet_plot(data, 'category','amt')

    #countplots
    count_plot(data,'category')

    #piecharts
    piechart(data,'gender')

    # import numpy as np
    # import pandas as pd
    # from sklearn.preprocessing import StandardScaler
    # import matplotlib.pyplot as plt

    # # Generating skewed data
    # data = np.random.exponential(scale=2, size=(1000, 1))
    # df = pd.DataFrame(data, columns=['SkewedData'])

    # # Before Standardization
    # plt.hist(df['SkewedData'], bins=30, edgecolor='k', alpha=0.7)
    # plt.title('Original Skewed Data')
    # plt.show()

    # # Applying StandardScaler
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(df)
    # df['ScaledData'] = scaled_data

    # # After Standardization
    # plt.hist(df['ScaledData'], bins=30, edgecolor='k', alpha=0.7)
    # plt.title('Standardized Skewed Data')
    # plt.show()
