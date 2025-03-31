'''
File: data_preprocessing.py

Author: Ali Reza (ARO) Omrani
Email: omrani.alireza95@gmail.com
Date: 17th March 2025

Description:
------
This file contains data pre processing functions.

Functions:
- embed_func(column_name, file, type): Embed the feature of df file based on the specific type (hash or label encoder).
- crossed_feature(column1,column2,new_name): Producing a new feature usign crossed feature.
- pca(data, columns, components_num): Calculating PCA from mentioned columns.
- is_null(file): Checking null values in the dataset.
- is_duplicated(file, drop = True): checking any duplication in the dataset.
- date_separation(file,origin_column,col_list): Producing years, months, ... from a date feature.

Requirements:
------
- pandas
- numpy
- torch
- sklearn
- hashlib

Usage Example:
------
>>> from utils.data_processing import embed_func
>>> embed_func('cc_num',file,'hash')

Notes:
------
- Functions are designed to handle pd.Dataframes.

'''

import pandas as pd
import numpy as np
from torch import nn
import torch
from sklearn.preprocessing import Normalizer,OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import hashlib

def embed_func(column_name, file, conversion_type = 'hash'):
    '''
    Embedding features using torch.nn.Embedding layer.

    Parameters:
        column_name(str): The name of the feature to embed. 
        file(pd.Dataframe): The pd.Dataframe containing the data.
        conversion_type(str): The type of feature conversion. 

    Returns:
        pd.Dataframe: pd.Dataframe with new features.

    Example:
        >>> new_feature_1 = embed_func('cc_num',file,'hash')
        >>> new_feature_2 = embed_func('merchant', file,'lblencode')

    Note:
    -----
        - The options for conversion_type are "hash" and "lblencode". Default value is "hash".
    '''
    
    column = file[column_name]
    num_unique = column.nunique()
    new_features = {}
    if conversion_type == 'hash':
        column = torch.tensor(column.astype(str).apply(lambda x: int(hashlib.sha256(str(x).encode()).hexdigest(),16) % (num_unique)).values,dtype=torch.long)
    elif conversion_type == "lblencode":
        label_encoder = LabelEncoder()
        column = torch.tensor(label_encoder.fit_transform(column),dtype=torch.long)
        num_unique = len(column.unique())
    
    embedding = nn.Embedding(num_unique,min(50,int(num_unique/2)))
    embedded_features = embedding(column).reshape(len(column),-1)
    for i in range(min(50,int(num_unique/2))):
        new_features[column_name+'_dim_'+str(i)] = embedded_features[:,i].detach().numpy()
    file = pd.concat([file.drop(columns=[column_name]),pd.DataFrame(new_features)])
    return file

def crossed_feature(column1,column2,new_name,file):
    '''
    Creating new feature using crossed_feature method.

    Parameters:
        column1(str): The name of the first feature to embed. 
        column2(str): The name of the second feature to embed. 
        new_name(str): The name of the new feature. 

    Returns:
        pd.Dataframe: pd.Dataframe with crossed_feature.

    Example:
        >>> new_feature_1 = crossed_feature(file['lat'],file['long'],'owner_loc_embed')
        >>> new_feature_2 = crossed_feature(file['merch_lat'],file['merch_long'],'merch_loc_embed')

    '''
        
    features = {}
    lbl_encoder = LabelEncoder()
    new_column = file[column1].astype(str) + '_' + file[column2].astype(str)
    new_column_encoder = torch.tensor(lbl_encoder.fit_transform(new_column), dtype=torch.long)
    num_unique = new_column.nunique()
    embedding = nn.Embedding(num_unique,min(50,int(num_unique/2)))
    embedding_features = embedding(new_column_encoder).reshape(len(file[column1]),-1)

    for i in range(min(50,int(num_unique/2))):
        features[new_name+f'_dim_{i}'] = embedding_features[:,i].detach().numpy()
    
    file = pd.concat([file.drop(columns=[column1,column2]),pd.DataFrame(features)])
    return file

def pca(data, columns, components_num = 2):
    '''
    Principal Component Analysis (PCA) calculations.

    Parameters:
        data (pd.Dataframe): The pd.Dataframe containing the data. 
        column (list): The list of columns considered for the PCA. 
        components_num (int): Number of components to keep. Default is set to 2.

    Returns:
        data (pd.Dataframe): Updated dataframe with caluclated components.

    Example:
        >>> data = pca(data,target_columns,2)
    '''

    pca_ = PCA(n_components=components_num)
    spatial_data = data[columns]
    pca_columns = []
    for i in range(1,components_num+1):
        pca_columns.append(f'pca{i}')

    data[pca_columns] = pca_.fit_transform(spatial_data)
    data = data.drop(columns = columns)
    return data

def is_null(file):
    '''
    Checking null values in the pd.Dataframe

    Parameters:
        file(pd.Dataframe): The pd.Dataframe containing the data.

    Returns:
        None

    Example:
        >>> is_null(file)
    '''

    null_info = file.isnull().sum()
    if null_info.sum()>0:
        print(null_info)
        for head in file.columns:
            if null_info[head]>0:
                file[head] = file[head].fillna(file[head].mean())
    
def is_duplicated(file, drop = True):
    '''
    Checking duplicate values in the pd.Dataframe

    Parameters:
        file(pd.Dataframe): The pd.Dataframe containing the data.
        drop(bool): Drop duplicated values or not. Default value is True.
    Returns:
        None

    Example:
        >>> is_duplicated(file, drop)
    '''

    if file.duplicated().sum()>0:
        if drop:
            file.drop_duplicates()
            print(f'{file.duplicated().sum()} duplicated values are dropped.')
        else:
            print(f'{file.duplicated().sum()} duplicated values.')

def date_separation(file,origin_column,col_list):
    '''
    Separating dates into specific times.

    Parameters:
        file(pd.Dataframe): The pd.Dataframe containing the data.
        origin_column(str): The name of the target feature containing dates. 
        col_list(lst of str): List of specific times to separate.

    Returns:
        pd.Dataframe: pd.Dataframe with new features.

    Example:
        >>> col_list = ['year','month','day','hour','minute','second']
        >>> file = date_separation(file,'trans_date_trans_time',col_list)

    Note:
    -----
        - The options for col_list are various from year to second.
    '''
    
    for item in col_list:
        file[origin_column+'_'+item] = getattr(file[origin_column].dt, item)
    file = file.drop(columns = [origin_column])
    return file

if __name__ == '__main__':
    address = r"D:\Job presentations\Schwarz\fraud\dataset4.csv"
    file = pd.read_csv(address)
    
    '''Checking null values...'''
    is_null(file)

    '''Checking duplications...'''
    is_duplicated(file)

    '''Date separation'''
    file['trans_date_trans_time'] = pd.to_datetime(file['trans_date_trans_time'])
    file = date_separation(file,'trans_date_trans_time',['year','month','day','hour','minute','second'])
    
    '''Age calculation'''
    file['dob'] = pd.to_datetime(file['dob'])
    file['age'] = file['dob'].apply(lambda x: pd.Timestamp('now').year - x.year - ((pd.Timestamp('now').month, pd.Timestamp('now').day) < (x.month, x.day)))
    
    bins = [0,20,30,40,50,60,70,80,100]
    labels = ['0-19','20-29','30-39','40-49','50-59','60-69','70-79','80+']

    file['age_group'] = pd.cut(file['age'],bins=bins,labels=labels,right=True)
    file = file.drop(columns=['dob','age'])
    
    '''Embedding'''
    cc_number_embed = embed_func('cc_num',file,'hash')
    trans_num_embed = embed_func('trans_num',file,'hash')
    merchant_embed = embed_func('merchant', file,'lblencode')
    city_embed = embed_func('city', file,'lblencode')
    zip_embed = embed_func('zip', file,'lblencode')
    job_embed = embed_func('job',file,'lblencode')


    '''Calculating PCA for columns with some possible relationships'''
    pca_cols = ['lat','long','merch_lat','merch_long']

    file = pca(file,pca_cols)
    
    '''concatenating new features'''
    file = pd.concat([file, cc_number_embed, merchant_embed, city_embed, zip_embed, job_embed,trans_num_embed],axis=1)
    
    '''dropping unhelpful features'''
    file = file.drop(columns = ['unix_time'])

    categorical_heads = ['category','gender','state', 'trans_hour',
                         'trans_month','trans_day','trans_minute','trans_second','age_group']
    numerical_heads = ['city_pop','amt']  

    '''Processing one hot encoding features...'''
    column_transformer = ColumnTransformer(
        transformers = [
            ('cat', OneHotEncoder(sparse_output=False), categorical_heads),
        ],
        remainder='passthrough', #Keep the other columns as they are
        force_int_remainder_cols = False
    )   

    encoded_df = column_transformer.fit_transform(file)
    
    cat_encoded_feature_names = column_transformer.named_transformers_['cat'].get_feature_names_out(
        categorical_heads)

    all_feature_names = list(cat_encoded_feature_names) + column_transformer.transformers_[1][2] #+ numerical_heads
    encoded_file = pd.DataFrame(encoded_df, columns=all_feature_names)
    
    '''Skewness correction'''
    for head in numerical_heads:
        file[head] = np.log(file[head])
    
    '''Normalizing...'''
    column_transformer = column_transformer = ColumnTransformer(
        transformers = [
            ('num', Normalizer(), numerical_heads),
        ],
        remainder='passthrough', #Keep the other columns as they are
        force_int_remainder_cols = False
    )   
    encoded_df = column_transformer.fit_transform(encoded_file)
    num_encoded_feature_names = column_transformer.named_transformers_['num'].get_feature_names_out(
        numerical_heads)
    all_feature_names = list(num_encoded_feature_names) + column_transformer.transformers_[1][2]
    encoded_file = pd.DataFrame(encoded_df, columns=all_feature_names)
    
    print('writing csv file...')
    pd.DataFrame.to_csv(encoded_file,path_or_buf="./dataset_encoded.csv",index=True)