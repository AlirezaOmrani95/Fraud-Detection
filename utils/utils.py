'''
File: utils.py

Author: Ali Reza (ARO) Omrani
Email: omrani.alireza95@gmail.com
Date: 17th March 2025

Description:
------
This file contains a simple FNN and data prep functions.

Classes:
------
- FraudDetectionNN: Represents a simple FNN model for Fraud Detection

Functions:
- data_prep (address, mode = None, random_state=1): Preparing the data for the model.
- data_loader(x_train, y_train, x_test, y_test, bsize = 128): Data Loader function.

Requirements:
------
- torch
- sklearn
- imblearn
- pandas

Notes:
------
- Functions are designed to preprocess to train a NN model.

'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import pandas as pd

class FraudDetectionNN(nn.Module):
    '''
    The FraudDetectionNN represents a simple FNN model for Fraud Detection.

    Attributes: 
    ------
    - input_dim (int): The input dimension of the network.
    
    Methods:
    ------
    - forward(x): Performs a forward pass through the network.
    '''
    def __init__(self, input_dim):
        '''
        Initializes the Simple FNN with given input_dim.
        
        Parameters:
        ------
        input_dim (int): The input dimension of the network.
        '''
        super(FraudDetectionNN,self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(input_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        '''
        Performs a forward pass through the network. 

        Parameters:
        ------
        - x (ndarray): The input data for the network.

        Returns:
        ------
        - ndarray: The output of the network.
        '''
        out = self.fc_block(x)
        return out
    
def data_prep (address, mode = None, random_state=1):
    '''
    Preparing the data for the model.
    
    Parameters:
    ------
    - address (str): The csv file path.
    - mode (str): The method for oversampling. 
    - random_state (int): Value for random state. The default value is 1.

    Return:
    ------
    - x_train, y_train, x_test, y_test (all Tensors): The train and test data.

    Example:
    ------
        >>> X_train, y_train, X_val, y_val = data_prep(address = adresss, mode = 'smote', random_state = seed_num)

    Note:
    ------
        - The options for mode are "smote" and None. Default value is None.
    '''

    drop_col = ['is_fraud']
    
    file = pd.read_csv(address)
    X = file.drop(columns=drop_col,axis=1).values
    y = file['is_fraud'].values

    '''Because the data used in this project was imbalanced, we used Stratified sampling.'''
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True, random_state=random_state,stratify=y)
    x_test = torch.tensor(x_test,dtype=torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)

    if mode == 'smote':
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(x_train,y_train)

        X_train_resampled = torch.tensor(X_train_resampled,dtype=torch.float32)
        y_train_resampled = torch.tensor(y_train_resampled,dtype=torch.float32).unsqueeze(1)

        return (torch.tensor(X_train_resampled,dtype=torch.float32),
                torch.tensor(y_train_resampled,dtype=torch.float32),
                x_test,
                y_test)
        
    else:
        return (x_train, y_train, x_test, y_test)

def data_loader(x_train, y_train, x_test, y_test, bsize = 128):
    '''
    Data Loader function.
    
    Parameters:
    ------
    - x_train (Tensor): The train input.
    - y_train (Tensor): The train target. 
    - x_test (Tensor): The test input.
    - y_test (Tensor): The test target.
    - bsize (int): Batch size value. Default values is 128.

    Return:
    ------
    - train_loader, test_loader (both DataLoader): The train and test Data Loader.

    '''
    train_dataset = TensorDataset(x_train,y_train)
    test_dataset = TensorDataset(x_test,y_test)

    train_loader = DataLoader(dataset= train_dataset,batch_size = bsize, shuffle=True)
    test_loader = DataLoader(dataset= test_dataset, batch_size= bsize*2)

    return train_loader, test_loader