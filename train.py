'''
File: data_preprocessing.py

Author: Ali Reza (ARO) Omrani
Email: omrani.alireza95@gmail.com
Date: 26th March 2025

Description:
------
This file contains training a model.

Functions:
- init_info(seed_num): Setting the initial values.
- train(model, optimizer, criterion, dataloader, device): Train Process.
- validation(model, criterion, dataloader, device): Validation Process.
- main(seed_num, bsize, epoch_num, data_prep_mode): Main function to initialize, train, and validate the fraud detection model.

Requirements:
------
- torch
- shap
- numpy
- tqdm
'''

import torch
import torch.nn as nn
import shap
import numpy as np
from utils.utils import FraudDetectionNN, data_prep, data_loader
from tqdm import tqdm

def init_info(seed_num):
    '''
    Setting the initial values.

    Parameters:
       - seed_num (int): The Seed Number. 

    Returns:
       - device (str): The device for model computation, either "cpu" or "cuda".

    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    np.random.seed(seed_num)

    return device

def train(model, optimizer, criterion, dataloader, device, epoch_num):
    '''
    Train Process.

    Parameters:
        - model (FraudDetectionNN): The custom Neural Network model for fraud detection.
        - optimizer (torch.optim.Adam): The optimizer used to update the model parameters.
        - criterion (torch.nn.BCELoss): The loss function used to compute the error.
        - dataloader (function): A function that yields batches of data for training.
        - device (str): The device for model computation, either "cpu" or "cuda".
        - epoch_num (int): The current epoch number.

    Returns:
        train_loss (float): The average training loss for the epoch.
        train_acc (float): The average training accuracy for the epoch.
    '''

    loss_lst, correct = 0, 0
    model.train()
    with tqdm(dataloader,desc=f'Training Epoch {epoch_num+1}/{EPOCHS}',leave=False, ncols=150) as t_data_loader:
        for idx, (x,y) in enumerate(t_data_loader):
            optimizer.zero_grad()
            
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss = criterion(output,y)
            loss.backward()
            optimizer.step()

            loss_lst += loss.item()
            pred = output.round()
            correct += pred.eq(y.view_as(pred)).sum().item()
            t_data_loader.set_postfix(train_loss= f"{loss_lst/(idx+1):.4f}",train_acc= f"{(100.0 * correct)/len(dataloader.dataset):.4f}")
    train_loss = loss_lst / len(dataloader.dataset)
    train_acc = 100. * correct / len(dataloader.dataset)
    return train_loss,train_acc
    
def validation(model, criterion, dataloader, device, epoch_num):
    '''
    Validation Process.

    Parameters:
        - model (FraudDetectionNN): The custom Neural Network model for fraud detection.
        - criterion (torch.nn.BCELoss): The loss function used to compute the error.
        - dataloader (function): A function that yields batches of data for training.
        - device (str): The device for model computation, either "cpu" or "cuda".
        - epoch_num (int): The current epoch number.

    Returns:
        val_loss (float): The average Validation loss for the epoch.
        val_acc (float): The average Validation accuracy for the epoch.
    '''
    loss_lst, correct = 0, 0
    model.eval()
    with tqdm(dataloader,desc=f'Validation Epoch {epoch_num+1}/{EPOCHS}',ncols=150, leave=False) as t_data_loader:
        with torch.no_grad():
            for idx, (x,y) in enumerate(t_data_loader):
                x, y = x.to(device), y.to(device)
                output = model(x)

                loss = criterion(output,y)
                
                loss_lst += loss.item()
                pred = output.round()
                correct += pred.eq(y.view_as(pred)).sum().item()
                t_data_loader.set_postfix(val_loss= f"{loss_lst/(idx+1):.4f}",train_acc= f"{(100.0 * correct)/len(dataloader.dataset):.4f}")
    val_loss = loss_lst/ len(dataloader.dataset)
    val_acc = 100. * correct / len(dataloader.dataset)
    return val_loss, val_acc 

def main(seed_num, bsize, data_prep_mode = None):
    '''
    Main function to initialize, train, and validate the fraud detection model.

    Parameters:
        - seed_num (int): The seed number for random number generation algorithm.
        - bsize (int): The batch size.
        - data_prep_mode (str): The method for oversampling. Default value is None.

    Returns:
        None

    Notes:
    -----
        - The options for data_prep_mode are "smote" and None.
    '''
    device = init_info(seed_num)
    adresss = r'd:\Job presentations\Schwarz\fraud\dataset4_encoded_train.csv'

    #Data Preparation
    print("Data Preparation...")
    X_train, y_train, X_val, y_val = data_prep(address = adresss, mode = data_prep_mode, random_state = seed_num)
    
    train_loader, val_loader = data_loader(X_train, y_train, X_val, y_val, bsize = bsize)

    # Define the model, loss function, and optimizer for training
    print('Defining the model, loss function, and optimizer for training')
    model = FraudDetectionNN(X_train.shape[1])
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

    #Training and Validation Section
    print("Training...")
    with tqdm(range(EPOCHS), unit='epoch',desc="Progress",ncols=150) as e_tqdm:
        for epoch in e_tqdm:
            train_loss, train_acc = train(model,optimizer,criterion,train_loader,device,epoch)

            valid_loss, valid_acc = validation(model,criterion,val_loader,device,epoch)


            e_tqdm.set_postfix(train_loss= f'{train_loss:.4f}',
                               train_acc= f"{train_acc:.4f}",
                               valid_loss= f"{valid_loss:.4f}",
                               valid_acc= f"{valid_acc:.4f}")
    #Explainability Section
    print('Explainability...')
    sample_input = torch.randn(1,X_train.shape[-1]).to(device)
    explainer = shap.DeepExplainer(model,sample_input)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)

if __name__ == '__main__':
    EPOCHS = 2
    main(seed_num= 1, bsize= 32, data_prep_mode= 'smote')