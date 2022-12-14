import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import os
import numpy as np
import env

from env import user, password, host



# remove columns that have too many missing values
def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    '''This function drop rows or columns based on the percent of values that are missing,
    dropping columns before rows'''
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    return df


# split the dataframe into train, validate, and test
def train_validate_test_split(df, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 15% of the original dataset, validate is .1765*.85= 15% of the 
    original dataset, and train is 70% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.15, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.1765, 
                                       random_state=seed)
    return train, validate, test


# This function removes columns that have information that is unnecessary 
def prepare_data(df):
    '''This function imputes missing values when applicable, and drops columns with too much missing data. Finally,
    it utilizes another function to handle the missing values based on the proportion of rows and column values missing'''
    
    #handle missing values
    df = handle_missing_values(df, prop_required_columns=.5, prop_required_rows=.75)
    
    #Drop columns with too many null values/extraneous information
    df = df.drop(columns=['PRCP', 'MAX', 'MIN', 'MEA', 'SNF']) 
    
    #Replace unnessecary data with 0
    df = df.replace({'T':0, '#VALUE!':0})
    
    #Drop remainder of rows with null values
    df = df.dropna()
    
    #Rename columns to something readable
    rename_dict = {
    'parcelid':'parcel_id'
                }
    #Convert columns from object to float
    df = df.astype({'Precip':'float', 'Snowfall':'float'})
    
    df = df.rename(columns=rename_dict)
    
    return df