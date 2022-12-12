import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans

# acquire
from env import get_db_url
from pydataset import data
import seaborn as sns

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")



#prep data for modeling
def model_prep(train, validate, test):
    '''Prepare train, validate, and test data for modeling'''
    
    # drop unused columns
    keep_cols = ['Snowfall',
                 'YR',
                 'MO',
                 'MeanTemp'
                ]
    
    train = train[keep_cols]
    validate = validate[keep_cols]
    test = test[keep_cols]
    
    # Split data into predicting variables (X) and target variable (y) and reset the index for each dataframe
    train_X = train.drop(columns='MeanTemp').reset_index(drop=True)
    train_y = train[['MeanTemp']].reset_index(drop=True)
    
    validate_X = validate.drop(columns='MeanTemp').reset_index(drop=True)
    validate_y = validate[['MeanTemp']].reset_index(drop=True)
    
    test_X = test.drop(columns='MeanTemp').reset_index(drop=True)
    test_y = test[['MeanTemp']].reset_index(drop=True)
    
    return train_X, validate_X, test_X, train_y, validate_y, test_y