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



# get the baseline for modeling
def get_mean(train_y, validate_y):
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values.
    y_train = pd.DataFrame(train_y)
    y_validate = pd.DataFrame(validate_y)
    
    # Predict property_value_pred_mean
    MeanTemp_pred_mean = train_y.MeanTemp.mean()
    train_y['MeanTemp_pred_mean'] = MeanTemp_pred_mean
    validate_y['MeanTemp_pred_mean'] = MeanTemp_pred_mean
    
    # RMSE of property_value_pred_mean
    rmse_train = mean_squared_error(y_train.MeanTemp,
                                y_train.MeanTemp_pred_mean) ** .5
    rmse_validate = mean_squared_error(y_validate.MeanTemp, y_validate.MeanTemp_pred_mean) ** (1/2)
    
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    
# linear regression model
def linear_regression(train_X, train_y, validate_X, validate_y):
    lm = LinearRegression(normalize=True)
    lm.fit(train_X, train_y.MeanTemp)
    train_y['MeanTemp_pred_lm'] = lm.predict(train_X)
    
    # evaluate: rmse
    rmse_train = mean_squared_error(train_y.MeanTemp, train_y.MeanTemp_pred_lm) ** (1/2)

    # predict validate
    validate_y['MeanTemp_pred_lm'] = lm.predict(validate_X)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate_y.MeanTemp, validate_y.MeanTemp_pred_lm) ** (1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    return rmse_train, rmse_validate


# lassolars model
def lassolars(train_X, train_y, validate_X, validate_y):
    # create the model object
    lars = LassoLars(alpha=1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series!
    lars.fit(train_X, train_y.MeanTemp)

    # predict train
    train_y['MeanTemp_pred_lars'] = lars.predict(train_X)

    # evaluate: rmse
    rmse_train = mean_squared_error(train_y.MeanTemp, train_y.MeanTemp_pred_lars) ** (1/2)

    # predict validate
    validate_y['MeanTemp_pred_lars'] = lars.predict(validate_X)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate_y.MeanTemp, validate_y.MeanTemp_pred_lars) ** (1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    return rmse_train, rmse_validate


# polynomial model
def polynomial(train_X, train_y, validate_X, validate_y, test_X):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(train_X)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(validate_X)
    X_test_degree2 =  pf.transform(test_X)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, train_y.MeanTemp)

    # predict train
    train_y['MeanTemp_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(train_y.MeanTemp, train_y.MeanTemp_pred_lm2) ** (1/2)

    # predict validate
    validate_y['MeanTemp_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate_y.MeanTemp, validate_y.MeanTemp_pred_lm2) ** 0.5

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    return rmse_train, rmse_validate



# lassolars model for test
def lassolars_test(test_X, test_y):
    # create the model object
    lars = LassoLars(alpha=1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series!
    lars.fit(test_X, test_y.MeanTemp)

    # predict test
    test_y['MeanTemp_pred_lars'] = lars.predict(test_X)

    # evaluate: rmse
    rmse_test = mean_squared_error(test_y.MeanTemp, test_y.MeanTemp_pred_lars) ** (1/2)

    print("RMSE for Lasso + Lars\nTest/In-Sample: ", rmse_test)
    return rmse_test