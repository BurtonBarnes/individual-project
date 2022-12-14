import pandas as pd
import env
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')
import wrangle as w
from scipy import stats

###--- Vizulaizations ---###

#Vizulization 1

def distribution_meantemp(train):
    '''This function makes a chart of the meantemp'''
    sns.histplot(x='MeanTemp',bins= 30, data=train, kde=True)
    plt.title('The distribution of the mean temperature')
    plt.xlabel("Mean Temperature")
    plt.show()
    
# Viz 2

def day_meantemp(train):
    '''This function makes a chart of the day vs meantemp'''
    sns.scatterplot(x='DA', y='MeanTemp', data=train)
    plt.title('Predicting Mean Temperature based on Day')
    plt.xlabel("Day")
    plt.ylabel("Mean Temperature")
    plt.show()

# Viz 3

def month_meantemp(train):
    '''This function makes a chart of the month vs meantemp'''
    sns.scatterplot(x='MO', y='MeanTemp', data=train)
    plt.title('Predicting Mean Temperature based on Month')
    plt.xlabel("Month")
    plt.ylabel("Mean Temperature")
    plt.show()
    
# Viz 3 statistical test
    
def get_month(train):
    observed = pd.crosstab(train.MO, train.MeanTemp)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

# Viz 4

def year_meantemp(train):
    '''This function makes a chart of the year vs meantemp'''
    sns.scatterplot(x='YR', y='MeanTemp', data=train)
    plt.title('Predicting Mean Temperature based on Year')
    plt.xlabel("Year")
    plt.ylabel("Mean Temperature")
    plt.show()

# Viz 4 statistical test
    
def get_year(train):
    observed = pd.crosstab(train.YR, train.MeanTemp)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
# Viz 5

def precipitation_meantemp(train):
    '''This function makes a chart of the precipitation vs meantemp'''
    sns.scatterplot(x='Precip', y='MeanTemp', data=train, hue = 'STA')
    plt.title('Predicting Mean Temperature with Precipitation')
    plt.xlabel("Precipitation")
    plt.ylabel("Mean Temperature")
    plt.show()

# Viz 5 statistical test
    
def get_precipitation(train):
    observed = pd.crosstab(train.Precip, train.MeanTemp)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
# Viz 6

def snowfall_meantemp(train):
    '''This function makes a chart of the meantemp vs snowfall'''
    sns.scatterplot(x='Snowfall', y='MeanTemp', data=train, hue = 'STA')
    plt.title('Predicting Mean Temperature with Snowfall')
    plt.xlabel("Snowfall")
    plt.ylabel("Mean Temperature")
    plt.show()

# Viz 6 statistical test
    
def get_snowfall(train):
    observed = pd.crosstab(train.Snowfall, train.MeanTemp)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')