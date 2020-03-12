# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 09:58:59 2020

@author: Admin
"""

### 1. EDA
## Loading libraries
import pandas as pd                   # package for data frame analysis
import numpy as np                    # package for mathematical calculation
import matplotlib.pyplot as plt     # package for graphical plot
import statsmodels.api as sm
#from datetime import datetime    # To access datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
#from statsmodels.tsa.stattools import adfuller, acf, pacf
from matplotlib.pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning, HessianInversionWarning
rcParams['figure.figsize'] = 15, 5

## Understanding the large csv file
trainpath = r'C:\Users\Admin\datasets\nyc_taxi_fare_train.csv'
# How many rows are there in this csv file
with open(trainpath) as file:
    n_rows = len(file.readlines())
print (f'Exact number of rows: {n_rows}') # 55423857 rows
# Look at the first five rows
pd.set_option('display.max_columns', 500) # Expand the output cols display
df_tmp = pd.read_csv(trainpath, nrows=5)
df_tmp.info()
df_tmp.head()

del df_tmp, n_rows, trainpath

df_tmp.head()
df_tmp.columns # We don't need these cols: key, pickup_longitude, pickup_latitude,
#dropoff_longitude, dropoff_latitude, passenger_count

## Loading dataset
# Set columns to most suitable type to optimize for memory usage
trainpath = r'C:\Users\Admin\datasets\nyc_taxi_fare_train.csv'
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str'}
cols = list(traintypes.keys())
train = pd.read_csv(trainpath, usecols=cols, dtype=traintypes)
train.info()
train['fare_amount'].describe()
train.index

## Data cleaning
# Changing pickup_datetime to right datetime format
train['pickup_datetime'] = train['pickup_datetime'].str.replace(" UTC","")
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
# Set index using pickup_datetime col
train.set_index('pickup_datetime', inplace=True, drop=False)
train.index

# First of all, get an overview, how fare_amount it was?
train['fare_amount'].plot(figsize=(15,8), title= 'Daily Taxi Fare', fontsize=14, label='fare_amount')
plt.xlabel("pickup_datetime")
plt.ylabel("fare_amount")
plt.legend(loc='best')
plt.show()

# Outlier values
# Hist
train['fare_amount'].hist(figsize=(15,8), bins=1000)
# Scatter plot
train_temp = train.drop(columns=['pickup_datetime'])
train_temp.plot(figsize=(15,8), style='k.')
plt.show()
# Create group of Taxi Fare
train_temp['fare_group'] = pd.cut(train_temp['fare_amount'],
             [-301, 0, 100, 200, 300, 400, 500, 93970], right=False)
train_count = train_temp['fare_group'].value_counts(sort=False)
train_pct = train_temp['fare_group'].value_counts(sort=False, normalize=True)
train_crosstab = pd.concat([train_count, train_pct], axis=1)
train_crosstab.columns = ['Counts', 'Percentage']
train_crosstab
del train_temp, train_count, train_pct

# Missing values
train.isnull().sum()

# Drop off data with fare_amount below 0, and above 100
train = train[(train['fare_amount'] > 0) & (train['fare_amount'] < 100)]
train['fare_amount'].describe()

## Data Transformation
# Feature extraction: Extract Year, Month, Day, Hour; Day of week, weekend
#from the pickup_datetime
train['year'] = train['pickup_datetime'].dt.year
train['month'] = train['pickup_datetime'].dt.month 
train['day'] = train['pickup_datetime'].dt.day
train['hour'] = train['pickup_datetime'].dt.hour
train['day_of_week'] = train['pickup_datetime'].dt.dayofweek 
# Make a weekend var
def is_weekend(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0 
train['weekend'] = train['pickup_datetime'].apply(is_weekend) 
train.head()
train.info()

# Let's deep dive into new features
# fare_amount will increase as the years pass by
train.groupby('year')['fare_amount'].mean().plot.bar(title= 'The Mean of Yearly Taxi Fare')
train.groupby('year')['fare_amount'].sum().plot.bar(title= 'The Sum of Yearly Taxi Fare')

train.groupby('month')['fare_amount'].mean().plot.bar(title='The Mean of Monthly Taxi Fare')

# Let’s look at the monthly mean of each year separately
temp = train.groupby(['year', 'month'])['fare_amount'].mean() 
temp.plot(figsize=(15,5), title= 'fare_amount (Monthwise)', fontsize=14)

del temp

train.groupby('day')['fare_amount'].mean().plot.bar(title='The Mean of Daily Taxi Fare')
train.groupby('hour')['fare_amount'].mean().plot.bar(title='The Mean of Hourly Taxi Fare') # Gia cao
train.groupby('hour')['fare_amount'].sum().plot.bar(title='The Sum of Hourly Taxi Fare') # Khung gio cao diem
train.groupby('weekend')['fare_amount'].mean().plot.bar(title='The Mean of Weekend Taxi Fare')
train.groupby('day_of_week')['fare_amount'].sum().plot.bar(title='The Sum of Day of week Taxi Fare')

# Let's look at the hourly, daily, weekly, monthly, yearly time series
# Aggregate the hourly time series to daily, weekly,
#and monthly time series to reduce the noise and make it more stable 
train = train.sort_index() # sort index
temp = train['fare_amount']
hourly = temp.resample('H').sum()
daily = temp.resample('D').sum()
weekly = temp.resample('W').sum()
monthly = temp.resample('M').sum()
yearly = temp.resample('A').sum()

fig, axs = plt.subplots(5,1) 
hourly.plot(figsize=(15,12), title= 'Hourly', fontsize=14, ax=axs[0])
daily.plot(figsize=(15,12), title= 'Daily', fontsize=14, ax=axs[1])
weekly.plot(figsize=(15,12), title= 'Weekly', fontsize=14, ax=axs[2])
monthly.plot(figsize=(15,12), title= 'Monthly', fontsize=14, ax=axs[3]) 
yearly.plot(figsize=(15,12), title= 'Yearly', fontsize=14, ax=axs[4])
plt.show()

del hourly, daily, weekly, monthly, yearly

# We will aggregating it on daily, then work on the daily time series
train = train.resample('D').sum().fillna(method='ffill')[['fare_amount']] # keep fare_amount only
train.head()

### 3. Model training
## Splitting data
# To divide the data into training and validation set, we will take last 6 months as the validation data and rest for training data. 
train_part = train.ix['2009-01-01':'2014-12-31']
valid_part = train.ix['2015-01-01':'2015-06-30']
# will look at how the train and validation part has been divided
train_part['fare_amount'].plot(figsize=(15,8), title= 'Daily fare_amount', fontsize=14, label='train_part')
valid_part['fare_amount'].plot(figsize=(15,8), title= 'Daily fare_amount', fontsize=14, label='valid_part')
plt.xlabel('Datetime')
plt.ylabel('fare_amount')
plt.legend(loc='best')
plt.show()

## Model selection
# RMSE function to check the accuracy of our model on validation data set
def RMSE(y, yhat):
    return np.sqrt(sum((yhat-y)**2)/len(y))

# Naive approach
y_hat = valid_part.copy()
naive_value = train_part['fare_amount'][-1]
y_hat['naive'] = naive_value
RMSE_naive = RMSE(valid_part['fare_amount'], y_hat['naive'])

plt.figure(figsize=(12,8)) 
plt.plot(train_part.index, train_part['fare_amount'], label='train_part') 
plt.plot(valid_part.index, valid_part['fare_amount'], label='valid_part') 
plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast') 
plt.legend(loc='best') 
plt.title('Naive Forecast - RMSE: {:.2f}'.format(RMSE_naive))
plt.show()

# Moving Average
y_hat = valid_part.copy()
y_hat['moving_avg_forecast'] = train_part['fare_amount'].rolling(10).mean().iloc[-1] # average of last 10 observations. 
RMSE_moving_avg = RMSE(valid_part['fare_amount'], y_hat['moving_avg_forecast'])

plt.figure(figsize=(12,8)) 
plt.plot(train_part.index, train_part['fare_amount'], label='train_part') 
plt.plot(valid_part.index, valid_part['fare_amount'], label='valid_part') 
plt.plot(y_hat.index, y_hat['moving_avg_forecast'], label='Moving Average Forecast') 
plt.legend(loc='best') 
plt.title('Moving Average Forecast (10) - RMSE: {:.2f}'.format(RMSE_moving_avg))
plt.show()


# Select a window size of MA is that give the lowest RMSE
y_hat = valid_part.copy()
rolling_para = []
RMSE = []
for i in range(10,60,1):
    y_hat['moving_avg_forecast'] = train_part['fare_amount'].rolling(i).mean().iloc[-1]
    rmse = sqrt(mean_squared_error(valid_part['fare_amount'], y_hat['moving_avg_forecast']))
    rolling_para.append(i)
    RMSE.append(rmse)
RMSE_table = pd.DataFrame(list(zip(rolling_para, RMSE)), columns=['rolling_para', 'RMSE'])
RMSE_table[RMSE_table['RMSE'] == RMSE_table['RMSE'].min()]

y_hat['moving_avg_forecast'] = train_part['fare_amount'].rolling(32).mean().iloc[-1] # average of last 32 observations. 
RMSE_moving_avg = RMSE(valid_part['fare_amount'], y_hat['moving_avg_forecast'])

plt.figure(figsize=(12,8)) 
plt.plot(train_part.index, train_part['fare_amount'], label='train_part') 
plt.plot(valid_part.index, valid_part['fare_amount'], label='valid_part') 
plt.plot(y_hat.index, y_hat['moving_avg_forecast'], label='Moving Average(32) Forecast') 
plt.legend(loc='best') 
plt.title('Moving Average Forecast (32) - RMSE: {:.2f}'.format(RMSE_moving_avg))
plt.show()

# Single Exponential Smoothing (also known as single exponential smoothing)
y_hat = valid_part.copy()
single_exp = SimpleExpSmoothing(np.asarray(train_part['fare_amount'])).fit(smoothing_level=0.9,
                         optimized=False)
y_hat['single_exp'] = single_exp.forecast(len(valid_part))
RMSE_single_exp = RMSE(valid_part['fare_amount'], y_hat['single_exp'])

plt.figure(figsize=(12,8))
plt.plot(train_part.index, train_part['fare_amount'], label='train_part') 
plt.plot(valid_part.index, valid_part['fare_amount'], label='valid_part') 
plt.plot(y_hat.index, y_hat['single_exp'], label='Single Exponential Forecast') 
plt.legend(loc='best') 
plt.title('Single Exponential Forecast (alpha = 0.9) - RMSE: {:.2f}'.format(RMSE_single_exp))
plt.show()

# We took smoothing_level from 0.5 - 0.9 and predicted based on each of these model, then looking for lowest RSME value
Smoothing_level = []
RMSE = []
for i in [0.5, 0.6, 0.7, 0.8, 0.9]:
    fit = SimpleExpSmoothing(np.asarray(train_part['fare_amount'])).fit(smoothing_level=i,
                         optimized=False)
    y_hat['single_exp'] = fit.forecast(len(valid_part))
    rmse = sqrt(mean_squared_error(valid_part['fare_amount'], y_hat['single_exp']))
    Smoothing_level.append(i)
    RMSE.append(rmse)
RMSE_table = pd.DataFrame(list(zip(Smoothing_level, RMSE)), columns=['Smoothing_level', 'RMSE'])
RMSE_table[RMSE_table['RMSE'] == RMSE_table['RMSE'].min()]

# Double Exponential Smoothing (also known as Holt’s Linear Trend Model)
y_hat = valid_part.copy()
double_exp = Holt(np.asarray(train_part['fare_amount'])).fit(optimized=True)
y_hat['double_exp'] = double_exp.forecast(len(valid_part))
RMSE_double_exp = RMSE(valid_part['fare_amount'], y_hat['double_exp'])

plt.figure(figsize=(12,8))
plt.plot(train_part.index, train_part['fare_amount'], label='train_part') 
plt.plot(valid_part.index, valid_part['fare_amount'], label='valid_part') 
plt.plot(y_hat.index, y_hat['double_exp'], label='Double Exponential Forecast') 
plt.legend(loc='best') 
plt.title('Double Exponential Forecast - RMSE: {:.2f}'.format(RMSE_double_exp))
plt.show()

# Triple Exponential Smoothing (also known as Holt winter’s model)
y_hat = valid_part.copy()
triple_exp = ExponentialSmoothing(np.asarray(train_part['fare_amount']),
                              trend="additive",
                              seasonal="additive",
                              seasonal_periods=7).fit(optimized=True)
y_hat['triple_exp'] = triple_exp.forecast(len(valid_part))
RMSE_triple_exp = RMSE(valid_part['fare_amount'], y_hat['triple_exp'])

plt.figure(figsize=(12,8))
plt.plot(train_part.index, train_part['fare_amount'], label='train_part') 
plt.plot(valid_part.index, valid_part['fare_amount'], label='valid_part') 
plt.plot(y_hat.index, y_hat['triple_exp'], label='Triple Exponential Forecast') 
plt.legend(loc='best') 
plt.title('Triple Exponential Forecast - RMSE: {:.2f}'.format(RMSE_triple_exp))
plt.show()

# ARIMA Model
# Check for Stationarity
# define Dickey-Fuller Test (DFT) function
import statsmodels.tsa.stattools as ts
def dftest(timeseries, t=30):
    dftest = ts.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value','Lags Used','Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=t).mean()
    rolstd = timeseries.rolling(window=t).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.grid()
    plt.show(block=False)

dftest(train_part['fare_amount'], 30)

def dftest_2(timeseries, t=30):
    dftest = ts.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value','Lags Used','Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


dftest_2(train_part['fare_amount'], 24) # no difference
dftest_2(train_part['fare_amount'], 30)
dftest_2(train_part['fare_amount'], 15)
dftest_2(train_part['fare_amount'], 7)

### Stationarize the time series
# Log transformation
train_part_log = np.log(train_part['fare_amount'])
dftest(train_part_log, 24)
# Square Root transformation
train_part_sqrt = np.sqrt(train_part['fare_amount'])
dftest(train_part_sqrt, 24) 

# Differencing on the Log Transformation of the Time Series
train_part_log_diff = train_part_log - train_part_log.shift(1)
train_part_log_diff.dropna(inplace=True)
dftest(train_part_log_diff)

## we have to find the optimized values for the p,d,q parameters
# ACF, PACF plot
ax1 = plt.subplot(211)
fig = sm.graphics.tsa.plot_acf(train_part_log_diff.squeeze(), lags=40, ax=ax1)
ax2 = plt.subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_part_log_diff, lags=40, ax=ax2)

# Finding the optimal values for the ARIMA(p,d,q) model
para_set = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 0), (0, 1, 2)]
data_value = train_part_log
for i in para_set:
    model = ARIMA(data_value, order=i)
    results_ARIMA = model.fit(disp=-1)
    # Measuring the quality of the model using AIC
    AIC = results_ARIMA.aic
    print('ARIMA {} - AIC: {}'.format(i, AIC))

## Scaling Back the Forecast
    
# Evaluating the Forecasted Series vs. the Original One on Valid part
valid_part_log = np.log(valid_part['fare_amount'])
model = ARIMA(valid_part_log, order=(1, 1, 1))
results_ARIMA_111 = model.fit(disp=-1)
predict_valid_ARIMA_diff = pd.Series(results_ARIMA_111.fittedvalues, copy=True)
predict_valid_ARIMA_diff_cumsum = predict_valid_ARIMA_diff.cumsum()
# assumed the first value of the log transformed time series to be the base value
predict_valid_ARIMA_log = pd.Series(valid_part_log.iloc[0], index=valid_part_log.index)
predict_valid_ARIMA_log = predict_valid_ARIMA_log.add(predict_valid_ARIMA_diff_cumsum,
                                                  fill_value=0)
predict_valid_ARIMA = np.exp(predict_valid_ARIMA_log)
RMSE_ARIMA = RMSE(valid_part['fare_amount'], predict_valid_ARIMA)

plt.figure(figsize=(12,8))
plt.plot(train_part.index, train_part['fare_amount'], label='train_part') 
plt.plot(valid_part.index, valid_part['fare_amount'], label='valid_part') 
plt.plot(predict_valid_ARIMA.index, predict_valid_ARIMA, label='ARIMA(1, 1, 1)') 
plt.legend(loc='best') 
plt.title('ARIMA(1, 1, 1) - RMSE: {:.2f}'.format(RMSE_ARIMA))
plt.show()

















