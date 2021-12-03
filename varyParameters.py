# Script to test out different parameters and see what effect this has on the model

import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from pandas.io.stata import StataParser
from pytrends.request import TrendReq
import time
import pandas as pd
import matplotlib
import google_trends_daily.gtrend as gtrend

from sklearn.model_selection import train_test_split
import platform

import RF
import lstm
import linear
import postprocessing
import platform
# Making a new preprocessing file so I can mess with the inputs/outputs
import preprocessing_for_variation as prep

# Just for reference for the JHU data files
startOfCovid = '1/22/20' # The first day where data has been recorded
endOfCovid = '10/30/21' # The last day of relevance to this project
# NOTE delay can be a maximum of 30 because of this end date and the way I load all the data at once

# Step 1: Get the trends data for a very large timeframe so we can pre-load this and not web-scrape each time

xDateFormat = '%Y-%m-%d'
if platform.system() == 'Windows':
    yDateFormat = '%#m/%#d/%y'
else:
    yDateFormat = '%-m/%-d/%y'

# Using a delay of 16 days
startDateX = '2020-02-01'
endDateX = '2021-09-30'
delay = timedelta(days=16)
startDateY = (datetime.strptime(startDateX, xDateFormat) + delay).strftime(yDateFormat)
endDateY = (datetime.strptime(endDateX, xDateFormat) + delay).strftime(yDateFormat)
geo = 'US-CA'
state = 'California'


X_all, Y_all_16delay, Y_all = prep.getDataNew(startDateX, endDateX, startDateY, endDateY, geo, state)
# X_all is the search term data for the entire giant period
# Y_all_16delay is the case data for the entire period, with the delay included

# Indexing this by date as well:
sdate = datetime.strptime(startDateX, xDateFormat)
edate = datetime.strptime(endDateX, xDateFormat)
dates = pd.date_range(sdate,edate,freq='d')
dateList = dates.strftime(xDateFormat).to_list()
X_all_dateIndexed = X_all.copy(deep=True)
X_all_dateIndexed['dates'] = dateList
X_all_dateIndexed.set_index('dates')
# Y is already indexed by date, but in format %#m/%#d/%y (windows) or %-m/%-d/%y (mac)



# Varying the split percentage for train and test ############################################################

# Get a subset of data based on a smaller timeframe (let this be a 4 month period)

# X_all is indexed by day number, rather than the date itself
# So, index X_all by the actual dates rather than just integers


startDate_subset_X = '2020-02-01'
endDate_subset_X = '2020-06-01'
startDate_subset_Y = '2/1/20'
endDate_subset_Y = '6/1/20'

X = X_all_dateIndexed[startDate_subset_X:endDate_subset_X]
Y = Y_all_16delay[startDate_subset_Y:endDate_subset_Y]

rmsdata_varyTrainPct = [] # initialize

percentages = [0.5, 0.6, 0.7, 0.8, 0.9]

for train_percentage in percentages:
    split_idx = round(len(Y)*train_percentage)
    # Linear regression
    linear_predict = linear.linear(X, Y, train_percentage)
    linear_rms = postprocessing.RMSE(Y[split_idx:], linear_predict)
    # LSTM
    lstm_predict = lstm.lstm(X, Y, train_percentage)
    lstm_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)
    # RF
    RF_kwargs = {'bootstrap': True,'criterion': 'squared_error','max_depth': None,'min_samples_leaf': 1,'n_estimators': 1000,'min_samples_split': 2,'random_state': 0 }
    RFmodel, RF_predict = RF.randomForest(X[:split_idx], X[split_idx:], Y[:split_idx], RF_kwargs)
    RF_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)

    rmsdata_varyTrainPct.append([train_percentage, linear_rms, lstm_rms, RF_rms])

    # Store the rms data 

trainPct_df = pd.DataFrame(rmsdata_varyTrainPct, columns=['Train %', 'Linear Regression', 'LSTM', 'Random Forest'])
trainPct_df.set_index('Train %')
trainPct_df.style.set_caption('Varying the Train/Test Split (Train Percentage): Effects on RMSE')
print(trainPct_df)


# Change the length of the timeframe being considered ############################################################
durations = [1,2,3,4,5,6,7,8,9] # months

# Arbitrary start date
startDate_subset_X = '2020-02-01'
startDate_subset_Y = '2/1/20'

rmsdata_varyDuration = [] # initialize

train_percentage = 0.75

for duration in durations:
    dt = timedelta(months=duration)
    endDate_subset_X = (datetime.strptime(startDate_subset_X, xDateFormat) + dt).strftime(xDateFormat)
    endDate_subset_Y = (datetime.strptime(startDate_subset_Y, yDateFormat) + dt).strftime(yDateFormat)
    X = X_all_dateIndexed[startDate_subset_X:endDate_subset_X]
    Y = Y_all_16delay[startDate_subset_Y:endDate_subset_Y]
    
    split_idx = round(len(Y)*train_percentage)
    # Linear regression
    linear_predict = linear.linear(X, Y, train_percentage)
    linear_rms = postprocessing.RMSE(Y[split_idx:], linear_predict)
    # LSTM
    lstm_predict = lstm.lstm(X, Y, train_percentage)
    lstm_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)
    # RF
    RF_kwargs = {'bootstrap': True,'criterion': 'squared_error','max_depth': None,'min_samples_leaf': 1,'n_estimators': 1000,'min_samples_split': 2,'random_state': 0 }
    RFmodel, RF_predict = RF.randomForest(X[:split_idx], X[split_idx:], Y[:split_idx], RF_kwargs)
    RF_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)

    rmsdata_varyDuration.append([duration, linear_rms, lstm_rms, RF_rms])

duration_df = pd.DataFrame(rmsdata_varyDuration, columns=['Duration', 'Linear Regression', 'LSTM', 'Random Forest'])
duration_df.set_index('Duration')
duration_df.style.set_caption('Varying the model duration (months of data): Effects on RMSE')
print(duration_df)


# Change the start time ############################################################
startDatesX = ['2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01']
dt = timedelta(days=16)
rmsdata_varyStartDate = []
for sdateX in startDatesX:
    sdateY = (datetime.strptime(sdateX, xDateFormat)).strftime(yDateFormat)
    edateX = (datetime.strptime(sdateX, xDateFormat) + dt).strftime(xDateFormat)
    edateY = (datetime.strptime(sdateY, yDateFormat) + dt).strftime(yDateFormat)

    X = X_all_dateIndexed[sdateX:edateX]
    Y = Y_all_16delay[sdateY:edateY]
    
    split_idx = round(len(Y)*train_percentage)
    # Linear regression
    linear_predict = linear.linear(X, Y, train_percentage)
    linear_rms = postprocessing.RMSE(Y[split_idx:], linear_predict)
    # LSTM
    lstm_predict = lstm.lstm(X, Y, train_percentage)
    lstm_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)
    # RF
    RF_kwargs = {'bootstrap': True,'criterion': 'squared_error','max_depth': None,'min_samples_leaf': 1,'n_estimators': 1000,'min_samples_split': 2,'random_state': 0 }
    RFmodel, RF_predict = RF.randomForest(X[:split_idx], X[split_idx:], Y[:split_idx], RF_kwargs)
    RF_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)

    rmsdata_varyStartDate.append([(sdateX+ ' : ' + edateX), linear_rms, lstm_rms, RF_rms])

startDate_df = pd.DataFrame(rmsdata_varyStartDate, columns=['Timeframe', 'Linear Regression', 'LSTM', 'Random Forest'])
startDate_df.set_index('Timeframe')
startDate_df.style.set_caption('Adjusting the timeframe period: Effects on RMSE')
print(startDate_df)



# Change the amount of time delay ############################################################

delays = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
rmsdata_varydelay = []
# Arbitrary start and end dates
sdateX = '2020-02-01'
edateX = '2020-06-01'
train_percentage = 0.75

for delay in delays:
    dt = timedelta(days=delay)

    sdateY = (datetime.strptime(sdateX, xDateFormat) + dt).strftime(yDateFormat)
    edateY = (datetime.strptime(edateX, xDateFormat) + dt).strftime(yDateFormat)

    X = X_all_dateIndexed[sdateX:edateX]
    Y = Y_all_16delay[sdateY:edateY]
    
    split_idx = round(len(Y)*train_percentage)
    # Linear regression
    linear_predict = linear.linear(X, Y, train_percentage)
    linear_rms = postprocessing.RMSE(Y[split_idx:], linear_predict)
    # LSTM
    lstm_predict = lstm.lstm(X, Y, train_percentage)
    lstm_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)
    # RF
    RF_kwargs = {'bootstrap': True,'criterion': 'squared_error','max_depth': None,'min_samples_leaf': 1,'n_estimators': 1000,'min_samples_split': 2,'random_state': 0 }
    RFmodel, RF_predict = RF.randomForest(X[:split_idx], X[split_idx:], Y[:split_idx], RF_kwargs)
    RF_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)

    rmsdata_varydelay.append([(sdateX+ ' : ' + edateX), linear_rms, lstm_rms, RF_rms])

delay_df = pd.DataFrame(rmsdata_varydelay, columns=['Delay, days', 'Linear Regression', 'LSTM', 'Random Forest'])
delay_df.set_index('Delay, days')
delay_df.style.set_caption('Adjusting the delay between searches and cases: Effects on RMSE')
print(delay_df)

# Change the state being considered ############################################################
# This will require re-parsing the data from gtrends

other_geos = ['US-MA', 'US-WA', 'US-TX', 'US-IL']
other_states = ['Massachusetts', 'Washington', 'Texas', 'Illinois']

# Setting the timeframe to a predefined period 
startDateX = '2020-02-01'
endDateX = '2021-09-30'
delay = timedelta(16)
startDateY = (datetime.strptime(startDateX, xDateFormat) + delay).strftime(yDateFormat)
endDateY = (datetime.strptime(endDateX, xDateFormat) + delay).strftime(yDateFormat)

train_percentage = 0.75

rmsdata_varyState = []

for i in range(len(other_states)):
    geo = other_geos[i]
    state = other_states[i]

    X, Y, Y_noDelay = prep.getDataNew(startDateX, endDateX, startDateY, endDateY, geo, state)

    split_idx = round(len(Y)*train_percentage)
    # Linear regression
    linear_predict = linear.linear(X, Y, train_percentage)
    linear_rms = postprocessing.RMSE(Y[split_idx:], linear_predict)
    # LSTM
    lstm_predict = lstm.lstm(X, Y, train_percentage)
    lstm_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)
    # RF
    RF_kwargs = {'bootstrap': True,'criterion': 'squared_error','max_depth': None,'min_samples_leaf': 1,'n_estimators': 1000,'min_samples_split': 2,'random_state': 0 }
    RFmodel, RF_predict = RF.randomForest(X[:split_idx], X[split_idx:], Y[:split_idx], RF_kwargs)
    RF_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)

    rmsdata_varyState.append([state, linear_rms, lstm_rms, RF_rms])

state_df = pd.DataFrame(rmsdata_varyState, columns=['State', 'Linear Regression', 'LSTM', 'Random Forest'])
state_df.set_index('State')
state.style.set_caption('Comparing the RMSE when training on different states for a given time period')
print(state_df)



