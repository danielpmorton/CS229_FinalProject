from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


import preprocessing
import lstm
import postprocessing 

# Fetch data 

# input parameters for getData
startDateX = '2021-01-01'
endDateX = '2021-03-31'
startDateY = '2021-01-17'
endDateY = '2021-04-16'
geo = 'US-TX'
state = 'Texas'

train_X, train_Y = preprocessing.getData(startDateX, endDateX, startDateY, endDateY, geo, state)

# define train/test split over timeframe
train_percentage = 0.6666

# Run linear regression model

# Run RF model

# Run LSTM model 
lstm_predict = lstm.lstm(train_X, train_Y, train_percentage)

split_idx = round(len(train_Y)*train_percentage)
lstm_rms = postprocessing.RMSE(train_Y[split_idx:], lstm_predict)
print(lstm_rms)

postprocessing.plotTrainTest(train_Y, lstm_predict, train_percentage, 'LSTM')


