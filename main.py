from math import sqrt
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

import preprocessing
import lstm
import postprocessing 
import RF

# Fetch data 

# input parameters for getData
startDateX = '2021-01-01'
endDateX = '2021-03-31'
startDateY = '2021-01-17'
endDateY = '2021-04-16'
geo = 'US-CA'
state = 'California'

# X is the google trends data and Y is the covid case number labels
X, Y = preprocessing.getData(startDateX, endDateX, startDateY, endDateY, geo, state)

# define train/test split over timeframe
train_percentage = 0.6666

# Run linear regression model
# (Reminder: Insert here!)

# Run LSTM model 
lstm_predict = lstm.lstm(X, Y, train_percentage)
split_idx = round(len(Y)*train_percentage)
lstm_rms = postprocessing.RMSE(Y[split_idx:], lstm_predict)
print(lstm_rms)

# Random Forest
# Train-Test Split for RF
RF_TTS_method = 1
if RF_TTS_method == 0:
    # Use the scikit learn train test split (random sampling)
    TTS_kwargs = { 'test_size': 1-train_percentage,
                'random_state': 0 } # Seed for random number generator
    train_features, test_features, train_labels, test_labels = preprocessing.RF_TTS(X, Y, TTS_kwargs)
else:
    # Use our manual method of TTS at split_idx
    train_features = X[:split_idx]
    test_features = X[split_idx:]
    train_labels = Y[:split_idx]
    test_labels = Y[split_idx:]
# Run RF model
RF_kwargs = {   'bootstrap': True,
                'criterion': 'squared_error',
                'max_depth': None, 
                'min_samples_leaf': 1,
                'n_estimators': 1000,
                'min_samples_split': 2, 
                'random_state': 0 } # Seed for random number generator
RFmodel, RF_predict = RF.randomForest(train_features, test_features, train_labels, RF_kwargs)

# Postprocessing for LSTM
postprocessing.plotTrainTest(Y, lstm_predict, train_percentage, 'LSTM')

# Postprocessing for RF
# Note: can probably use the same postprocessing function as with LSTM for this
feature_list = list(X.columns)
features = np.array(X)
postprocessing.plotRF(features, feature_list, Y, RF_predict)
postprocessing.getRFImportances(RFmodel, feature_list)
postprocessing.showAccuracyInfo(RF_predict, test_labels)
postprocessing.plotRFTrees(RFmodel, feature_list, train_features, train_labels) # not working!!
