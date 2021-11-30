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

# input parameters for getData
startDateX = '2021-01-01'
endDateX = '2021-03-31'
startDateY = '2021-01-17'
endDateY = '2021-04-16'
geo = 'US-CA'
state = 'California'


train_X, train_Y = preprocessing.getData(startDateX, endDateX, startDateY, endDateY, geo, state)

train_X = train_X.values
train_Y = train_Y.values
train_Y = train_Y.reshape((-1,1))
train_Y_unscaled = train_Y
concat = concatenate((train_X, train_Y), axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(concat)
train_X = scaled[:, :-1]
train_Y = scaled[:, -1]
train_X_scaled = train_X

test_X = train_X[60:,:]
test_Y = train_Y[60:]
train_X = train_X[:60,:]
train_Y = train_Y[:60:]
test_X_scaled = test_X

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()
model.add(LSTM(100, input_shape = (train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_Y, epochs=200, batch_size=90, validation_data=(test_X, test_Y), verbose=2, shuffle=False)

yhat = model.predict(test_X)
concat = concatenate((test_X_scaled, yhat), axis=1)
y = scaler.inverse_transform(concat)
y = y[:,-1]
print(y)
pyplot.plot(train_Y_unscaled[60:], label='actual')
pyplot.plot(y, label='predict')
pyplot.legend()
pyplot.show()