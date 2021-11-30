from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import numpy as np

def RMSE(actual_y, predict_y):
    rms = mean_squared_error(actual_y, predict_y, squared=False)
    return rms

def plotTrainTest(actual_y, predict_y, train_percentage, model_name):
    """
    plots predicted and actual case numbers over entire timeframe (train and test)
    """

    split_idx = split_idx = round(len(actual_y)*train_percentage)

    train_days = np.linspace(1, split_idx, num = split_idx)
    test_days = np.linspace(split_idx, len(actual_y), num = len(actual_y)-split_idx)
    

    pyplot.plot(train_days, actual_y[:split_idx], label='Actual (Train)')
    pyplot.plot(test_days, actual_y[split_idx:], label = 'Actual (Test)')
    pyplot.plot(test_days, predict_y, label='Predicted (Test)')
    pyplot.xlabel('Day Number')
    pyplot.ylabel('Daily Case Count')
    pyplot.title(model_name + ' Case Prediction')
    pyplot.legend()
    pyplot.show()