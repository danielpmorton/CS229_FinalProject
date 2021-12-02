import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def linear(train_X, train_Y, train_percentage):
    split_idx = round(len(train_Y)*train_percentage)
    train_X = train_X[:split_idx]
    test_X = train_X[split_idx:]
    train_Y = train_Y[:split_idx]
    test_Y = train_Y[split_idx:]

    regr = LinearRegression()
    regr.fit(train_X, train_Y)
    y_pred = regr.predict(test_X)
    return y_pred