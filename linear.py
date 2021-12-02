import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def linear(X, Y, train_percentage):
    split_idx = round(len(Y)*train_percentage)
    print(X.shape)
    train_X = X[:split_idx]
    test_X = X[split_idx:]
    train_Y = Y[:split_idx]
    test_Y = Y[split_idx:]

    regr = LinearRegression()
    regr.fit(train_X, train_Y)
    print(train_X.shape)
    print(test_X.shape)
    y_pred = regr.predict(test_X)
    return y_pred