
'''
Random Forest
CS229 Final Project, Fall 2021

Overview
- Load this file into a separate script
- Call randomForest(features, labels) to run the main function
- The features, labels data is assumed to be preprocessed in this separate script

Citation
- Credit for most of this code goes to Will Koehrsen
    - "Random Forest in Python", https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

TO-DO
- Need to figure out a good way to include the dates in the plots! When doing the test/train split, 
  need to recover the info about what dates correspond to which test/train data points

'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def randomForest(train_features, test_features, train_labels, RF_kwargs):
    '''
    ** Main Random Forest Function **

    Inputs: Features x in pandas dataframe format
                - Each column is a different search trend
                - Each row is a different day (but the index is just a number not a date? check this)
                - Entries are the relative importance of the search term on that day
            Labels y in pandas series format

    Output: Returns the predicted values for the timeframe dictated by the inputs

    '''
    # Instantiate model 
    rf = RandomForestRegressor(**RF_kwargs)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    # Make predictions based on the test data
    predictions = rf.predict(test_features)
    return rf, predictions