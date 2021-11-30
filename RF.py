
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

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import time

# Imports specifically for RF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error
import pydot

# Tree plotting
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'


def randomForest(features, labels):
    '''
    ** Main Random Forest Function **

    Inputs: Features x in pandas dataframe format
                - Each column is a different search trend
                - Each row is a different day (but the index is just a number not a date? check this)
                - Entries are the relative importance of the search term on that day
            Labels y in pandas series format

    Output: Returns the predicted values for the timeframe dictated by the inputs

    '''
    # ********** PARAMETERS **********
    # Train-Test-Split 
    TTS_kwargs = {
        'test_size': 0.25,
        'random_state': 0 # Seed for random number generator
    }
    # Random Forest
    RF_kwargs = {
        'bootstrap': True,
        'criterion': 'squared_error',
        'max_depth': None, 
        'min_samples_leaf': 1,
        'n_estimators': 1000,
        'min_samples_split': 2, 
        'random_state': 0 # Seed for random number generator
        }

    # What additional functions do we want to run?
    bool_getImportances = True
    bool_plotActualVsPredicted = True
    bool_displayAccuracyInfo = True
    bool_plotImportances = True
    bool_plotTrees = False # Can't get this one to work quite right. Some installation issues with graphviz

    # ********** End parameters **********

    # Get a list of all of the search terms we are considering
    feature_list = list(features.columns)
    # Convert our table of search importances to a numpy array
    features = np.array(features)
    # Use Skicit-learn to split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, **TTS_kwargs)
    # Instantiate model 
    rf = RandomForestRegressor(**RF_kwargs)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    # Make predictions based on the test data
    predictions = rf.predict(test_features)

    # Optional components of the RF model
    if bool_plotTrees:
        plotTrees(rf, feature_list, train_features, train_labels)
    if bool_getImportances:
        getImportances(rf, feature_list, bool_plotImportances)
    if bool_plotActualVsPredicted:
        plotActualVsPredicted(features, feature_list, labels, predictions)
    if bool_displayAccuracyInfo:
        displayAccuracyInfo(predictions, test_labels)

    return predictions

# Other functions:

def plotTrees(rf, feature_list, train_features, train_labels):
    # Visualization of decision tree
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')
    print('The depth of this tree is:', tree.tree_.max_depth)
    # Limit depth of tree to 2 levels
    rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)
    rf_small.fit(train_features, train_labels)
    # Extract the small tree
    tree_small = rf_small.estimators_[5]
    # Save the tree as a png image
    export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
    graph.write_png('small_tree.png')


def getImportances(rf, feature_list, bool_plotImportances):
    # Importances
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    if bool_plotImportances:
        # Set the style
        plt.style.use('fivethirtyeight')
        # list of x locations for plotting
        x_values = list(range(len(importances)))
        # Make a bar chart
        plt.bar(x_values, importances, orientation = 'vertical')
        # Tick labels for x axis
        plt.xticks(x_values, feature_list, rotation='vertical')
        # Axis labels and title
        plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
        plt.show()


def plotActualVsPredicted(features, feature_list, labels, predictions):
    
    # Creating a list of integers in place of actual dates (can fix this afterwards)
    dayIDs_labels = range(0,len(labels))
    dayIDs_predictions = range(0,len(predictions))

    # Dataframe with true values and dates
    true_data = pd.DataFrame(data = {'date': dayIDs_labels, 'actual': labels})
    # Dataframe with predictions and dates
    predictions_data = pd.DataFrame(data = {'date': dayIDs_predictions, 'prediction': predictions}) 
    # Plot the actual values
    plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
    # Plot the predicted values
    plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
    plt.legend()
    # Graph labels
    plt.xlabel('Date')
    plt.ylabel('Covid Cases')
    plt.title('Actual and Predicted Values')
    plt.show()

    # Make the data accessible for plotting
    true_data['covid symptoms'] = features[:, feature_list.index('covid symptoms')]
    true_data['coronavirus'] = features[:, feature_list.index('coronavirus')]
    true_data['covid'] = features[:, feature_list.index('covid')]
    # Plot all the data as lines
    plt.plot(true_data['date'], true_data['actual'], 'b-', label  = 'actual', alpha = 1.0)
    plt.plot(true_data['date'], true_data['covid symptoms'], 'y-', label  = 'covid symptoms', alpha = 1.0)
    plt.plot(true_data['date'], true_data['coronavirus'], 'k-', label = 'coronavirus', alpha = 0.8)
    plt.plot(true_data['date'], true_data['covid'], 'r-', label = 'covid', alpha = 0.3)
    # Formatting plot
    plt.legend()
    # Lables and title
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.title('Actual Cases and Search Queries')
    plt.show()


def displayAccuracyInfo(predictions, test_labels):
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'Cases.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    

# ARCHIVE - NOT BEING USED ###########################
# def rerunWithFewerVariables(feature_list, train_features, train_labels, test_features, test_labels, importances):
#     # 2 Most Important Features
#     # New random forest with only the two most important variables
#     rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
#     # Extract the two most important features
#     important_indices = [feature_list.index('covid symptoms'), feature_list.index('coronavirus')]
#     train_important = train_features[:, important_indices]
#     test_important = test_features[:, important_indices]
#     # Train the random forest
#     rf_most_important.fit(train_important, train_labels)
#     # Make predictions and determine the error
#     predictions = rf_most_important.predict(test_important)
#     errors = abs(predictions - test_labels)
#     # Display the performance metrics
#     print('Mean Absolute Error:', round(np.mean(errors), 2), 'Cases.')
#     mape = np.mean(100 * (errors / test_labels))
#     accuracy = 100 - mape
#     print('Accuracy:', round(accuracy, 2), '%.')