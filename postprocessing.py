from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def RMSE(actual_y, predict_y):
    rms = mean_squared_error(actual_y, predict_y, squared=False)
    return rms

def MAPE(actual_y, predict_y):
    error = mean_absolute_percentage_error(actual_y, predict_y)
    return error

def plotTrainTest(actual_y, predict_y, train_percentage, model_name):
    """
    plots predicted and actual case numbers over entire timeframe (train and test)
    """

    split_idx = split_idx = round(len(actual_y)*train_percentage)

    train_days = np.linspace(1, split_idx, num = split_idx)
    test_days = np.linspace(split_idx, len(actual_y), num = len(actual_y)-split_idx)
    

    plt.plot(train_days, actual_y[:split_idx], label='Actual (Train)')
    plt.plot(test_days, actual_y[split_idx:], label = 'Actual (Test)')
    plt.plot(test_days, predict_y, label='Predicted (Test)')
    plt.xlabel('Day Number')
    plt.ylabel('Daily Case Count')
    plt.title(model_name + ' Case Prediction')
    plt.legend()
    plt.show()


# RF plotting stuff - moved over from the other file ######

def plotRF(features, feature_list, labels, predictions):
    
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

def plotRFTrees(rf, feature_list, train_features, train_labels):
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import export_graphviz
    import pydot
    import graphviz
    import os
    os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'

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

def getRFImportances(rf, feature_list, bool_plotImportances):
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

def showAccuracyInfo(predictions, test_labels):
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'Cases.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')