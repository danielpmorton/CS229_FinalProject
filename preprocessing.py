import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from pytrends.request import TrendReq
import time
import pandas as pd
import matplotlib
import google_trends_daily.gtrend as gtrend

from sklearn.model_selection import train_test_split
import platform

# Define search queries of interest 
queries = ['covid', 'coronavirus', 'covid-19', 'covid cases', 'coronavirus cases', 'covid symptoms', 
'coronavirus symptoms', 'cough', 'virus', 'vaccine', 'covid vaccine']
num_queries = len(queries)


def getTrendsData(startDate, endDate, geo, queries):
    # INPUTS ------------------------
    # startDate  | example: '2020-01-22' -- of form YEAR-MO-DY
    # endDate    | example: '2020-04-22'
    # geo        | example: 'US-CA'
    # -------------------------------

    # Parameters for the gtrend function
    pytrend = TrendReq(hl='en-US')
    cat=0
    gprop=''

    # Getting size
    num_queries = len(queries)

    # Parsing the dates as an input to the function
    # Also, using these dates to parse the difference in days and initialize arrays automatically
    d1 = datetime.strptime(startDate, '%Y-%m-%d')
    d2 = datetime.strptime(endDate, '%Y-%m-%d')
    num_days = (d2-d1).days +1 # Adding 1 to match Megan's values (inclusivity of the end date?)

    # Initializing a matrix to store the training data
    trendsData = np.zeros((num_days, num_queries))
    
    i = 0
    for keyword in queries: 
        df = gtrend.get_daily_trend(pytrend, keyword, startDate, endDate, geo=geo, cat=cat, gprop=gprop, verbose=True, tz=0)
        trendsData[:,i] = np.array(df[keyword])
        i += 1

    return trendsData

def getAllJHUdata(state, JHU_startDate, JHU_endDate):

    df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv', header=0)

    # Extracting just the data for a single state
    df_state = df[df['Province_State']== state]

    # Extract just the colums with the case data by day, then sum over the colums to give the state totals rather than per county
    df_state_cases_cumulative = df_state.loc[:,JHU_startDate: JHU_endDate].sum(axis=0)

    # NOTE that these are CUMULATIVE totals

    # If we want non-cumulative data, we can calculate the change by day. 
    # Note that the difference value on day 1/22/20 (the first day) will be NaN
    # This is not much of a concern because this is evaluated before there are any cases

    df_state_cases_day = df_state_cases_cumulative.diff()

    # We can still set the first value to be 0 manually though
    df_state_cases_day[JHU_startDate] = 0

    return df_state_cases_day

def getData(startDateX, endDateX, startDateY, endDateY, geo, state):

    """
    Retrieves JHU covid case data and search query results from Google trends API

    startDateX: start date for Google trends query data in the format: '2021-01-01'
    endDateX: end date for Google trends query data in the format: '2021-03-31'
    startDateY: start date for covid case data in the format: '2021-01-01'
    endDateY: end date for coivd case data in the format: '2021-03-31'
    geo: location parameter for Google trends API in the format: 'US-CA'
    state: state location for covid case data in the format: 'California'

    returns:
    trends_X_df: google trends frequency data as pandas table 
    cases_Y: case data as pandas table
    """

    # Reformat date into correct format for JHU covid case data API
    d1 = datetime.strptime(startDateY, '%Y-%m-%d')
    d2 = datetime.strptime(endDateY, '%Y-%m-%d')
    num_days = (d2-d1).days +1 # Adding 1 to match Megan's values (inclusivity of the end date?)
    dayIDs = np.arange(num_days).reshape(-1,1)

    # WINDOWS BELOW
    if platform.system() == 'Windows':
        startDate_JHU_format = d1.strftime('%#m/%#d/%y') # Converting from date format for trends to JHU df labels
        endDate_JHU_format = d2.strftime('%#m/%#d/%y')
    else:
        startDate_JHU_format = d1.strftime('%-m/%-d/%y') # Converting from date format for trends to JHU df labels
        endDate_JHU_format = d2.strftime('%-m/%-d/%y') # Converting from date format for trends to JHU df labels
    # NOTE: if the above gives an error, the formatting for the strftime with no zero padding is different depending on windows vs linux
    # The # symbol in the month and say fields removes the zero padding on windows


    # Querying Google trends API
    trends_X = getTrendsData(startDateX, endDateX, geo, queries)
    # Put into a dataframe with labeled columns with the search queries
    trends_X_df = pd.DataFrame(trends_X, columns=queries)

    # Querying JHU API
    startOfCovid = '1/22/20' # The first day where data has been recorded
    endOfCovid = '10/30/21' # The last day of relevance to this project
    dailyCases = getAllJHUdata(state, startOfCovid, endOfCovid) # THIS IS A PANDAS SERIES NOT DF

    # Get just the case numbers corresponding to the range of days we're observing
    # These are the Y values for the model
    cases_Y = dailyCases[startDate_JHU_format:endDate_JHU_format]
    
    return trends_X_df, cases_Y

def RF_TTS(features, labels, TTS_kwargs):
    # Train-Test-Split for Random Forest
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, **TTS_kwargs)
    return train_features, test_features, train_labels, test_labels