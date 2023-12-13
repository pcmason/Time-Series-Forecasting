'''
This program uses 5 methods (date time, lag, window, rolling window & expanding window) to create the features for a time series problem.
Uses the Min. Daily Temperature dataset with observations over 10 years (1981-90) from Melbourne, Australia
'''

import pandas as pd

# Load in the dataset as a series
series = pd.read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True)
series = series.squeeze()

# Use the month and day as the features (Date Time method)
# Create a dataframe to hold the extra columns
dataframe = pd.DataFrame()
# Store month & day information unpacked from the timestamp info
dataframe['month'] = [series.index[i].month for i in range(len(series))]
dataframe['day'] = [series.index[i].day for i in range(len(series))]
# Now store the actual temperature data in its own column
dataframe['temperature'] = [series[i] for i in range(len(series))]
# Output the first 5 values from the dataframe
print('Date Time Features:')
print(dataframe.head(), '\n')

# Predict the value of the current time (t) given the value at previous time (t-1)
# Get the temperature in a dataframe
temps = pd.DataFrame(series.values)
# Concatenate the shifted columns together into a new dataframe along the column axis
dataframeLag = pd.concat([temps.shift(1), temps], axis=1)
# Name columns for clarity
dataframeLag.columns = ['t-1', 't']
# Output first 5 rows of lag dataframe
print('Lag Features:')
print(dataframeLag.head(), '\n')

# Expand the window from a length of 1 to 3
dataframe3Lag = pd.concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
# Name columns for clarity
dataframe3Lag.columns = ['t-3', 't-2', 't-1', 't']
# Output first 5 rows of lag dataframe with window width of 3
print('Lag window width of 3:')
print(dataframe3Lag.head(), '\n')

# Use a rolling window with the mean of the previous 2 values
# Shift the series
shifted = temps.shift(1)
# Create the rolling dataset using the previous 2 values and just the mean
window = shifted.rolling(window=2)
means = window.mean()
# Use concat to construct a new dataframe with the new columns
dataframeRoll = pd.concat([means, temps], axis=1)
dataframeRoll.columns = ['mean(t-2,t-1)', 't']
print('Rolling Mean:')
print(dataframeRoll.head(), '\n')

# Create a rolling window using the min, max & mean summary statistics
width = 3
shifted2 = temps.shift(width-1)
# Create the rolling dataset
window = shifted2.rolling(window=width)
# Create the dataframe with concat
dfRoll = pd.concat([window.min(), window.mean(), window.max(), temps], axis=1)
# Name the columns for clarity
dfRoll.columns = ['min', 'mean', 'max', 't']
print('Rolling min, max & mean:')
print(dfRoll.head(), '\n')

# Final feature selection for this is to use an expanding window
window = temps.expanding()
# Get the min, mean & max values of the expanding window
dataframeExp = pd.concat([window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)
dataframeExp.columns = ['min', 'mean', 'max', 't']
print('Expanding Window:')
print(dataframeExp.head(), '\n')