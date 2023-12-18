'''
This program uses basic summary statistics and the Augmented Dickey-Fuller Test to determine if the Daily Female
Births & Airline Passengers datasets are stationary or non-stationary time series datasets.
'''

import pandas as pd
from numpy import log
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot


# Load in the Daily Female Births dataset
birthSeries = pd.read_csv('daily-female-births.csv', header=0, index_col=0)
# Show a graph of the dataset
birthSeries.plot()
pyplot.show()
# Create a histogram for the dataset
birthSeries.hist()
pyplot.show()

# Split time series into 2 sequences
births = birthSeries.values
split1 = round(len(births) / 2)
b1, b2 = births[0:split1], births[split1:]
# Calculate the summary statistics for each group and output them
mean1, mean2 = b1.mean(), b2.mean()
var1, var2 = b1.var(), b2.var()
print('\n Summary Statistics for births data: \n')
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

# Perform augmented dickey-fuller test on female births dataset
bResult = adfuller(births)
print('\n Augmented Dickey-Fuller Test for births data: \n')
print('ADF Statistic: %f' % bResult[0])
print('p-value: %f' % bResult[1])
print('Critical Values:')
for key, value in bResult[4].items():
    print('\t%s: %.3f' % (key, value))


# Load in the Airline Passengers dataset
airSeries = pd.read_csv('airline-passengers.csv', header=0, index_col=0)
# Show graph of airline passengers dataset
airSeries.plot()
pyplot.show()
# Show data as a histogram
airSeries.hist()
pyplot.show()

# Split into 2 groups
air = airSeries.values
split2 = round(len(air) / 2)
a1, a2 = air[0:split2], air[split2:]
# Calculate and output summary statistics for the 2 groups
airMean1, airMean2 = a1.mean(), a2.mean()
airVar1, airVar2 = a1.var(), a2.var()
print('\n Summary Statistics for airlines data: \n')
print('mean1=%f, mean2=%f' % (airMean1, airMean2))
print('variance1=%f, variance2=%f' % (airVar1, airVar2))

# Run augmented dickey-fuller test on the airline passenger dataset
aResult = adfuller(air)
print('\n Augmented Dickey-Fuller Test for airline data: \n')
print('ADF Statistic: %f' % aResult[0])
print('p-value: %f' % aResult[1])
print('Critical Values:')
for key, value in aResult[4].items():
    print('\t%s: %.3f' % (key, value))


# Use a log transform on the airline passenger dataset
logAir = log(air)
# Plot the new histogram and data for the log airline passenger data
pyplot.hist(logAir)
# Histogram makes it appear as though this is stationary
pyplot.show()
pyplot.plot(logAir)
# Actual plot of the data shows that this is not stationary
pyplot.show()

# Calculate the mean & std deviation for the logged dataset
split3 = round(len(logAir) / 2)
log1, log2 = logAir[0:split3], logAir[split3:]
# Get and output summary statistics for logged dataset
lMean1, lMean2 = log1.mean(), log2.mean()
lVar1, lVar2 = log1.var(), log2.var()
print('\n Summary Statistics for logged airlines data: \n')
print('mean1=%f, mean2=%f' % (lMean1, lMean2))
print('variance1=%f, variance2=%f' % (lVar1, lVar2))

# Run the augmented dickey-fuller test on the logged dataset
lResult = adfuller(logAir)
print('\n Augmented Dickey-Fuller Test for logged data: \n')
print('ADF Statistic: %f' % lResult[0])
print('p-value: %f' % lResult[1])
print('Critical Values:')
for key, value in lResult[4].items():
    print('\t%s: %.3f' % (key, value))


