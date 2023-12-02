'''
This is a program that runs simple descriptive statistics on the Daily Female Births
dataset and outputs the data as a line graph.
'''

from pandas import read_csv
from matplotlib import pyplot

# Load birth data using read_csv
series = read_csv('daily-total-female-births.csv', header=0, parse_dates=[0], index_col=0)
# Convert from dataframe to series
series = series.squeeze()
print(type(series), '\n')
# Look at first 10 observations in series
print(series.head(10), '\n')
# Get number of observations
print(series.size, '\n')
# Get only the observations from the month of January
print(series['1959-01'], '\n')
# Get the descriptive statistics of the series
print(series.describe(), '\n')
# Output the series as a line graph
pyplot.plot(series)
pyplot.show()
