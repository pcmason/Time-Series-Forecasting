'''
This program rescales the data from the Minimum Daily Temperature dataset using
standardization and normalization techniques.
'''

from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot
from math import sqrt

# Load dataset and print first 5 rows
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
print(series.head())
# Prepare data for rescaling
values = series.values
values = values.reshape((len(values), 1))

# Normalize time series data
print('Normalize the dataset\n')
# Train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
# Normalize dataset & print the first 5 rows
normalized = scaler.transform(values)
print('Normalized Values:\n')
for i in range(5):
    print(normalized[i])
# Print added to make output better looking
print()
# Inverse transform and print first 5 rows
inversed = scaler.inverse_transform(normalized)
print('Original Values:\n')
for i in range(5):
    print(inversed[i])
# Print added to make output better looking
print()

# Plot a histogram of the temperature data
series.hist()
pyplot.show()

# Standardize time series data
print('Standardize the dataset\n')
# Train the standardization
standScaler = StandardScaler()
standScaler = standScaler.fit(values)
print('Mean: %f, StdDev: %f' % (standScaler.mean_, sqrt(standScaler.var_)))
# Standardize the dataset & print first 5 rows
print('Standardized Values:\n')
standardized = standScaler.transform(values)
for i in range(5):
    print(standardized[i])
# Print added to make output better looking
print()
# Inverse transform and print original first 5 values
inversedStand = standScaler.inverse_transform(standardized)
print('Original Values:\n')
for i in range(5):
    print(inversedStand[i])
