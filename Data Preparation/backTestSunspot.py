'''
This program uses 3 backtesting methods (train-test split, multiple train-test splits, and walk forward validation)
on the Monthly Sunspot dataset to see which method of splitting works best for time series datasets
'''
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Load in the Monthly Sunspot dataset as a series
sunSeries = pd.read_csv('sunspots.csv', header=0, index_col=0)
#print(sunSeries.head())

# Plot the data
sunSeries.plot()
pyplot.show()

# Split the dataset using one train-test split
x = sunSeries.values
#print(x)
# Use a split of 66-34
train_size = int(len(x) * 0.66)
# Now just split the data based on the split point
train, test = x[0:train_size], x[train_size:len(x)]
print('Normal train-test split:')
print('Observations: %d' % (len(x)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)), '\n')

# Now plot the training & test sets using different colors
graphTest = [None for i in train] + [z for z in test]
graphTest = np.asarray(graphTest, dtype='object')

# Create the test list for this dataset, this is probably not the most efficient method but does work
for i in range(len(graphTest)):
    # Skip None values
    if graphTest[i] == None:
        continue
    else:
        # Else convert the 1D array into a float value
        graphTest[i] = graphTest[i][0].astype(float)

pyplot.plot(train)
pyplot.plot(graphTest)
pyplot.show()

# Now implement repeating train-test splits
splits = TimeSeriesSplit(n_splits=3)
#pyplot.figure(1)
index = 1
count = 1
print('Multiple train-test splits:')
for train_index, test_index in splits.split(x):
    train = x[train_index]
    test = x[test_index]
    print('Split #: %d' % count)
    count += 1
    print('Observations: %d' % (len(train) + len(test)))
    print('Training Observations: %d' % (len(train)))
    print('Testing Observations: %d' % (len(test)))
    #pyplot.subplot(310 + index)
    #pyplot.plot(train)
    #pyplot.plot([None for i in train] + [x for x in test])
    index+= 1
#pyplot.show()
print()
# Finally use walk forward validation to split the data
# Min # of observations needed to train model
n_train = 500
n_records = len(x)
print('Each model for walk forward validation:')
for i in range(n_train, n_records):
    train, test = x[0:i], x[i:i+1]
    print('Train=%d, test=%d' % (len(train), len(test)))

