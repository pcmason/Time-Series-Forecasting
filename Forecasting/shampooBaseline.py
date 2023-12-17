'''
 An example implementing a persistence algorithm (using t-1 value to predict t value) to 
 create a baseline assessment for model performance. Use the Shampoo Sales dataset which 
 monthly data for shampoo sales over 3 years. 
'''
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error


# Create a parsing method for the dataset
def parser(x):
   # Set year to 1900 and make it clear data is in year-month description
   return pd.to_datetime('190'+x, format='%Y-%m')


# Define the persistence model (predict t as t-1 value)
def model_persistence(x):
   return x


# Download dataset
shampSeries = pd.read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
# Ensure this is a series instead of a dataframe
shampSeries = shampSeries.squeeze()

# Output the time series dataset as a graph
shampSeries.plot()
pyplot.show()

# Create a lagged dataset
values = pd.DataFrame(shampSeries.values)
# Use shift to get t-1 values
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't']
print('Head of Lagged Dataframe:')
print(dataframe.head())

# Split into train & test split
x = dataframe.values
# First 66% of dataset is training set, rest is test set
train_size = int(len(x) * 0.66)
train, test = x[1:train_size], x[train_size:]
# Split train & test split into input and output variables
train_x, train_y = train[:, 0], train[:, 1]
test_x, test_y = test[:, 0], test[:, 1]

# Make predictions using walk-forward validation
predictions = list()
for x in test_x:
   yhat = model_persistence(x)
   predictions.append(yhat)
# Get the mean squared error of the baseline (persistence) algorithm
test_score = mean_squared_error(test_y, predictions)
print('\nTest MSE: %.3f' % test_score)

# Plot predictions & expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [z for z in test_y])
pyplot.plot([None for i in train_y] + [z for z in predictions])
pyplot.show()