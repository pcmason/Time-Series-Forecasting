'''
In this program implement the ARIMA (AutoRegressive Integrated Moving Average) model on the Shampoo Sales dataset.
'''
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas.plotting import autocorrelation_plot


# Create a parsing method for the dataset
def parser(x):
   # Set year to 1900 and make it clear data is in year-month description
   return pd.to_datetime('190'+x, format='%Y-%m')


shamSeries = pd.read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
shamSeries = shamSeries.squeeze()
#print(shamSeries.head())

# Show the autocorrelation for the shampoo sales data
autocorrelation_plot(shamSeries)
pyplot.show()

shamSeries.index = shamSeries.index.to_period('M')
# Fit the model, set p=5, d=1, & q=0
model = ARIMA(shamSeries, order=(5, 1, 0))
model_fit = model.fit()
# Summary of the fitted model
print('\n', model_fit.summary())
# Line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# Density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# Summary stats of residuals
print('\n', residuals.describe(), '\n')

# Split shampoo dataset into train and test sets
shamp = shamSeries.values
# Train will be 66% of data with test as the rest
size = int(len(shamp) * 0.66)
train, test = shamp[0:size], shamp[size:len(shamp)]
# Manually keep track of all observations
history = [x for x in train]
predictions = list()
# Walk-forward validation
for t in range(len(test)):
    model2 = ARIMA(history, order=(5,1,0))
    model_fit2 = model2.fit()
    output = model_fit2.forecast()
    # Get the prediction & add it to the predictions list
    yhat = output[0]
    predictions.append(yhat)
    # Get the actual observed value and add it to the history list
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# Evaluate the forecasts with RMSE
rmse = sqrt(mean_squared_error(test, predictions))
print('\nTest RMSE: %.3f' % rmse)
# Plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()