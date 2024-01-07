'''
This file implements the grid search method for finding the best parameters for the ARIMA model (p, d, & q)
used by evaluating predictions with each parameter set within a range by the user and by calculating the RMSE.
'''

import warnings
from math import sqrt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Evaluate an ARIMA model for a given order (p, d, q)
def evaluate_arima_model(x, arima_order):
    # Prepare train & test set
    train_size = int(len(x) * 0.66)
    train, test = x[0:train_size], x[train_size:]
    history = [x for x in train]
    # Make predictions
    predictions = list()
    for t in range(len(test)):
        # Create a new ARIMA model for each time step
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # Calculate out of sample RMSE
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


# Evaluate combinations of p, d & q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    # Ensure the inputs are floating point values
    dataset = dataset.astype('float32')
    # Store the best score and combo of p, d & q
    best_score, best_cfg = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                # Must use an exception as often times this method can fail, but do not want to crash the whole program
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s RMSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# Define a datetime parser to give the shampoo sales time series an absolute year component
def parser(x):
    return pd.to_datetime('190'+x, format='%Y-%m')


# Now set the range of p, d & q values to be evaluated
p_values = [0, 1, 2, 4, 6, 8]
d_values = range(0, 2)
q_values = range(0, 2)
# Turn of warnings as they can cause a lot of unnecessary noise
warnings.filterwarnings('ignore')

# Now import the shampoo dataset as a series
shampSeries = pd.read_csv('shampoo.csv', header=0, index_col=0, parse_dates=True, date_parser=parser)
shampSeries = shampSeries.squeeze()

# Evaluate the p, q & d values on the shampoo sales time series
#print('\nShampoo Grid Search:\n')
#evaluate_models(shampSeries.values, p_values, d_values, q_values)

# Now import the Daily Female Births dataset
birthSeries = pd.read_csv('daily-female-births.csv', header=0, index_col=0, parse_dates=True)
birthSeries = birthSeries.squeeze()

# Evaluate the p, q & d values on the Daily Female Births time series
print('\nBirths Grid Search:\n')
evaluate_models(birthSeries.values, p_values, d_values, q_values)