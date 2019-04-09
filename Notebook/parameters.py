from enum import Enum
import os

### CONSTANTS, HYPERPARAMETERS AND SHARED FUNCTIONS

# select OHLC data
OHLC = 5

# Traning From 1971 - 2016
TRANING_DATA_SIZE = 11247

# data info
FILE_NAME = 'EURUSDDAILY.csv'
FEATURE_RANGE = (-1,1)
WINDOW_SIZE = 30
FORECAST_SIZE = 5
EXPONENTIAL_AVERAGE = 8

#LSTM settings 
BATCH_SIZE = 30
LSTM_CELL_SIZE = 64


# optimization settings
EPOCHS = 100
MIN_EPOCHS = 40
L2_REGULARIZATION = 0.001
DROPOUT = 0.5