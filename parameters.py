from enum import Enum
import os

### CONSTANTS, HYPERPARAMETERS AND SHARED FUNCTIONS

# select OHLC data
OHLC = 5
DATA_SIZE = 10000

# data info
FILE_NAME = './data/EURUSDDAILY.csv'
FEATURE_RANGE = (-1,1)
WINDOW_SIZE = 100
FORECAST_SIZE = 30

#LSTM settings 
BATCH_SIZE = 30
LSTM_CELL_SIZE = 64


# optimization settings
EPOCHS = 100
MIN_EPOCHS = 40
L2_REGULARIZATION = 0.001
DROPOUT = 0.5