import tensorflow as tf
import data
import parameters as pa

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dataset
eurusddata = data.TimeSeriesData(pa.FILE_NAME, pa.TRANING_DATA_SIZE, 
pa.OHLC, pa.FEATURE_RANGE)
traning_data = data.Dataset(eurusddata.normalized_data(), pa.FORECAST_SIZE,
 pa.WINDOW_SIZE, pa.FORECAST_SIZE)

iterator = traning_data.zip_data().batch(2).make_one_shot_iterator()

print(eurusddata.normalized_data().shape)