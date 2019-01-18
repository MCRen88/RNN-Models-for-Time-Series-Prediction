import tensorflow as tf
import data
import parameters as pa

tf.enable_eager_execution()
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dataset
traning_data = data.Dataset(data.CurrencyData(pa.FILE_NAME, pa.DATA_SIZE, pa.OHLC, pa.FEATURE_RANGE), pa.FORECAST_SIZE, pa.WINDOW_SIZE)
iterator = traning_data.zip_data().make_one_shot_iterator()

#cost  = tf.losses.mean

for i in iterator:
    print(i)