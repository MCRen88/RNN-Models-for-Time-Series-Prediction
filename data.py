import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import tensorflow as tf

class CurrencyData:

    def __init__(self, filepath, datasize, OHCL, minmaxrange):
        self.dataset = np.array([pd.read_csv(filepath).values[-datasize:, OHCL]], dtype="f").T
        self.scaler = MinMaxScaler(feature_range = minmaxrange)

    def get_dataset(self):
        return self.dataset

    def normalized_data(self):
        self.minmax = self.scaler.fit_transform(self.dataset)
        return self.minmax

    def ExpMovingAverage(self, window):
        values = self.normalized_data()[:, 0]
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a =  np.convolve(values, weights, mode='full')[:len(values)]
        a[:window] = a[window]
        return a

    def inverse_data(self, data):
        return self.scaler.inverse_transform(data)

    
class Dataset:
    '''  Create a Sliding Window Tensorflow Dataset for Traning, Validation and Testing '''

    def __init__ (self, currency_data, forecast, window):
        self.data = currency_data.ExpMovingAverage(8).astype("float32")
        self.forecast_size = forecast
        self.window_size = window 
        self.traning = None
        self.forcasting = None

    def traning_data(self):
        train = tf.data.Dataset.from_tensor_slices(self.data[:-self.forecast_size])
        return  train.window(size = self.window_size, shift = 1 , drop_remainder= True).flat_map(
            lambda x: x.batch(self.window_size))

    def forecasting_data(self):
        forecast = tf.data.Dataset.from_tensor_slices(self.data[self.window_size:])
        return forecast.window(size = self.forecast_size, shift = 1, drop_remainder=True).flat_map(
            lambda x: x.batch(self.forecast_size))

    def zip_data(self):
        return tf.data.Dataset.zip((self.traning_data(), self.forecasting_data()))
