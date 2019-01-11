#import numpy as np 
import tensorflow as tf

import data
import parameters as pa

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dataset
traning_data = data.Dataset(data.CurrencyData(pa.FILE_NAME, pa.DATA_SIZE, pa.OHLC, pa.FEATURE_RANGE), pa.FORECAST_SIZE, pa.WINDOW_SIZE)
iterator = traning_data.zip_data().make_one_shot_iterator()
tf_data = iterator.get_next()

# parameters 
Num_Layers = 1
lstm_size = 64


'''def lstm_cell():
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    return tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=1.0 , output_keep_prob=1.0,)

stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
    [lstm_cell  for _ in range(Num_Layers)])
'''

lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)

def _rnn_reformat(x, input_dims, n_steps):
    x_ = tf.transpose(x, [1, 0, 2])

    # reshape to (n_steps*batch_size, input_dims)
    x_ = tf.reshape(x_, [-1, input_dims])

    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = tf.split(x_, n_steps, 0)
    
    return x_reformat


# GRAPH
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, pa.WINDOW_SIZE, 1])
y = tf.placeholder(tf.float32, [None, pa.FORECAST_SIZE])

global_step = tf.Variable(0, name='global_step', trainable=False)

weights = tf.Variable(tf.random_normal(shape=[lstm_size, pa.FORECAST_SIZE]))
bias = tf.Variable(tf.random_normal(shape=[pa.FORECAST_SIZE]))

outputs, _states = tf.nn.static_rnn(lstm, _rnn_reformat(x,1,pa.WINDOW_SIZE)[0], dtype=tf.float32)

pred, state_size = tf.matmul(outputs, weights) + bias, lstm.state_size
cost = tf.reduce_mean(tf.losses.mean_squared_error(labels = y, predictions = pred))






