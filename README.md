# RNN Models for Time Series Prediction
Implementation of different RNN architectures for Time Series Prediction for Currency Markets. 

In this project I experimented with different architectures of Recurrent Neural Networks (RNNs) for Time Series Prediction. The data I used consist of Daily EURUSD prices. The architectures, I will be implementing and comparing are:

1.	LSTM with Residual Connections

![alt text](https://github.com/vinit97/Attention-Based-RNN-for-Time-Series-Prediction/blob/master/Pics/lstm_residual.png)
2.	Attention Based LSTM

![alt text](https://github.com/vinit97/Attention-Based-RNN-for-Time-Series-Prediction/blob/master/Pics/lstm_attention.png)

##### Read the Prediction Report for more information. 
##### Note: Orange is prediction, Blue is real data
### 1. LSTM with Residual Connections

![alt text](https://github.com/vinit97/Attention-Based-RNN-for-Time-Series-Prediction/blob/master/Pics/residual_pred.png)

Results: MAE: 0.0803 MAPE: 8.5362 %

### 2. Attention Based LSTM

![alt text](https://github.com/vinit97/Attention-Based-RNN-for-Time-Series-Prediction/blob/master/Pics/attention_pred.png)

Results: MAE: 0.02462221 MAPE: 1.73296783 %

### Example of Attention Weights during Prediction:

![alt text](https://github.com/vinit97/Attention-Based-RNN-for-Time-Series-Prediction/blob/master/Pics/attention_weight.png)
