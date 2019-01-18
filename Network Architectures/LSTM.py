import tensorflow as tf

def lstm(lstm_size, num_layers, dropout=False, output_prob=0.5, input_prob=1.0):
    return tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_size, dropout, output_prob, input_prob) for _ in range(num_layers)])
    
def lstm_cell(lstm_size, dropout, output_prob, input_prob):
    cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
    
    if(dropout==True):
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_prob, output_keep_prob=output_prob)
        
    return cell

def _rnn_reformat(x, input_dims, n_steps):
    """
    This function reformat input to the shape that standard RNN can take. 
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # permute batch_size and n_steps
    x_ = tf.transpose(x, [1, 0, 2])
    # reshape to (n_steps*batch_size, input_dims)
    x_ = tf.reshape(x_, [-1, input_dims])    
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = tf.split(x_, n_steps, 0)
    
    return x_reformat

if __name__ == "__main__":
    print("Debug")